from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from .faiss_store import FaissIndex
from .models import ProfileArtifacts, ZoteroItem
from .settings import Settings
from .storage import ProfileStorage
from .utils import json_dumps, utc_now
from .vectorizer import TextVectorizer
from .problem_ranking import build_problem_profiles

logger = logging.getLogger(__name__)

# Bump when the profile.json layout changes in a way a consumer must notice.
# 1 -> 2 added the manifest block (schema_version, vector_dim, derived_fingerprint).
PROFILE_SCHEMA_VERSION = 2


class ProfileBuilder:
    def __init__(
        self,
        base_dir: Path | str,
        storage: ProfileStorage,
        settings: Settings,
        vectorizer: TextVectorizer | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.storage = storage
        self.settings = settings
        self.vectorizer = vectorizer or TextVectorizer.from_settings(settings)
        # The signature must name the encoder that actually ran, not the one config
        # asks for. When EMBEDDING_API_KEY is missing, build_vectorizer() silently
        # falls back to a 384-dim local model; stamping the bundle with the remote
        # 1024-dim signature produced a profile that passed the workflow's
        # compatibility gate and then queried a 384-dim index with 1024-dim vectors.
        self.embedding_signature = (
            getattr(self.vectorizer, "signature", None)
            or self.settings.embedding.cache_signature()
        )
        self.artifacts = ProfileArtifacts(
            sqlite_path=str(self.base_dir / "data" / "profile.sqlite"),
            faiss_path=str(self.base_dir / "data" / "faiss.index"),
            profile_json_path=str(self.base_dir / "data" / "profile.json"),
        )

    def run(self) -> ProfileArtifacts:
        items = [item for item in self.storage.iter_items() if item.title.strip()
                 and item.raw.get("data", {}).get("itemType") not in {"note", "attachment", "annotation"}]
        if not items:
            raise RuntimeError("No items found in storage; run ingest before building profile.")

        vectors = self._vectors_for(items)

        logger.info("Building FAISS index")
        index, order = FaissIndex.from_vectors(vectors)
        index.save(self.artifacts.faiss_path)

        profile_summary = self._summarize(items, vectors)
        json_path = Path(self.artifacts.profile_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_dumps(profile_summary, indent=2), encoding="utf-8")
        logger.info("Wrote profile summary to %s", json_path)
        return self.artifacts

    def _vectors_for(self, items: List[ZoteroItem]) -> np.ndarray:
        """Embed the library, reusing cached vectors for unchanged items.

        The whole library used to be re-encoded on every run even though the
        vectors were already being written to SQLite and never read back. With a
        remote encoder that is also a per-run bill, so the cache is keyed on the
        model signature and the Zotero item version: change the model and
        everything is recomputed, edit one paper and only that paper is.
        """
        cached = self.storage.cached_embeddings(self.embedding_signature)
        separator = self.vectorizer.text_separator

        pending = [item for item in items if item.key not in cached]
        if pending:
            logger.info(
                "Embedding %d of %d library items (%d reused from cache)",
                len(pending), len(items), len(items) - len(pending),
            )
            fresh = self.vectorizer.encode(
                [item.content_for_embedding(separator) for item in pending]
            )
            self.storage.set_embeddings(
                [(item.key, vector.tobytes(), self.embedding_signature, item.version)
                 for item, vector in zip(pending, fresh)]
            )
            for item, vector in zip(pending, fresh):
                cached[item.key] = vector.tobytes()
        else:
            logger.info("Embedding cache hit for all %d library items", len(items))

        dimension = len(next(iter(cached.values()))) // 4 if cached else 0
        vectors = np.vstack(
            [np.frombuffer(cached[item.key], dtype=np.float32) for item in items]
        ) if items else np.zeros((0, dimension), dtype=np.float32)
        return vectors

    def _summarize(self, items: List[ZoteroItem], vectors: np.ndarray) -> dict:
        authors = Counter()
        venues = Counter()
        for item in items:
            authors.update(item.creators)
            venue = item.raw.get("data", {}).get("publicationTitle")
            if venue:
                venues.update([venue])

        centroid = np.mean(vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        top_authors = [{"author": k, "count": v} for k, v in authors.most_common(20)]
        top_venues = [{"venue": k, "count": v} for k, v in venues.most_common(20)]

        return {
            # --- manifest: what a consumer must check before trusting the rest ---
            "schema_version": PROFILE_SCHEMA_VERSION,
            "generated_at": utc_now().isoformat(),
            "item_count": len(items),
            "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
            "model": self.vectorizer.model_name,
            "embedding_signature": self.embedding_signature,
            # Fingerprint of the config the DERIVED layer below was computed from, so a
            # facet edit is detectable without re-uploading a 27 MB bundle.
            "derived_fingerprint": self.settings.research.derived_fingerprint(),
            # --- base layer: needs the local Zotero library and the encoder ---
            "centroid": centroid.tolist(),
            "top_authors": top_authors,
            "top_venues": top_venues,
            "index_items": [{"title": item.title, "doi": item.doi} for item in items],
            # --- derived layer: recomputable from the stored vectors + current config ---
            "problem_profiles": build_problem_profiles(items, vectors, self.settings.research,
                                                       getattr(self, "feedback_entries", ())),
        }


def verify_profile(base_dir: Path | str, settings: Settings) -> List[str]:
    """Check a downloaded profile against itself and against the current config.

    Returns a list of problems; empty means consistent. The bundle mixes several data
    layers with different update conditions, and the only check that existed compared
    one of them -- the encoder signature -- so a bundle could be internally
    inconsistent (index shorter than the item list, vectors of the wrong width) or
    carry centroids from a superseded facet definition and still be accepted.
    """
    base = Path(base_dir)
    problems: List[str] = []
    json_path = base / "data" / "profile.json"
    if not json_path.exists():
        return [f"{json_path} is missing"]

    import json as _json

    profile = _json.loads(json_path.read_text(encoding="utf-8"))
    version = profile.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        problems.append(
            f"profile schema_version is {version!r}, this code writes "
            f"{PROFILE_SCHEMA_VERSION!r}; rebuild with `profile --local --bundle`"
        )

    declared = int(profile.get("item_count") or 0)
    listed = len(profile.get("index_items") or [])
    if declared != listed:
        problems.append(f"item_count says {declared} but index_items has {listed} entries")

    storage = ProfileStorage(base / "data" / "profile.sqlite")
    try:
        rows = storage.connect().execute("SELECT COUNT(*) FROM items").fetchone()[0]
    except Exception as exc:  # noqa: BLE001 - a missing/corrupt sqlite is a problem to report
        problems.append(f"cannot read profile.sqlite: {exc}")
        rows = None
    finally:
        storage.close()
    if rows is not None and declared and rows < declared:
        problems.append(f"profile.sqlite holds {rows} items, fewer than the declared {declared}")

    faiss_path = base / "data" / "faiss.index"
    if faiss_path.exists():
        index = FaissIndex.load(str(faiss_path))
        if declared and getattr(index, "ntotal", declared) != declared:
            problems.append(
                f"FAISS index holds {index.ntotal} vectors but the profile declares {declared}"
            )
        dim = getattr(index, "dim", None) or getattr(index, "dimension", None)
        want = profile.get("vector_dim") or settings.embedding.dimensions
        if dim and want and int(dim) != int(want):
            problems.append(f"FAISS index is {dim}-dim but the profile/config expect {want}")
    else:
        problems.append(f"{faiss_path} is missing")

    expected_sig = settings.embedding.cache_signature()
    actual_sig = profile.get("embedding_signature")
    if actual_sig != expected_sig:
        problems.append(
            f"profile was built with {actual_sig!r} but config/embedding.yaml now says "
            f"{expected_sig!r}; vectors from different models are not comparable"
        )
    return problems


def rederive_profile(base_dir: Path | str, settings: Settings) -> dict:
    """Recompute the derived layer from the stored vectors and the CURRENT config.

    The derived layer is just the facet centroids: which library papers belong to each
    research direction, averaged. That depends only on config/research.yaml and on
    vectors that are already in profile.sqlite -- no Zotero access, no encoder call, no
    bill. Previously the only way to pick up a facet edit was to rebuild the whole
    profile locally and re-upload 27 MB, which is a manual step that silently did not
    happen: on 2026-09-19 the published bundle still carried centroids built from a
    microstructure-flavoured facet after the facet had been rewritten.

    Returns the rewritten profile dict.
    """
    import json as _json

    base = Path(base_dir)
    json_path = base / "data" / "profile.json"
    profile = _json.loads(json_path.read_text(encoding="utf-8"))
    was = profile.get("derived_fingerprint")
    now = settings.research.derived_fingerprint()

    storage = ProfileStorage(base / "data" / "profile.sqlite")
    try:
        items = [item for item in storage.iter_items() if item.title.strip()]
        keyed = {key: vector for key, vector in storage.fetch_all_embeddings()}
    finally:
        storage.close()

    items = [item for item in items if item.key in keyed]
    if not items:
        raise RuntimeError("No embedded items in profile.sqlite; cannot rederive.")
    vectors = np.vstack([np.frombuffer(keyed[item.key], dtype=np.float32) for item in items])

    profile["problem_profiles"] = build_problem_profiles(items, vectors, settings.research)
    profile["derived_fingerprint"] = now
    profile["derived_at"] = utc_now().isoformat()
    json_path.write_text(json_dumps(profile, indent=2), encoding="utf-8")
    if was == now:
        logger.info("Derived profile already matched the current facets (%s); recomputed anyway", now)
    else:
        logger.info("Recomputed facet centroids for the current config: %s -> %s", was, now)
    for key, value in sorted((profile.get("problem_profiles") or {}).items()):
        logger.info("  %-22s %5d  %s", key, value.get("count", 0), value.get("name", key))
    return profile


__all__ = ["ProfileBuilder", "PROFILE_SCHEMA_VERSION", "rederive_profile", "verify_profile"]
