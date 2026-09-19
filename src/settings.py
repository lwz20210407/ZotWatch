from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


class ZoteroApiConfig(BaseModel):
    user_id: str = Field(..., alias="user_id")
    api_key_env: str = Field("ZOTERO_API_KEY", alias="api_key_env")
    page_size: int = 100
    polite_delay_ms: int = 200

    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable '{self.api_key_env}' is required for Zotero API access."
            )
        return key


class ZoteroLocalConfig(BaseModel):
    """Direct access to zotero.sqlite, for building the profile from the whole library.

    The Web API only exposes items synced to zotero.org; anything with
    items.version = 0 is invisible to it.
    """

    data_dir: str = ""

    def resolved_dir(self) -> Optional[str]:
        return os.path.expandvars(os.path.expanduser(self.data_dir)) if self.data_dir else None


class ZoteroConfig(BaseModel):
    mode: str = "api"
    api: ZoteroApiConfig = Field(default_factory=ZoteroApiConfig)
    local: ZoteroLocalConfig = Field(default_factory=ZoteroLocalConfig)

    @validator("mode")
    def validate_mode(cls, value: str) -> str:
        allowed = {"api", "bbt"}
        if value not in allowed:
            raise ValueError(f"Unsupported Zotero mode '{value}'. Allowed: {sorted(allowed)}")
        return value


class AltmetricConfig(BaseModel):
    enabled: bool = False
    api_key_env: Optional[str] = None

    def api_key(self) -> Optional[str]:
        if not self.enabled or not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


class OpenAlexConfig(BaseModel):
    enabled: bool = True
    mailto: str = "you@example.com"


class CrossRefConfig(BaseModel):
    enabled: bool = True
    mailto: str = "you@example.com"


class ArxivConfig(BaseModel):
    enabled: bool = True
    categories: List[str] = Field(default_factory=lambda: ["cs.LG"])


class BioRxivConfig(BaseModel):
    enabled: bool = True
    from_days_ago: int = 30


class MedRxivConfig(BaseModel):
    enabled: bool = False
    from_days_ago: int = 30


class PublicCandidatesApiConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://rbsfoisrcaxacwodbuzg.supabase.co/functions/v1"
    publishable_key: Optional[str] = None
    api_key_env: str = "SUPABASE_PUBLISHABLE_KEY"
    page_size: int = 200
    timeout_seconds: int = 30

    def api_key(self) -> str:
        if self.publishable_key:
            return self.publishable_key
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Either 'publishable_key' or environment variable '{self.api_key_env}' is required for the public candidate API."
            )
        return key


class SourcesConfig(BaseModel):
    window_days: int = 30
    page_size: int = Field(100, ge=1, le=200)
    query_max_pages: int = Field(2, ge=1, le=10)
    venue_max_pages: int = Field(5, ge=1, le=20)
    request_interval_seconds: float = Field(0.5, ge=0.0)
    tracked_venue_issns: Dict[str, str] = Field(default_factory=dict)
    queries: List[str] = Field(default_factory=list)
    tracked_venues: List[str] = Field(default_factory=list)
    include_keywords: List[str] = Field(default_factory=list)
    # Terms that name a mechanics-of-materials object. A semantically discovered
    # candidate must hit one of these before it may skip the keyword group sets.
    mechanics_anchor_keywords: List[str] = Field(default_factory=list)
    required_keyword_groups: List[List[str]] = Field(default_factory=list)
    required_any_group_sets: List[List[List[str]]] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)
    require_topic_match: bool = False
    public_api: PublicCandidatesApiConfig = Field(default_factory=PublicCandidatesApiConfig)
    openalex: OpenAlexConfig = Field(default_factory=OpenAlexConfig)
    crossref: CrossRefConfig = Field(default_factory=CrossRefConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    biorxiv: BioRxivConfig = Field(default_factory=BioRxivConfig)
    medrxiv: MedRxivConfig = Field(default_factory=MedRxivConfig)
    altmetric: AltmetricConfig = Field(default_factory=AltmetricConfig)


class EmbeddingConfig(BaseModel):
    # "model_name" is a natural field name here but collides with pydantic's
    # protected "model_" namespace.
    model_config = {"protected_namespaces": ()}

    provider: Literal["openai-compatible", "openai", "remote", "local"] = "openai-compatible"
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    base_url: str = "https://api.siliconflow.cn/v1"
    dimensions: Optional[int] = Field(1024, ge=8, le=8192)
    api_key_env: str = "EMBEDDING_API_KEY"
    timeout_seconds: float = Field(120.0, gt=0)
    local_fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_separator: str = "\n"
    neighbors: int = Field(5, ge=1, le=50)
    batch_size: int = Field(32, ge=1, le=256)

    def cache_signature(self) -> str:
        """Identity of the vector space, for invalidating stored embeddings.

        Vectors produced by different models or dimensions are not comparable, so
        a cached embedding is only reusable when this signature is unchanged.
        """
        return f"{self.provider}:{self.model_name}:{self.dimensions or 'native'}"


class TranslationConfig(BaseModel):
    """Chinese titles for English papers. Best-effort; failures keep the original."""

    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    base_url: str = "https://api.siliconflow.cn/v1"
    model_name: str = "Qwen/Qwen3-8B"
    api_key_env: str = "EMBEDDING_API_KEY"
    batch_size: int = Field(20, ge=1, le=60)
    timeout_seconds: float = Field(90.0, gt=0)


class ScoreWeights(BaseModel):
    similarity: float = 0.45
    recency: float = 0.15
    citations: float = 0.15
    altmetric: float = 0.10
    journal_quality: float = 0.08
    author_bonus: float = 0.02
    venue_bonus: float = 0.05

    def normalized(self) -> "ScoreWeights":
        total = sum(self.dict().values())
        if not total:
            raise ValueError("Score weights sum to zero; at least one positive weight is required.")
        normalized = {k: v / total for k, v in self.dict().items()}
        return ScoreWeights(**normalized)


class ScoreScales(BaseModel):
    """Saturation points that map unbounded raw signals onto [0, 1].

    Every scoring component must be bounded, otherwise a term with no upper limit
    (citations, in particular) silently dominates the weighted sum and the
    absolute label thresholds stop meaning anything.
    """

    recency_half_life_days: float = Field(21.0, gt=0.0)
    citation_saturation: float = Field(50.0, gt=0.0)
    altmetric_saturation: float = Field(100.0, gt=0.0)
    sjr_floor: float = Field(0.3, ge=0.0)
    sjr_ceiling: float = Field(4.0, gt=0.0)
    journal_unknown: float = Field(0.25, ge=0.0, le=1.0)

    @validator("sjr_ceiling")
    def ceiling_above_floor(cls, value: float, values: Dict[str, Any]) -> float:
        floor = values.get("sjr_floor", 0.0)
        if value <= floor:
            raise ValueError("sjr_ceiling must be greater than sjr_floor")
        return value


class Thresholds(BaseModel):
    must_read: float = 0.70
    consider: float = 0.45


class ResearchPriority(BaseModel):
    name: str
    required_groups: List[List[str]]
    multiplier: float = Field(1.0, gt=0.0, le=1.0)
    match_fields: Literal["title", "title_abstract"] = "title_abstract"

    @validator("required_groups")
    def validate_groups(cls, value: List[List[str]]) -> List[List[str]]:
        if not value or any(not group or any(not term.strip() for term in group) for group in value):
            raise ValueError("Research priority groups must contain nonempty terms.")
        return value


class ScoringConfig(BaseModel):
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    scales: ScoreScales = Field(default_factory=ScoreScales)
    # Retained for backward compatibility with older config files; the recency
    # component is now a continuous decay driven by scales.recency_half_life_days.
    decay_days: Dict[str, int] = Field(
        default_factory=lambda: {"fast": 30, "medium": 60, "slow": 180}
    )
    whitelist_authors: List[str] = Field(default_factory=list)
    whitelist_venues: List[str] = Field(default_factory=list)
    research_priorities: List[ResearchPriority] = Field(default_factory=list)
    default_priority_multiplier: float = Field(1.0, gt=0.0, le=1.0)


class TrackedAuthor(BaseModel):
    name: str
    openalex_ids: List[str]
    reason: str = ""
    enabled: bool = True
    institution_ids: List[str] = Field(default_factory=list)
    evidence_dois: List[str] = Field(default_factory=list)
    verified_on: str = ""

    @validator("openalex_ids")
    def validate_author_ids(cls, value: List[str]) -> List[str]:
        if not value or any(not re.fullmatch(r"A\d+", item) for item in value):
            raise ValueError("Tracked authors require explicit OpenAlex A-identifiers.")
        return list(dict.fromkeys(value))

    @validator("institution_ids")
    def validate_institution_ids(cls, value: List[str]) -> List[str]:
        if any(not re.fullmatch(r"I\d+", item) for item in value):
            raise ValueError("Institution guards require OpenAlex I-identifiers.")
        return value


class AuthorWatchConfig(BaseModel):
    enabled: bool = False
    max_pages: int = Field(3, ge=1, le=10)
    max_report_items: int = Field(20, ge=0, le=100)
    score_bonus: float = Field(0.03, ge=0, le=0.1)
    topic_keywords: List[str] = Field(default_factory=lambda: [
        "plasticity", "constitutive", "ductile fracture", "damage model", "damage evolution",
        "yield surface", "flow stress", "hardening", "fracture criterion", "strain localization",
        "应力状态", "本构", "塑性", "屈服", "损伤演化", "延性断裂",
    ])
    authors: List[TrackedAuthor] = Field(default_factory=list)


class Settings(BaseModel):
    zotero: ZoteroConfig
    sources: SourcesConfig
    scoring: ScoringConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    author_watch: AuthorWatchConfig = Field(default_factory=AuthorWatchConfig)
    citation_watch: "CitationWatchConfig" = Field(default_factory=lambda: CitationWatchConfig())
    research: "ResearchConfig" = Field(default_factory=lambda: ResearchConfig())
    network: "NetworkConfig" = Field(default_factory=lambda: NetworkConfig())


class CitationSeed(BaseModel):
    openalex_id: str
    title: str
    doi: str = ""
    reason: str = ""

    @validator("openalex_id")
    def valid_id(cls, value: str) -> str:
        if not re.fullmatch(r"W\d+", value):
            raise ValueError("Citation seeds require an OpenAlex W identifier")
        return value


class CitationWatchConfig(BaseModel):
    enabled: bool = False
    seeds: List[CitationSeed] = Field(default_factory=list)
    max_pages: int = Field(2, ge=1, le=5)
    max_requests: int = Field(40, ge=1, le=100)
    max_reference_candidates: int = Field(400, ge=1, le=2000)
    dynamic_seed_count: int = Field(5, ge=0, le=20)
    max_classic_items: int = Field(5, ge=0, le=30)
    min_similarity: float = Field(0.40, ge=0, le=1)
    score_bonus: float = Field(0.06, ge=0, le=0.1)


class ResearchFacet(BaseModel):
    id: str
    name: str
    terms: List[str]
    # Optional conjunction: the facet applies only when the text ALSO names one of
    # these. terms alone are OR-ed, which is right for conditions and phenomena but
    # wrong for a material object -- "LPBF 316L steel plasticity and ductile fracture"
    # matched the Ti-6Al-4V facet on the word LPBF, and because primary_problem is an
    # argmax over facet centroids the report then filed a steel paper under "增材 TC4".
    requires: List[str] = Field(default_factory=list)
    use: str
    verify: str
    semantic_query: str = ""
    collection_keys: List[str] = Field(default_factory=list)
    seed_dois: List[str] = Field(default_factory=list)


class ResearchConfig(BaseModel):
    def derived_fingerprint(self) -> str:
        """Hash of everything the derived profile layer reads from this config.

        The bundle's embedding_signature only names the encoder, so editing a facet's
        terms or name left the published profile carrying centroids built from the old
        definition -- and the workflow's compatibility check had no way to notice. On
        2026-09-19 that nearly wasted half a filtering change: the facet query was
        rewritten locally while the release asset still pointed at microstructure.

        Covers exactly the fields build_problem_profiles and facet_ids consume, so a
        cosmetic edit elsewhere in research.yaml does not force a needless rebuild.
        """
        import hashlib

        parts = []
        for facet in self.facets:
            parts.append("\x1f".join([
                facet.id, facet.name,
                "\x1e".join(facet.terms),
                "\x1e".join(facet.requires),
                "\x1e".join(facet.collection_keys),
                "\x1e".join(facet.seed_dois),
            ]))
        blob = "\x1d".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    enabled: bool = False
    feedback_max_adjustment: float = Field(0.08, ge=0, le=0.15)
    feedback_owner: str = ""
    feedback_repository: str = ""
    version_batch_size: int = Field(10, ge=0, le=30)
    proposal_min_papers: int = Field(2, ge=2, le=10)
    proposal_limit: int = Field(10, ge=0, le=30)
    facets: List[ResearchFacet] = Field(default_factory=list)
    # Off by default: a feature that has not proved useful should not run every week,
    # take up space in the report and spend budget just because the code exists.
    collaboration_analysis: bool = False
    semantic_enabled: bool = True
    semantic_min_similarity: float = Field(0.40, ge=0, le=1)
    diversity_pool: int = Field(100, ge=20, le=300)
    diversity_penalty: float = Field(0.20, ge=0, le=0.5)
    exploration_slots: int = Field(2, ge=0, le=5)
    semantic_backfill_items: int = Field(2, ge=0, le=5)


class NetworkConfig(BaseModel):
    openalex_anonymous_budget: float = Field(0.08, ge=0, le=0.1)
    openalex_key_budget: float = Field(0.8, ge=0, le=1)
    openalex_max_requests: int = Field(160, ge=1, le=1000)
    crossref_max_requests: int = Field(180, ge=1, le=1000)
    topic_queries_per_run: int = Field(40, ge=1, le=200)
    cache_hours: int = Field(48, ge=1, le=168)


Settings.model_rebuild()




def _expand_env_vars(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    if isinstance(data, str):
        return os.path.expandvars(data)
    return data

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data = _expand_env_vars(data)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


def load_settings(base_dir: Path | str) -> Settings:
    base = Path(base_dir)
    zotero_cfg = _load_yaml(base / "config" / "zotero.yaml")
    sources_cfg = _load_yaml(base / "config" / "sources.yaml")
    scoring_cfg = _load_yaml(base / "config" / "scoring.yaml")
    author_path = base / "config" / "authors.yaml"
    author_cfg = _load_yaml(author_path) if author_path.exists() else {}
    citation_path = base / "config" / "citations.yaml"
    citation_cfg = _load_yaml(citation_path) if citation_path.exists() else {}
    research_path = base / "config" / "research.yaml"
    research_cfg = _load_yaml(research_path) if research_path.exists() else {}
    network_path = base / "config" / "network.yaml"
    network_cfg = _load_yaml(network_path) if network_path.exists() else {}
    embedding_path = base / "config" / "embedding.yaml"
    embedding_cfg = _load_yaml(embedding_path) if embedding_path.exists() else {}
    translation_path = base / "config" / "translation.yaml"
    translation_cfg = _load_yaml(translation_path) if translation_path.exists() else {}
    return Settings(
        zotero=ZoteroConfig(**zotero_cfg),
        sources=SourcesConfig(**sources_cfg),
        scoring=ScoringConfig(**scoring_cfg),
        embedding=EmbeddingConfig(**embedding_cfg),
        translation=TranslationConfig(**translation_cfg),
        author_watch=AuthorWatchConfig(**author_cfg),
        citation_watch=CitationWatchConfig(**citation_cfg),
        research=ResearchConfig(**research_cfg),
        network=NetworkConfig(**network_cfg),
    )


__all__ = [
    "Settings",
    "load_settings",
    "ZoteroConfig",
    "SourcesConfig",
    "ScoringConfig",
    "ScoreScales",
    "EmbeddingConfig",
    "TranslationConfig",
]
