"""Full-text downloads as implicit feedback.

The owner does not click the feedback links (5 clicks in three weeks) and does not
add digest papers to Zotero straight away (0 of 89 additions since August had been
delivered first). What they do is download the full text of the papers they want,
into topic folders, and import them into Zotero in batches later. On 2026-10-05, 29
of the 97 papers delivered so far were already sitting in those folders. That is the
signal, and it costs the owner nothing.

Two halves:

* scan (on the owner's PC, from tools/refresh_profile.ps1): read the download folders
  and write data/download-signal.json -- English title, DOI when the file name is one,
  the topic folder, the date. No paths. It travels inside the private profile bundle.
* implicit_entries (in the weekly run): papers that were delivered and then
  downloaded become positive feedback entries. They join the ranking only; explicit
  feedback always wins, and they are never written into the persisted history, so a
  deleted download takes its signal with it next week.

The folders themselves are listed in data/download-roots.txt, which is not committed:
the repository is public and the paths name the owner's machine.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rapidfuzz import fuzz, process

from .lineage import norm_doi
from .utils import atomic_json

SIGNAL = Path("data") / "download-signal.json"
ROOTS = Path("data") / "download-roots.txt"
TITLE_MATCH = 92  # token_set_ratio; measured: genuine matches scored 92.7-100 on 2026-10-05

# The owner's renamer writes 【中文标题English Title】.pdf, sometimes behind a prefix such
# as 博士论文_2024_. The English title is what follows the LAST Chinese character inside
# the brackets: the Chinese half itself can contain Latin words ("考虑Lode相关的...").
_BRACKETED = re.compile(r"【([^【】]+)】")
_CJK = re.compile(r"[㐀-鿿　-〿＀-￯]")
_DOI_NAME = re.compile(r"^(10\.\d{4,9})_(.+)$")


def title_from_name(name: str) -> str:
    match = _BRACKETED.search(name)
    if not match:
        return ""
    inside = match.group(1)
    last = max((m.end() for m in _CJK.finditer(inside)), default=0)
    title = re.sub(r"\s+", " ", inside[last:]).strip(" -—–_")
    return title if len(title) >= 15 and re.match(r"[A-Za-z0-9]", title) else ""


def doi_from_name(stem: str) -> str:
    # ScanSCI with auto_rename off saves 10.1016/j.x.1 as 10.1016_j.x.1.pdf. The first
    # underscore is the slash; later ones were already in the DOI or stood for symbols.
    match = _DOI_NAME.match(stem)
    return norm_doi(f"{match.group(1)}/{match.group(2)}") if match else ""


def _seen(path: Path) -> str:
    try:  # long Windows paths need the \\?\ prefix for stat
        target = "\\\\?\\" + str(path.resolve()) if os.name == "nt" else str(path)
        return datetime.fromtimestamp(os.stat(target).st_mtime, timezone.utc).date().isoformat()
    except OSError:
        return ""


def scan(roots) -> dict:
    items, seen = [], set()
    for root in [Path(r) for r in roots]:
        if not root.is_dir():
            continue
        for dirpath, _, files in os.walk(root):
            folder = Path(dirpath).relative_to(root).parts[0] if Path(dirpath) != root else ""
            for name in files:
                if not name.lower().endswith(".pdf"):
                    continue
                stem = name[:-4]
                title, doi = title_from_name(stem), doi_from_name(stem)
                key = doi or title.casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                items.append({"title": title, "doi": doi, "folder": folder,
                              "seen": _seen(Path(dirpath) / name)})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "items": items}


def delivered_titles(state) -> dict:
    """casefolded title -> DOI for every delivered paper the history can name."""
    out = {}
    for doi, meta in (state.get("catalog") or {}).items():
        title = (meta or {}).get("title") if isinstance(meta, dict) else ""
        if title and norm_doi(doi).startswith("10."):
            out[re.sub(r"\s+", " ", html.unescape(title)).strip().casefold()] = norm_doi(doi)
    return out


def matches(signal, state) -> list:
    """Delivered papers found among the downloads: (doi, title, folder, seen)."""
    titles = delivered_titles(state)
    delivered = set(titles.values()) | {norm_doi(k) for k in (state.get("sent") or {}) if str(k).startswith("10.")}
    keys = list(titles)
    found = {}
    for item in signal.get("items", []):
        doi = norm_doi(item.get("doi"))
        if doi and doi in delivered:
            found.setdefault(doi, (doi, item.get("title") or "", item.get("folder", ""), item.get("seen", "")))
            continue
        title = (item.get("title") or "").casefold()
        if not title or not keys:
            continue
        best = process.extractOne(title, keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= TITLE_MATCH:
            found.setdefault(titles[best[0]], (titles[best[0]], item["title"], item.get("folder", ""), item.get("seen", "")))
    return list(found.values())


def implicit_entries(base_dir, state, config, rating="transferable"):
    """FeedbackEntry rows for delivered-then-downloaded papers, plus a summary."""
    from .research_features import FeedbackEntry, facet_ids

    path = Path(base_dir) / SIGNAL
    if not path.exists():
        return {"available": False, "entries": [], "matched": 0}
    signal = json.loads(path.read_text(encoding="utf-8"))
    found = matches(signal, state)
    entries = []
    for doi, title, _, _ in found:
        # One by one: a DOI the feedback model rejects must cost that paper, not the batch.
        # Facets come from the title alone -- the history keeps no abstracts.
        try:
            entries.append(FeedbackEntry(doi=doi, rating=rating,
                                         facets=facet_ids(SimpleNamespace(title=title, abstract=""), config)))
        except ValueError:
            continue
    folders = {}
    for _, _, folder, _ in found:
        folders[folder or "（根目录）"] = folders.get(folder or "（根目录）", 0) + 1
    return {"available": True, "entries": entries, "matched": len(found),
            "downloads": len(signal.get("items", [])), "scanned_at": signal.get("generated_at"),
            "folders": dict(sorted(folders.items(), key=lambda kv: -kv[1]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write data/download-signal.json from the download folders.")
    parser.add_argument("command", choices=["scan"])
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    roots_file = args.base_dir / ROOTS
    if not roots_file.exists():
        print(f"No {ROOTS}; nothing to scan (one folder per line enables the signal).")
        return
    roots = [line.strip() for line in roots_file.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    signal = scan(roots)
    atomic_json(args.base_dir / SIGNAL, signal)
    print(f"download signal: {len(signal['items'])} papers from {len(roots)} folder(s)")


if __name__ == "__main__":
    main()
