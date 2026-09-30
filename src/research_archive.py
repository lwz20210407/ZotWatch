"""Immutable private dossiers, explicit public summaries, and safe report backlinks."""
import argparse
import hashlib
import html
import json
import logging
import re
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .utils import atomic_json
from .aminer_source import clean_doi
from .research_dossier import export_dossier, safe_url
from .research_features import work_feedback_ids

logger = logging.getLogger(__name__)
ARCHIVE_ID = re.compile(r"r_[a-f0-9]{16}")
TOMBSTONE = '<!doctype html><meta charset="utf-8"><p>此专题公开概览已撤回。</p>'


def valid_paper_id(value):
    if not isinstance(value, str) or len(value) > 300:
        return False
    if value.startswith("doi:"):
        return clean_doi(value[4:]) == value[4:]
    return bool(re.fullmatch(r"aminer:[a-z0-9_-]{1,128}|snapshot:[a-f0-9]{20}", value))


def read_private_json(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Invalid private archive path or size")
    return json.loads(path.read_text("utf-8"))


@contextmanager
def publication_lock(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".publish.lock"
    token = uuid.uuid4().hex
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as f:
        f.write(token)
    try:
        yield
    finally:
        if lock.exists() and lock.read_text() == token:
            lock.unlink()


def validate_ledger(ledger):
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise ValueError("Unsupported dossier schema")
    if not isinstance(ledger.get("topic"), str) or len(ledger["topic"]) > 500:
        raise ValueError("Invalid topic")
    papers, evidence = ledger.get("papers"), ledger.get("evidence")
    if not isinstance(papers, list) or not 1 <= len(papers) <= 20 or not isinstance(evidence, list) or len(evidence) > 300:
        raise ValueError("Invalid dossier size")
    ids = []
    for paper in papers:
        if not isinstance(paper, dict) or not valid_paper_id(paper.get("work_id")):
            raise ValueError("Invalid paper identity")
        if not isinstance(paper.get("title"), str) or len(paper["title"]) > 2000:
            raise ValueError("Invalid paper title")
        if not isinstance(paper.get("fields"), dict) or len(paper["fields"]) > 20:
            raise ValueError("Invalid comparison fields")
        if paper.get("evidence_level") not in {"abstract", "abstract_slice", "local_text", "title_only"}:
            raise ValueError("Invalid evidence level")
        if not isinstance(paper.get("next_checks", []), list):
            raise ValueError("Invalid next checks")
        paper.setdefault("next_checks", [])
        if not all(isinstance(v, str) for v in paper["next_checks"]):
            raise ValueError("Invalid next checks")
        paper.setdefault("document_status", "not_supplied")
        paper.setdefault("url", "")
        if not isinstance(paper["url"], str) or not isinstance(paper["document_status"], str):
            raise ValueError("Invalid paper metadata")
        safe_url(paper["url"])  # Reject malformed URLs before creating any files.
        aliases = paper.get("external_ids") or {}
        if not isinstance(aliases, dict):
            raise ValueError("Invalid external identities")
        if aliases.get("doi") and clean_doi(aliases["doi"]) != aliases["doi"]:
            raise ValueError("Invalid DOI alias")
        if aliases.get("aminer") and (not isinstance(aliases["aminer"], str) or not re.fullmatch(r"[a-z0-9_-]{1,128}", aliases["aminer"])):
            raise ValueError("Invalid AMiner alias")
        for prefix in ("doi", "aminer"):
            if paper["work_id"].startswith(prefix + ":") and aliases.get(prefix) and paper["work_id"] != prefix + ":" + aliases[prefix]:
                raise ValueError("Conflicting paper identity")
        ids.append(paper["work_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate paper identity")
    evidence_ids = set()
    for row in evidence:
        if (not isinstance(row, dict) or row.get("work_id") not in ids or
                not re.fullmatch(r"e:[a-f0-9]{16}", str(row.get("id", ""))) or row["id"] in evidence_ids):
            raise ValueError("Invalid evidence identity")
        if not all(isinstance(row.get(k), str) for k in ("locator", "quote", "level")):
            raise ValueError("Invalid evidence text")
        evidence_ids.add(row["id"])
    for paper in papers:
        for field in paper["fields"].values():
            if not isinstance(field, dict) or field.get("status") not in {"unknown", "mentioned"}:
                raise ValueError("Invalid evidence field")
            if not isinstance(field.get("label"), str):
                raise ValueError("Invalid field label")
            if not isinstance(field.get("mentions"), list) or not all(isinstance(v, str) for v in field["mentions"]):
                raise ValueError("Invalid mention list")
            if (not isinstance(field.get("evidence_ids"), list) or
                    not all(isinstance(v, str) for v in field["evidence_ids"]) or
                    not set(field["evidence_ids"]) <= evidence_ids):
                raise ValueError("Unresolved evidence reference")
            for eid in field["evidence_ids"]:
                if next(e["work_id"] for e in evidence if e["id"] == eid) != paper["work_id"]:
                    raise ValueError("Evidence attached to the wrong paper")
    if not isinstance(ledger.get("citations"), list) or len(ledger["citations"]) > 100 or not isinstance(ledger.get("unresolved_citations"), list):
        raise ValueError("Invalid citation lists")
    for edge in ledger["citations"]:
        if (not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("relation") != "cites" or
                not all(isinstance(edge.get(k), str) for k in ("marker", "locator", "quote", "reference_locator", "reference_quote"))):
            raise ValueError("Invalid citation relation")
    if len(ledger["unresolved_citations"]) > 500 or not all(isinstance(row, dict) and
            all(isinstance(row.get(k, ""), str) for k in ("from", "locator", "reason")) for row in ledger["unresolved_citations"]):
        raise ValueError("Invalid unresolved citation records")
    documents = ledger.get("documents", {})
    if not isinstance(documents, dict) or not set(documents) <= set(ids) or not all(isinstance(v, dict) for v in documents.values()):
        raise ValueError("Invalid document inventory")
    return ledger


def content_hash(ledger):
    canonical = {k: v for k, v in ledger.items() if k not in {"created_at", "discovery"}}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def import_dossier(dossier_dir, archive_dir):
    dossier_dir = Path(dossier_dir).resolve()
    source = (dossier_dir / "evidence-ledger.json").resolve()
    if not source.is_relative_to(dossier_dir):
        raise ValueError("Ledger escapes its directory")
    if not source.is_file() or source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("A bounded evidence-ledger.json is required")
    ledger = validate_ledger(json.loads(source.read_text("utf-8")))
    checksum = content_hash(ledger)
    ident = "r_" + checksum[:16]
    root = Path(archive_dir).resolve()
    destination = (root / ident).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Archive path escapes its root")
    if destination.exists():
        existing = load_archive(root, ident)
        if content_hash(existing) != checksum:
            raise ValueError("Existing archive differs; refusing to overwrite")
        return {"id": ident, "path": str(destination), "reused": True}
    destination.mkdir(parents=True, exist_ok=False)
    # Never copy or execute an input HTML file; render from validated data.
    export_dossier(ledger, destination, archive_notice="归档导入仅校验结构与内容哈希，未重新读取原文核验；以下保留原证据包的状态。")
    atomic_json(destination / "manifest.json", {"schema_version": 1, "id": ident,
        "content_sha256": checksum, "topic": ledger["topic"], "paper_count": len(ledger["papers"]),
        "imported_at": datetime.now(timezone.utc).isoformat(), "private": True})
    return {"id": ident, "path": str(destination), "reused": False}


def load_archive(archive_dir, ident):
    if not ARCHIVE_ID.fullmatch(ident):
        raise ValueError("Invalid archive ID")
    root = Path(archive_dir).resolve()
    path = (root / ident).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Archive path escapes its root")
    ledger = validate_ledger(read_private_json(path, "evidence-ledger.json"))
    manifest = read_private_json(path, "manifest.json")
    checksum = content_hash(ledger)
    if not isinstance(manifest, dict) or checksum != manifest.get("content_sha256") or manifest.get("id") != ident or ident != "r_" + checksum[:16]:
        raise ValueError("Archive content changed outside the archive manager")
    return ledger


def public_registry(directory):
    directory = Path(directory).resolve()
    path = (directory / "index.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Public registry escapes its directory")
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Public registry is too large")
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("entries"), dict):
        raise ValueError("Invalid public registry")
    for key, value in data["entries"].items():
        if (not ARCHIVE_ID.fullmatch(key) or not isinstance(value, dict) or
                not isinstance(value.get("paper_ids"), list) or not all(valid_paper_id(v) for v in value["paper_ids"]) or
                not isinstance(value.get("topic"), str) or len(value["topic"]) > 500 or
                not re.fullmatch(r"[a-f0-9]{64}", str(value.get("summary_sha256", "")))):
            raise ValueError("Invalid public entry")
    return data


def publish_summary(archive_dir, public_dir, ident):
    with publication_lock(public_dir):
        return _publish_summary(archive_dir, public_dir, ident)


def _summary_path(public_dir, ident, registry):
    root = Path(public_dir).resolve()
    path = (root / (ident + ".html")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Summary escapes public directory")
    if path.exists():
        current = path.read_text("utf-8")
        expected = registry["entries"].get(ident, {}).get("summary_sha256")
        if current != TOMBSTONE and hashlib.sha256(current.encode()).hexdigest() != expected:
            raise ValueError("Public summary was edited outside the archive manager")
    return path


def _publish_summary(archive_dir, public_dir, ident):
    ledger = load_archive(archive_dir, ident)
    public_dir = Path(public_dir); public_dir.mkdir(parents=True, exist_ok=True)
    registry = public_registry(public_dir)
    path = _summary_path(public_dir, ident, registry)
    esc = lambda value: html.escape(str(value), quote=True)
    paper_ids, items = set(), []
    for paper in ledger["papers"]:
        paper_ids.add(paper["work_id"])
        aliases = paper.get("external_ids") or {}
        if aliases.get("doi"):
            paper_ids.add("doi:" + aliases["doi"])
        if aliases.get("aminer"):
            paper_ids.add("aminer:" + aliases["aminer"])
        url = safe_url(paper.get("url"))
        title = f'<a href="{esc(url)}">{esc(paper["title"])}</a>' if url else esc(paper["title"])
        items.append(f'<li>{title}</li>')
    # Deliberate allow-list: no quotations, filenames, private notes, or local paths.
    page = '<!doctype html><meta charset="utf-8"><title>专题公开概览</title><style>body{max-width:900px;margin:40px auto;padding:20px;font:16px/1.8 sans-serif}li{margin:18px 0}</style>'
    page += f'<h1>{esc(ledger["topic"])}</h1><p>本页是经显式导出的论文目录。完整证据包保存在本地，公开页不含全文摘录、文档路径或私人笔记。</p><ul>{"".join(items)}</ul>'
    path.write_text(page, "utf-8")
    registry["entries"][ident] = {"topic": ledger["topic"], "paper_ids": sorted(paper_ids),
        "published_summary_at": datetime.now(timezone.utc).isoformat(), "summary_sha256": hashlib.sha256(page.encode()).hexdigest()}
    atomic_json(public_dir / "index.json", registry)
    return {"id": ident, "summary": str(public_dir / (ident + ".html")), "contains_fulltext": False}


def withdraw_summary(public_dir, ident):
    with publication_lock(public_dir):
        return _withdraw_summary(public_dir, ident)


def _withdraw_summary(public_dir, ident):
    if not ARCHIVE_ID.fullmatch(ident):
        raise ValueError("Invalid archive ID")
    public_dir = Path(public_dir)
    registry = public_registry(public_dir)
    if ident not in registry["entries"]:
        raise ValueError("Summary is not registered")
    path = _summary_path(public_dir, ident, registry)
    registry["entries"].pop(ident)
    # Keep old URLs readable without retaining the published bibliography.
    path.write_text(TOMBSTONE, "utf-8")
    atomic_json(public_dir / "index.json", registry)


def attach_topics(works, public_dir):
    try:
        registry = public_registry(public_dir)
        entries = {}
        for key, row in registry["entries"].items():
            path = _summary_path(public_dir, key, registry)
            if path.is_file() and hashlib.sha256(path.read_text("utf-8").encode()).hexdigest() == row["summary_sha256"]:
                entries[key] = row
    except (OSError, ValueError, TypeError, AttributeError):
        logger.warning("Public research registry invalid; topic links omitted")
        entries = {}
    result = []
    for work in works:
        aliases = {ident if ident.startswith("aminer:") else "doi:" + ident for ident in work_feedback_ids(work)}
        links = [{"title": row.get("topic", "专题"), "url": "research/" + key + ".html"}
                 for key, row in entries.items() if aliases.intersection(row["paper_ids"])]
        result.append(work.model_copy(update={"extra": {**work.extra, "research_topics": links[:3]}}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["import", "publish-summary", "withdraw-summary", "list"])
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--dossier", type=Path)
    parser.add_argument("--id")
    args = parser.parse_args(argv)
    archive = args.archive_dir or args.base_dir / "data/research-archive"
    public = args.public_dir or args.base_dir / "reports/research"
    try:
        if args.command == "import":
            if not args.dossier: parser.error("import requires --dossier")
            result = import_dossier(args.dossier, archive)
        elif args.command == "publish-summary":
            if not args.id: parser.error("publish-summary requires --id")
            result = publish_summary(archive, public, args.id)
        elif args.command == "withdraw-summary":
            if not args.id: parser.error("withdraw-summary requires --id")
            withdraw_summary(public, args.id); result = {"id": args.id, "status": "withdrawn_for_next_deployment"}
        else:
            result = [{"id": p.name, "topic": load_archive(archive, p.name)["topic"]} for p in sorted(archive.glob("r_*")) if p.is_dir() and ARCHIVE_ID.fullmatch(p.name)]
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        parser.error("Invalid dossier/registry or inaccessible path; existing content was not intentionally replaced")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
