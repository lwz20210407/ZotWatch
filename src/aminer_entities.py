"""Explicit, free entity lookups. Results are review candidates, never automatic ID merges."""
import argparse
import json
from pathlib import Path

from .aminer_client import AMinerClient, AMinerError
from .settings import AMinerConfig


def lookup(client, kind, name, *, organization=""):
    if kind not in {"person", "organization", "venue"}:
        raise ValueError("Unsupported entity kind")
    name = name.strip()
    if not name or len(name) > 300 or len(organization) > 300:
        raise ValueError("Entity names must be 1..300 characters")
    params = {"person": {"name": name, "size": 5}, "organization": {"orgs": [name]},
              "venue": {"name": name}}[kind]
    if kind == "person" and organization:
        params["org"] = organization
    rows = client.query(kind + "_search", params)
    fields = {"id", "org_id", "name", "name_zh", "name_en", "org_name", "org", "org_zh",
              "aliases", "interests", "venue_type", "n_citation"}
    candidates = [{k: v for k, v in row.items() if k in fields} for row in rows]
    return {"kind": kind, "query": name, "candidates": candidates, "selected_id": None,
            "status": "needs_review" if candidates else "no_match",
            "note": "Name matches do not prove identity. Verify institution/works or ISSN before mapping IDs; no tracking configuration was modified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--person")
    group.add_argument("--organization")
    group.add_argument("--venue")
    parser.add_argument("--org", default="", help="Institution filter for person lookup")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.org and not args.person:
        parser.error("--org only applies to --person")
    kind = next(k for k in ("person", "organization", "venue") if getattr(args, k))
    client = AMinerClient(AMinerConfig(max_requests=2, max_attempts=1), args.output_dir / "cache", query_version="entities-v1")
    try:
        result = lookup(client, kind, getattr(args, kind), organization=args.org)
    except (AMinerError, ValueError) as exc:
        print(json.dumps({"status": "failed", "category": str(exc) if isinstance(exc, AMinerError) else "invalid_input"}))
        raise SystemExit(1)
    result["api"] = client.summary()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "entity-review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"status": result["status"], "candidates": len(result["candidates"]), "api": result["api"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
