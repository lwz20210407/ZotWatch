"""Offline evaluation against explicit human labels; missing labels are not negatives."""
import argparse
import json
from pathlib import Path

from .research_features import normalize_doi


def evaluate(ranking, labels, k=20):
    judgments = {normalize_doi(r["doi"]): r["relevant"] for r in labels}
    if any(type(value) is not bool for value in judgments.values()):
        raise ValueError("relevant labels must be explicit booleans")
    seen, top = set(), []
    for row in ranking:
        doi = normalize_doi(row.get("doi"))
        if doi and doi not in seen:
            top.append(row); seen.add(doi)
        if len(top) == k:
            break
    assessed = [judgments[normalize_doi(r["doi"])] for r in top if normalize_doi(r["doi"]) in judgments]
    positives = sum(assessed)
    facets = sorted({f for row in top if judgments.get(normalize_doi(row["doi"])) is True for f in row.get("facets", [])})
    return {"k": k, "returned": len(top), "judged": len(assessed), "relevant": positives,
            "precision_at_k": positives / len(top) if top and len(assessed) == len(top) else None,
            "judged_precision": positives / len(assessed) if assessed else None,
            "label_coverage": len(assessed) / len(top) if top else 0,
            "covered_research_facets": facets,
            "warning": "仅评估已标注集合；不代表全网召回率。评估标签应与训练反馈隔离。"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    result = {name: evaluate(json.loads(Path(path).read_text(encoding="utf-8"))["ranking"], labels, args.top)
              for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
