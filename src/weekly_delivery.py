"""Reserve the weekly SMTP submission and recover a generated, unsent digest."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import smtplib
import subprocess
from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

from .models import CandidateWork
from .notify_email import notify
from .watch_history import WatchHistory


def reserve_delivery(delivery_key: str) -> None:
    if not re.fullmatch(r"(?:\d{4}-W\d{2}|run-\d+)", delivery_key):
        raise ValueError("Invalid delivery key")
    # POST create-ref rejects an existing ref even when both runs use the same SHA.
    subprocess.run([
        "gh", "api", "--method", "POST", f"repos/{os.environ['GITHUB_REPOSITORY']}/git/refs",
        "-f", f"ref=refs/tags/zotwatch-mail-{delivery_key}", "-f", f"sha={os.environ['GITHUB_SHA']}",
        "--silent",
    ], check=True, timeout=30)


def deliver_digest(base: Path, delivery_key: str) -> None:
    reserved = False

    def before_submission():
        nonlocal reserved
        reserve_delivery(delivery_key)
        reserved = True

    try:
        notify(base, before_submission=before_submission)
    except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused, smtplib.SMTPDataError):
        # An explicit SMTP rejection leaves nobody accepted. Keep the reservation
        # when the connection drops or only some recipients were refused.
        if reserved:
            subprocess.run([
                "gh", "api", "--method", "DELETE",
                f"repos/{os.environ['GITHUB_REPOSITORY']}/git/refs/tags/zotwatch-mail-{delivery_key}",
                "--silent",
            ], check=True, timeout=30)
        raise


def restore_digest(base: Path, artifact: Path, week: str) -> int:
    reports = artifact / "reports"
    feed = ElementTree.parse(reports / "feed.xml")
    generated_at = parsedate_to_datetime(feed.findtext("./channel/lastBuildDate", ""))
    beijing = generated_at.astimezone(timezone(timedelta(hours=8)))
    if beijing.strftime("%G-W%V") != week:
        raise ValueError("Source digest belongs to a different Beijing delivery week")
    stamp = beijing.strftime("%Y%m%d")
    names = (f"report-{stamp}.html", f"digest-{stamp}.html", f"digest-{stamp}.txt", "feed.xml")
    if any(not (reports / name).is_file() for name in names):
        raise ValueError("The source artifact is missing the generated report, digest or feed")
    works = []
    for item in feed.findall("./channel/item"):
        identifier = item.findtext("guid", "")
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        # RSS includes version alerts, which are not normal works in sent history.
        if "版本与更正提醒" in [node.text for node in item.findall("category")]:
            continue
        if not identifier or not title:
            raise ValueError("Invalid source feed item")
        doi = next((match.group(1) for candidate in (identifier, link)
                    if (match := re.fullmatch(r"(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)",
                                             candidate, re.IGNORECASE))), None)
        works.append(CandidateWork(source="recovered_digest", identifier=identifier, title=title, doi=doi))
    current = base / "reports"
    current.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(reports / name, current / name)
    history = WatchHistory(base / "data/watch-state")
    snapshot = artifact / "data/watch-state/pending.json"
    if snapshot.exists():
        pending = json.loads(snapshot.read_text("utf-8"))
        history.stage(works)
        merged = json.loads(history.pending.read_text("utf-8"))
        merged["sent"].update(pending["sent"])
        history.pending.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        history.stage(works)
    return len(works)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("restore", "send"))
    parser.add_argument("--week", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    if args.command == "restore":
        count = restore_digest(Path.cwd(), args.artifact, args.week)
        print(f"Restored digest with {count} papers from original GitHub Actions artifact")
    else:
        key = f"run-{os.environ['GITHUB_RUN_ID']}" if args.manual else args.week
        deliver_digest(Path.cwd(), key)


if __name__ == "__main__":
    main()
