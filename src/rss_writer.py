from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from .models import RankedWork
from .citation_watch import recommendation_reasons

logger = logging.getLogger(__name__)


def site_url() -> str:
    """Public Pages URL, derived from the repository instead of hardcoded."""
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if "/" in repository:
        owner, name = repository.split("/", 1)
        return f"https://{owner.lower()}.github.io/{name}/"
    return os.getenv("ZOTWATCH_SITE_URL", "") or "https://github.com/"


def write_rss(
    works: Iterable[RankedWork],
    output_path: Path | str,
    *,
    title: str = "ZotWatch Feed",
    link: str | None = None,
    description: str = "Zotero 兴趣画像驱动的文献追踪",
) -> Path:
    works_list = list(works)
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = title
    # Previously this defaulted to "https://example.com" and was never overridden
    # by the caller, so every reader showed example.com as the feed home page.
    ET.SubElement(channel, "link").text = link or site_url()
    ET.SubElement(channel, "description").text = description
    discovered_at = datetime.now(timezone.utc)
    ET.SubElement(channel, "lastBuildDate").text = _format_rfc822(discovered_at)

    for work in works_list:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = work.title
        if work.url:
            ET.SubElement(item, "link").text = work.url
        # The identifier is an OpenAlex ID or a DOI, not a URL. RSS 2.0 treats a
        # guid as a permalink unless told otherwise.
        guid = ET.SubElement(item, "guid")
        guid.set("isPermaLink", "false")
        guid.text = work.identifier
        # pubDate is the date this feed learned about the paper, not the paper's
        # own publication date. Readers sort and mark-as-read by pubDate, so using
        # the publication date buried every "经典文献补漏" item -- a 2014 paper
        # entered the feed dated 2014 and was sorted below everything else or
        # treated as already seen.
        ET.SubElement(item, "pubDate").text = _format_rfc822(discovered_at)
        if work.published:
            ET.SubElement(item, "{http://purl.org/dc/elements/1.1/}date").text = work.published.date().isoformat()
        if work.venue:
            ET.SubElement(item, "category").text = work.venue
        description_lines = []
        if work.extra.get("report_channel"):
            ET.SubElement(item, "category").text = work.extra["report_channel"]
            description_lines.append(work.extra["report_channel"])
        description_lines.extend(recommendation_reasons(work))
        if work.extra.get("original_doi"):
            description_lines.extend([f"原论文: {work.extra['original_doi']}",
                                      work.extra.get("update_source", ""), work.extra.get("date_note", "")])
        for card in work.extra.get("transfer_cards", []):
            description_lines.extend([card["topic"] + ": " + card["use"], "迁移前检查: " + card["verify"],
                                      card["level"] + "片段: " + card["evidence"], card["status"]])
        for signal in work.extra.get("abstract_method_signals", []):
            description_lines.append("摘要方法线索: " + ", ".join(signal["methods"]) + " " + signal["role"] + "；" + signal["caveat"])
        for feedback in work.extra.get("feedback_links", []):
            description_lines.append("公开阅读反馈 " + feedback["name"] + ": " + feedback["url"])
        if work.abstract:
            description_lines.append(work.abstract)
        published_text = work.published.isoformat() if work.published else "Unknown"
        description_lines.append(f"Published: {published_text}")
        description_lines.append(f"Venue: {work.venue or 'Unknown'}")
        if work.extra.get("research_priority"):
            description_lines.append(f"研究类型: {work.extra['research_priority']}")
        if work.extra.get("watched_authors"):
            names = ", ".join(author["name"] for author in work.extra["watched_authors"])
            description_lines.append(f"重点作者新作: {names}")
            ET.SubElement(item, "category").text = "重点作者新作"
        ET.SubElement(item, "description").text = "\n".join(description_lines)

    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
    tree = ET.ElementTree(rss)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    logger.info("Wrote RSS feed with %d items to %s", len(works_list), path)
    return path


def _format_rfc822(dt: datetime | None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")


__all__ = ["write_rss", "site_url"]
