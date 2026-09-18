"""Email delivery for the weekly digest.

This used to be an 85-line heredoc inside the GitHub Actions workflow, which meant
it could not be imported, tested or dry-run. Two behavioural changes beyond the
move:

* The digest is the HTML *body* (multipart/alternative), not an attachment.
  University Exchange servers and mobile clients routinely strip, quarantine or
  refuse to preview HTML attachments, so the previous email delivered six lines of
  links and hid the actual content in a file.
* The recipient is configuration, not a literal. It was hardcoded to a personal
  address in a public repository.
"""
from __future__ import annotations

import html
import logging
import os
import re
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import List, Optional, Sequence

from .utils import beijing_now

logger = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 30
SEND_ATTEMPTS = 3
PREVIEW_ITEMS = 8


class EmailNotConfigured(RuntimeError):
    """Raised when required SMTP settings are missing."""


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipients: Sequence[str]

    @classmethod
    def from_env(cls) -> "SmtpConfig":
        required = {
            "SMTP_HOST": os.getenv("SMTP_HOST"),
            "SMTP_USERNAME": os.getenv("SMTP_USERNAME"),
            "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD"),
            "EMAIL_TO": os.getenv("EMAIL_TO"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise EmailNotConfigured(
                f"Email not delivered; missing secrets/env: {', '.join(missing)}"
            )
        username = required["SMTP_USERNAME"] or ""
        recipients = [part.strip() for part in re.split(r"[,;]", required["EMAIL_TO"] or "") if part.strip()]
        return cls(
            host=required["SMTP_HOST"] or "",
            port=int(os.getenv("SMTP_PORT") or "587"),
            username=username,
            password=required["SMTP_PASSWORD"] or "",
            sender=os.getenv("SMTP_FROM") or username,
            recipients=recipients,
        )


def site_urls() -> tuple[str, str, str]:
    """Pages URL, feed URL and run URL, derived from the repository."""
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if "/" in repository:
        owner, name = repository.split("/", 1)
        pages = f"https://{owner.lower()}.github.io/{name}/"
    else:
        pages = os.getenv("ZOTWATCH_SITE_URL", "") or ""
    feed = f"{pages}feed.xml" if pages else ""
    run_id = os.getenv("GITHUB_RUN_ID")
    run = f"https://github.com/{repository}/actions/runs/{run_id}" if repository and run_id else ""
    return pages, feed, run


def latest_report(reports_dir: Path, prefix: str = "report") -> Optional[Path]:
    """Newest file by filename date.

    Sorting by mtime was unreliable: the workflow copies reports with `cp -r`,
    which rewrites every mtime to the copy time. Names carry a YYYYMMDD stamp, so
    lexicographic order on the name is both correct and deterministic.
    """
    reports = sorted(reports_dir.glob(f"{prefix}-*.html"), key=lambda path: path.name)
    return reports[-1] if reports else None


def latest_digest(reports_dir: Path) -> Optional[Path]:
    """The lean email body, falling back to the full report if absent."""
    return latest_report(reports_dir, "digest") or latest_report(reports_dir, "report")


def _extract_entries(report_html: str, limit: int = PREVIEW_ITEMS) -> List[tuple[str, str]]:
    """Pull (title, url) pairs out of the rendered report for the summary block."""
    pattern = re.compile(
        r"<h2>\s*\d+\.\s*<a href=\"(?P<url>[^\"]*)\">(?P<title>.*?)</a>\s*</h2>",
        re.DOTALL,
    )
    entries: List[tuple[str, str]] = []
    for match in pattern.finditer(report_html):
        title = re.sub(r"<[^>]+>", "", match.group("title")).strip()
        entries.append((html.unescape(title), html.unescape(match.group("url"))))
        if len(entries) >= limit:
            break
    return entries


def build_message(
    config: SmtpConfig,
    *,
    report_path: Optional[Path],
    feed_path: Optional[Path],
    now: Optional[datetime] = None,
) -> EmailMessage:
    now = now or beijing_now()
    pages_url, feed_url, run_url = site_urls()
    report_html = report_path.read_text(encoding="utf-8") if report_path and report_path.exists() else ""
    entries = _extract_entries(report_html)

    msg = EmailMessage()
    msg["Subject"] = f"文献周报 {now:%Y-%m-%d}"
    msg["From"] = config.sender
    msg["To"] = ", ".join(config.recipients)
    msg["Date"] = formatdate(now.timestamp(), localtime=True)
    # A stable per-week Message-ID plus References threads the digests into one
    # conversation instead of scattering them across the mailbox.
    domain = config.sender.split("@")[-1] or "zotwatch.local"
    msg["Message-ID"] = make_msgid(idstring=f"zotwatch-{now:%Y%m%d}", domain=domain)
    thread_root = f"<zotwatch-digest@{domain}>"
    msg["References"] = thread_root
    msg["In-Reply-To"] = thread_root
    if pages_url:
        msg["List-Archive"] = f"<{pages_url}>"

    # `watch` renders the plain-text alternative alongside the HTML digest; fall
    # back to scraping titles out of the body only when that file is missing.
    text_path = report_path.with_suffix(".txt") if report_path else None
    if text_path and text_path.exists():
        text_body = text_path.read_text(encoding="utf-8")
    else:
        lines = [f"文献周报 {now:%Y-%m-%d}", ""]
        lines += ([f"{i}. {t}\n   {u}" for i, (t, u) in enumerate(entries, 1)]
                  or ["本轮无通过筛选且尚未推送的文献。"])
        lines += ["", f"完整周报：{pages_url or '未配置'}", f"RSS：{feed_url or '未配置'}"]
        text_body = "\n".join(lines)
    if run_url:
        text_body += f"\n运行记录：{run_url}"
    msg.set_content(text_body)

    if report_html:
        # digest-*.html is already a standalone email document; only a legacy
        # full report needs the link footer appended.
        body = report_html if "ZOTWATCH" in report_html else _inline_body(
            report_html, pages_url, feed_url, run_url
        )
        msg.add_alternative(body, subtype="html")
    if feed_path and feed_path.exists():
        msg.add_attachment(
            feed_path.read_bytes(), maintype="application", subtype="rss+xml", filename=feed_path.name
        )
    return msg


def _inline_body(report_html: str, pages_url: str, feed_url: str, run_url: str) -> str:
    """Use the report itself as the email body, with a link footer appended."""
    footer_parts = []
    if pages_url:
        footer_parts.append(f'<a href="{html.escape(pages_url)}">网页版</a>')
    if feed_url:
        footer_parts.append(f'<a href="{html.escape(feed_url)}">RSS</a>')
    if run_url:
        footer_parts.append(f'<a href="{html.escape(run_url)}">运行记录</a>')
    footer = (
        '<hr /><p style="color:#666;font-size:.85rem">' + " · ".join(footer_parts) + "</p>"
        if footer_parts
        else ""
    )
    if "</body>" in report_html:
        return report_html.replace("</body>", f"{footer}</body>", 1)
    return report_html + footer


def send(msg: EmailMessage, config: SmtpConfig, *, attempts: int = SEND_ATTEMPTS) -> None:
    """Send with bounded retries; a single flaky SMTP session used to fail the run."""
    context = ssl.create_default_context()
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            if config.port == 465:
                with smtplib.SMTP_SSL(
                    config.host, config.port, context=context, timeout=SMTP_TIMEOUT_SECONDS
                ) as server:
                    server.login(config.username, config.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(config.host, config.port, timeout=SMTP_TIMEOUT_SECONDS) as server:
                    server.starttls(context=context)
                    server.login(config.username, config.password)
                    server.send_message(msg)
            logger.info("Sent digest to %s", ", ".join(config.recipients))
            return
        except (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused):
            # Credentials and recipients will not become valid by retrying.
            raise
        except (smtplib.SMTPException, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = 5.0 * attempt
            logger.warning(
                "SMTP attempt %d/%d failed with %s; retrying in %.0fs",
                attempt, attempts, exc.__class__.__name__, delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"SMTP delivery failed after {attempts} attempts") from last_error


def notify(base_dir: Path, *, dry_run: bool = False) -> Optional[Path]:
    """Build and send the digest. In dry-run mode write the message to disk instead."""
    reports_dir = Path(base_dir) / "reports"
    report_path = latest_digest(reports_dir)
    feed_path = reports_dir / "feed.xml"
    if report_path is None:
        logger.warning("No report found in %s; sending link-only digest", reports_dir)

    if dry_run:
        config = SmtpConfig(
            host="dry-run", port=587, username="dry-run@example.invalid",
            password="", sender="dry-run@example.invalid",
            recipients=[os.getenv("EMAIL_TO") or "dry-run@example.invalid"],
        )
        msg = build_message(config, report_path=report_path, feed_path=feed_path)
        preview = reports_dir / "email-preview.eml"
        preview.write_bytes(bytes(msg))
        logger.info("Dry run: wrote %s (%d bytes)", preview, preview.stat().st_size)
        return preview

    config = SmtpConfig.from_env()
    msg = build_message(config, report_path=report_path, feed_path=feed_path)
    send(msg, config)
    return None


__all__ = ["notify", "build_message", "send", "SmtpConfig", "EmailNotConfigured", "latest_report"]
