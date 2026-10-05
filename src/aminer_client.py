"""Bounded, free-endpoint-only AMiner reads. Never log credentials or response errors."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
BASE_URL = "https://datacenter.aminer.cn/gateway/open_platform"
ENDPOINTS = {
    "recommend": ("POST", "/api/v3/paper/rec5"),
    "search": ("GET", "/api/paper/search"),
    "info": ("POST", "/api/paper/info"),
    "person_search": ("POST", "/api/person/search"),
    "organization_search": ("POST", "/api/organization/search"),
    "venue_search": ("POST", "/api/venue/search"),
}


class AMinerError(requests.RequestException):
    """Only a controlled error category, never a server body or request headers."""


class AMinerClient:
    def __init__(self, config, cache_dir: Path, *, session=None, token=None, query_version="1"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.session = session if session is not None else requests.Session()
        self._token = token if token is not None else os.getenv(config.api_key_env, "")
        self._token = self._token.removeprefix("Bearer ").strip()
        self.query_version = query_version
        self.started = time.monotonic()
        self.calls = 0
        self.cache_hits = 0
        self.by_endpoint = {}
        self.warnings = []
        # Why a request failed, by exception CLASS only. Every run on 2026-10-01 logged
        # a bare "network_failure", because the cause was discarded with `from None`, so
        # there was no way to tell a connect timeout from a slow read, a TLS failure or
        # DNS -- and no basis for tuning anything. Class names carry no secrets; the
        # exception message is deliberately not recorded (it can include the URL).
        self.network_errors = []
        self.stopped = False
        self.last_request = None

    def warn(self, category):
        if category not in self.warnings:
            self.warnings.append(category)
            logger.warning("AMiner source incomplete: %s", category)

    def summary(self):
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "by_endpoint": dict(self.by_endpoint), "warnings": list(self.warnings),
                "network_errors": list(self.network_errors),
                "free_endpoints_only": True}

    def _key(self, endpoint, params):
        value = json.dumps([2, self.query_version, endpoint, params], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def _read(self, path):
        try:
            d = json.loads(path.read_text("utf-8"))
            age = time.time() - d["at"]
            if 0 <= age <= self.config.cache_hours * 3600 and isinstance(d["rows"], list):
                self.cache_hits += 1
                return d["rows"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _write(self, path, rows):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            # Exclude the credential even if a broken service echoes it in a field.
            text = json.dumps({"at": time.time(), "rows": rows}, ensure_ascii=False)
            if self._token:
                text = text.replace(self._token, "[redacted]")
            tmp.write_text(text, "utf-8")
            tmp.replace(path)
        except OSError:
            self.warn("cache_write_failed")

    def query(self, endpoint, params, *, reserve=0):
        if endpoint not in ENDPOINTS:
            raise AMinerError("endpoint_not_allowed")
        if not self._token:
            self.stopped = True
            raise AMinerError("credential_missing")
        if self.stopped:
            raise AMinerError("source_stopped")
        path = self.cache_dir / (self._key(endpoint, params) + ".json")
        cached = self._read(path)
        if cached is not None:
            try:
                return self._rows(endpoint, cached)
            except AMinerError:
                self.cache_hits -= 1  # Corrupt/schema-incompatible cache is refreshed.
        method, route = ENDPOINTS[endpoint]
        for attempt in range(self.config.max_attempts):
            if self.calls >= max(0, self.config.max_requests - reserve):
                raise AMinerError("request_cap")
            remaining = self.config.max_run_seconds - (time.monotonic() - self.started)
            if remaining <= 0:
                self.stopped = True
                raise AMinerError("time_cap")
            if reserve and endpoint != "info":
                remaining -= min(30, self.config.max_run_seconds * 0.2)
                if remaining <= 0:
                    raise AMinerError("discovery_time_cap")
            delay = max(0, self.config.interval_seconds - (time.monotonic() - self.last_request)) if self.last_request else 0
            if delay >= remaining:
                raise AMinerError("time_cap")
            if delay:
                time.sleep(delay)
            self.calls += 1
            self.by_endpoint[endpoint] = self.by_endpoint.get(endpoint, 0) + 1
            self.last_request = time.monotonic()
            try:
                response = self.session.request(method, BASE_URL + route,
                    headers={"Authorization": self._token, "X-Platform": "zotwatch",
                             "X-Skill-Name": "zotwatch-aminer-source", "X-Skill-Version": "1"},
                    **({"params": params} if method == "GET" else {"json": params}),
                    timeout=min(self.config.timeout_seconds, remaining - delay), allow_redirects=False)
            except requests.RequestException as exc:
                elapsed = round(time.monotonic() - self.last_request, 1)
                self.network_errors.append({"endpoint": endpoint, "attempt": attempt + 1,
                                            "error": type(exc).__name__, "elapsed_s": elapsed})
                logger.info("AMiner %s attempt %d failed: %s after %.1fs",
                            endpoint, attempt + 1, type(exc).__name__, elapsed)
                if attempt + 1 < self.config.max_attempts:
                    continue
                raise AMinerError("network_failure") from None
            try:
                body = response.json()
            except ValueError:
                body = None
            code = str(body.get("code", "")) if isinstance(body, dict) else ""
            if response.status_code in (401, 403) or code in {"40301", "40302", "40307", "40308"}:
                self.stopped = True
                raise AMinerError("credential_or_permission_failure")
            if response.status_code == 429 or code == "40306":
                # Never occupy the weekly worker sleeping through a long cooldown.
                try:
                    wait = min(max(float(response.headers.get("Retry-After", 60)), 1), 3600)
                except (ValueError, TypeError):
                    wait = 60
                self._cooldown(wait)
                self.stopped = True
                raise AMinerError("rate_limited")
            if response.status_code >= 500 or code == "50001":
                if attempt + 1 < self.config.max_attempts:
                    continue
                raise AMinerError("service_failure")
            if response.status_code != 200:
                raise AMinerError("http_failure")
            if not isinstance(body, dict) or body.get("success") is not True or code not in ("", "200"):
                raise AMinerError("business_or_schema_failure")
            rows = body.get("data")
            if isinstance(rows, dict):
                rows = rows.get("items")
            rows = self._rows(endpoint, rows)
            self._write(path, rows)
            return rows
        raise AMinerError("service_failure")

    @staticmethod
    def _rows(endpoint, rows):
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise AMinerError("invalid_data_shape")
        # rec5 live contract: data=[{analyzed_topics, offset, size, papers:[...]}].
        if endpoint == "recommend" and rows and all("papers" in r for r in rows):
            if not all(isinstance(r["papers"], list) for r in rows):
                raise AMinerError("invalid_recommendation_shape")
            rows = [paper for group in rows for paper in group["papers"]]
            if not all(isinstance(r, dict) for r in rows):
                raise AMinerError("invalid_recommendation_shape")
        return rows

    def _cooldown(self, seconds):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "cooldown.json").write_text(json.dumps({"until": time.time() + seconds}), "utf-8")
        except OSError:
            pass

    def check_cooldown(self):
        active = False
        try:
            active = json.loads((self.cache_dir / "cooldown.json").read_text("utf-8"))["until"] > time.time()
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if active:
            self.stopped = True
            raise AMinerError("rate_limited_cooldown")
