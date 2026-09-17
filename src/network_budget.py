"""Per-provider budgets, durable GET cache and cooldowns; never persist credentials."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from .http_utils import DeferredRequest, _retry_delay, RETRYABLE_STATUS_CODES


class BudgetSession(requests.Session):
    def __init__(self, directory, config):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "checkpoint.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.config = config
        self.calls = {}
        self.cache_hits = 0
        self.last_semantic = 0.0

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def defer_host(self, url, delay):
        host = urlparse(url).hostname
        self.state.setdefault("cooldowns", {})[host] = time.time() + max(0, delay)
        self.save()

    def rotate(self, name, values, limit):
        if not values:
            return []
        cursor = self.state.setdefault("rotation", {}).get(name, 0) % len(values)
        selected = (values[cursor:] + values[:cursor])[:limit]
        self.state["rotation"][name] = (cursor + len(selected)) % len(values)
        self.save()
        return selected

    def remaining(self):
        day = datetime.now(timezone.utc).date().isoformat()
        ledger = self.state.get("ledger", {})
        spent = ledger.get("openalex", 0) if ledger.get("date") == day else 0
        ceiling = self.config.openalex_key_budget if os.getenv("OPENALEX_API_KEY") else self.config.openalex_anonymous_budget
        return max(0, ceiling - spent)

    def request(self, method, url, **kwargs):
        host = urlparse(url).hostname
        if host not in {"api.openalex.org", "api.crossref.org"} or method.upper() != "GET":
            return super().request(method, url, **kwargs)
        params = dict(kwargs.get("params") or {})
        safe_params = {k: v for k, v in params.items() if k not in {"api_key", "mailto"}}
        key = hashlib.sha256(json.dumps([url, safe_params], sort_keys=True).encode()).hexdigest()
        cache = self.directory / (key + ".json")
        # Check cache before cooldown so already-retrieved pages remain usable.
        if cache.exists() and time.time() - cache.stat().st_mtime <= self.config.cache_hours * 3600:
            response = requests.Response()
            response.status_code = 200
            response._content = cache.read_bytes()
            response.encoding = "utf-8"
            response.url = url
            self.cache_hits += 1
            return response
        if self.state.get("cooldowns", {}).get(host, 0) > time.time():
            raise DeferredRequest("Provider cooldown is active")
        cap = self.config.openalex_max_requests if host == "api.openalex.org" else self.config.crossref_max_requests
        if self.calls.get(host, 0) >= cap:
            raise DeferredRequest("Per-run provider request budget exhausted")
        if host == "api.openalex.org":
            cost = 0.001 if any(k in params for k in ("search", "search.semantic", "search.exact")) else 0.0001 if urlparse(url).path.rstrip("/") == "/works" else 0
            if cost > self.remaining() + 1e-12:
                raise DeferredRequest("OpenAlex daily local budget exhausted")
            day = datetime.now(timezone.utc).date().isoformat()
            if self.state.get("ledger", {}).get("date") != day:
                self.state["ledger"] = {"date": day, "openalex": 0.0}
            self.state["ledger"]["openalex"] += cost
            api_key = os.getenv("OPENALEX_API_KEY")
            if api_key:
                params["api_key"] = api_key
            if "search.semantic" in params:
                time.sleep(max(0, 1.05 - (time.monotonic() - self.last_semantic)))
                self.last_semantic = time.monotonic()
        self.calls[host] = self.calls.get(host, 0) + 1
        self.save()  # Debit attempts even if transport fails; conservative budget accounting.
        kwargs["params"] = params
        try:
            response = super().request(method, url, **kwargs)
        except requests.RequestException as exc:
            # Requests' default exception embeds URLs, which may include an API key.
            error = requests.Timeout if isinstance(exc, requests.Timeout) else requests.ConnectionError
            raise error(f"{host} transport failure ({type(exc).__name__})") from None
        response.url = url  # Avoid credential-bearing query strings in HTTP error logs.
        if response.status_code in RETRYABLE_STATUS_CODES:
            delay = _retry_delay(response.headers.get("Retry-After"), 1.5)
            if delay > 20:
                self.defer_host(url, delay)
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                return response
            temp = cache.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp.replace(cache)
        return response

    def summary(self):
        return {"requests": self.calls, "cache_hits": self.cache_hits,
                "local_openalex_remaining_usd": round(self.remaining(), 5),
                "cooldowns": self.state.get("cooldowns", {})}
