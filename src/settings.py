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


class ZoteroConfig(BaseModel):
    mode: str = "api"
    api: ZoteroApiConfig = Field(default_factory=ZoteroApiConfig)

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


class Thresholds(BaseModel):
    must_read: float = 0.75
    consider: float = 0.5


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
    author_watch: AuthorWatchConfig = Field(default_factory=AuthorWatchConfig)
    citation_watch: "CitationWatchConfig" = Field(default_factory=lambda: CitationWatchConfig())


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
    return Settings(
        zotero=ZoteroConfig(**zotero_cfg),
        sources=SourcesConfig(**sources_cfg),
        scoring=ScoringConfig(**scoring_cfg),
        author_watch=AuthorWatchConfig(**author_cfg),
        citation_watch=CitationWatchConfig(**citation_cfg),
    )


__all__ = [
    "Settings",
    "load_settings",
    "ZoteroConfig",
    "SourcesConfig",
    "ScoringConfig",
]
