"""Configuration and project-relative paths for the Facebook pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    content_dir: Path = PROJECT_ROOT / "data/facebook/content"
    user_dir: Path = PROJECT_ROOT / "data/facebook/user"
    work_dir: Path = PROJECT_ROOT / "facebook_persona_fresh"
    schema_path: Path = PROJECT_ROOT / "schema/dimension.json"
    max_rows_per_file: int = 0
    top_k: int = 100_000
    min_posts: int = 50
    min_text_chars: int = 1_000
    min_post_text_chars: int = 20
    min_history_days: float = 356
    max_profile_chars: int = 35_000
    max_post_text_chars: int = 2_000
    max_dims_per_chunk: int = 50
    max_llm_users: int = 0
    post_shards: int = 50
    llm_provider: str = os.environ.get("LLM_PROVIDER", "local")
    model: str = os.environ.get("LLM_MODEL", "Qwen3-14B")
    llm_endpoint: str = os.environ.get(
        "LLM_ENDPOINT", "http://203.113.152.4:7777/llm/v1/chat/completions"
    )
    llm_authorization: str = os.environ.get("LLM_AUTHORIZATION", "")
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.environ.get(
        "OPENROUTER_MODEL", "google/gemma-4-31b-it:free"
    )
    llm_timeout_seconds: int = 300

    @property
    def post_shards_dir(self) -> Path:
        return self.work_dir / "post_shards"

    @property
    def selected_users_path(self) -> Path:
        return self.work_dir / "selected_users.jsonl"

    @property
    def user_profiles_path(self) -> Path:
        return self.work_dir / "user_profiles.jsonl"

    @property
    def history_shards_dir(self) -> Path:
        return self.work_dir / "history_shards"

    @property
    def compact_profiles_path(self) -> Path:
        return self.work_dir / "compact_profiles.jsonl"

    @property
    def personas_path(self) -> Path:
        return self.work_dir / "personas_1290.jsonl"

    @property
    def persona_stats_path(self) -> Path:
        return self.work_dir / "persona_stats.json"

    @property
    def persona_coverage_chart_path(self) -> Path:
        return self.work_dir / "persona_category_coverage.png"

    @property
    def prompt_log_dir(self) -> Path:
        return self.work_dir / "prompt_log"
