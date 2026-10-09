"""Configuration and project-relative paths for the Facebook pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ingest reads every one of these source batches (data/facebook/batchN) and
# fans the users out across NUM_FRESH_BATCHES output batches so downstream
# stages can process each output batch independently.
SOURCE_BATCH_NUMBERS = (2, 3, 4, 5, 6, 7)
NUM_FRESH_BATCHES = 13
# Output batches are named batchN starting here, so they don't collide with
# the already-processed facebook_persona_fresh/batch1 (built from data batch1).
FRESH_BATCH_NAME_START = 2

# Which output batch (FRESH_BATCH_NAME_START..FRESH_BATCH_NAME_START+NUM_FRESH_BATCHES-1,
# i.e. 2..14) the prepare/compact/extract/stats stages operate on; the number is the
# literal folder suffix (FRESH_BATCH=2 -> facebook_persona_fresh/batch2). Ingest always
# writes all of them in a single pass.
FRESH_BATCH = int(os.environ.get("FRESH_BATCH", str(FRESH_BATCH_NAME_START)))


@dataclass(frozen=True)
class Config:
    fresh_batch: int = FRESH_BATCH
    content_dirs: tuple[Path, ...] = tuple(
        PROJECT_ROOT / f"data/facebook/batch{n}/content" for n in SOURCE_BATCH_NUMBERS
    )
    user_dirs: tuple[Path, ...] = tuple(
        PROJECT_ROOT / f"data/facebook/batch{n}/user" for n in SOURCE_BATCH_NUMBERS
    )
    fresh_root: Path = PROJECT_ROOT / "facebook_persona_fresh"
    work_dir: Path = PROJECT_ROOT / "facebook_persona_fresh" / f"batch{FRESH_BATCH}"
    schema_path: Path = PROJECT_ROOT / "schema/dimensions_final.json"
    # Prior schema revisions personas may have been extracted under; the ``migrate``
    # stage uses these to carry forward unchanged dimensions onto ``schema_path``.
    legacy_schema_paths: tuple[Path, ...] = (
        PROJECT_ROOT / "schema/dimensions.json",
        PROJECT_ROOT / "schema/dimension.json",
    )
    min_posts: int = 50
    min_text_chars: int = 1_000
    min_post_text_chars: int = 20
    max_contents: int = 100
    timeline_content_ratio: float = 0.4
    min_history_days: float = 356
    user_threshold_grid_posts: tuple[int, ...] = (
        1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200,
    )
    user_threshold_grid_chars: tuple[int, ...] = (
        50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000,
    )
    user_threshold_grid_days: tuple[float, ...] = (
        0, 30, 60, 90, 120, 180, 270, 356, 450,
    )
    user_threshold_scenarios: tuple[tuple[str, int, int, float], ...] = (
        ("Rat long", 5, 200, 90),
        ("De xuat", 10, 300, 180),
        ("Trung binh", 15, 500, 180),
        ("Nguong cu", 50, 1000, 356),
    )
    max_profile_chars: int = 20_000
    max_dims_per_chunk: int = 30
    max_llm_users: int = 5000
    llm_workers: int = 10
    preprocess_workers: int = int(
        os.environ.get("PREPROCESS_WORKERS", str(os.cpu_count()-8 or 4))
    )
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
    llm_timeout_seconds: int = 1800
    llm_bench_concurrency_levels: tuple[int, ...] = tuple(
        int(level) for level in os.environ.get(
            "LLM_BENCH_LEVELS", "1,2,4,8,12,16,24,32,64,128"
        ).split(",") if level.strip()
    )
    llm_bench_requests_per_worker: int = int(
        os.environ.get("LLM_BENCH_REQUESTS_PER_WORKER", "2")
    )
    llm_bench_prompt_index: int = int(os.environ.get("LLM_BENCH_PROMPT_INDEX", "14"))

    def fresh_batch_dir(self, batch_number: int) -> Path:
        """Work dir for output batch ``batch_number`` (the literal folder suffix, e.g. 2..14)."""
        return self.fresh_root / f"batch{batch_number}"

    @property
    def post_shards_dir(self) -> Path:
        return self.work_dir / "post_shards"

    @property
    def selected_users_path(self) -> Path:
        return self.work_dir / "selected_users.jsonl"

    @property
    def user_threshold_report_path(self) -> Path:
        # Dataset-wide (aggregates every ingested batch), not scoped to one FRESH_BATCH.
        return self.fresh_root / "user_threshold_report.json"

    @property
    def user_threshold_chart_path(self) -> Path:
        return self.fresh_root / "user_threshold_chart.png"

    @property
    def user_profiles_path(self) -> Path:
        return self.work_dir / "user_profiles.jsonl"

    @property
    def history_shards_dir(self) -> Path:
        return self.work_dir / "history_shards"

    @property
    def compact_shards_dir(self) -> Path:
        return self.work_dir / "compact_shards"

    @property
    def personas_path(self) -> Path:
        return self.work_dir / "personas_1290.jsonl"

    @property
    def migrated_personas_path(self) -> Path:
        return self.work_dir / "personas_migrated.jsonl"

    @property
    def persona_stats_path(self) -> Path:
        return self.work_dir / "persona_stats.json"

    @property
    def persona_coverage_chart_path(self) -> Path:
        return self.work_dir / "persona_category_coverage.png"

    @property
    def prompt_log_dir(self) -> Path:
        return self.work_dir / "prompt_log"

    @property
    def llm_bench_path(self) -> Path:
        return self.work_dir / "llm_bench.json"
