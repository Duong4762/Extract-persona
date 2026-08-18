# Amazon persona extractor

`amazon_extractor.py` converts local Amazon Reviews 2023 JSONL files into
schema-constrained persona records. The pipeline has four resumable stages:

1. **ingest** — scan review files in the requested year range and write reviews
   to JSONL shards grouped by `user_id`.
2. **prepare** — rank eligible reviewers, optionally enrich their reviews with
   product metadata, filter weak/duplicate reviews, and make an 80/20 temporal
   construction/validation split.
3. **compact** — turn each construction split into a bounded text profile for
   the model. Validation reviews are deliberately excluded.
4. **extract** — call the OpenAI-compatible endpoint in schema chunks, validate
   returned values/evidence, and append one 1,290-field persona per reviewer.

## Usage

```powershell
python amazon_extractor.py ingest
python amazon_extractor.py prepare
python amazon_extractor.py compact
python amazon_extractor.py extract --max-llm-users 10
```

Run every stage with `python amazon_extractor.py all`. By default, artifacts are
written to `amazon_persona_fresh/` and the schema is read from
`schema/dimensions.json`. Use `--work-dir` and `--schema-path` to override them.

All relative paths are resolved from the project directory containing
`amazon_extractor.py`, not from the shell's current directory. Default inputs:

- `data/amazon/reviews/`
- `data/amazon/metadata/`

Intermediate data is file-only; SQLite is not used:

- `review_shards/reviews-*.jsonl`: normalized reviews grouped by user shard.
- `selected_users.jsonl`: ranked eligible reviewers.
- `user_histories.prepared.jsonl`: filtered construction/validation histories.
- `compact_profiles.jsonl`: bounded profiles sent to the model.
- `personas_1290.jsonl`: final persona records.

The extraction endpoint is configured through `AMAZON_LLM_ENDPOINT`,
`AMAZON_LLM_MODEL`, and optionally `AMAZON_LLM_AUTHORIZATION`. Existing user IDs
in `personas_1290.jsonl` are skipped, so extraction can resume safely.
