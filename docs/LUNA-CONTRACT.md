# Owned Luna and episode-scoring patch

Base: seo-see/K-NHIB `2e27fa354101e572984b74ef74d4602392dfa32c`.
Source authored only. No test, benchmark import, installation or provider request has executed.

The exact `gpt-6-luna` route uses direct Responses with explicit medium/max effort and
standard/pro mode. Luna has no automatic retries or provider/model fallback. It excludes
web-search conditions in this bounded implementation: built-in tool charges/input growth
would need another budget contract. Text-input assembly includes guideline text; PDF input
is explicit. No expected/gold label is included in the model request.

Before generation, the Luna path calls the official `/v1/responses/input_tokens` endpoint;
missing or excessive counts stop generation. Per-case classification is a single model
response, with explicit input/output caps. A six-call default run cap rejects accidentally
launching the full corpus. The count call is also an external request, so neither count nor
generation may run without later API approval. Token preflight does not reserve provider
quota or guarantee billing. Transport ignores inherited proxy settings and redirects;
failed/uncertain responses retain unknown usage rather than zero. Runtime/API acceptance
and complete financial reconciliation remain unverified.

Use exact repeated `--case-id` arguments, `--models gpt-6-luna`, explicit
`--reasoning-effort medium|max`, `--reasoning-mode standard`, `--max-input-tokens`,
`--max-output-tokens`, `--max-run-calls`, `--experiment-id` and `--output-dir` in a
later approved batch. One experiment ID may identify paired arms; episode IDs separate
requested model, effort, mode, condition, replicate and input/config hash. Retries retain
all attempts and validate original episode identity before sending. Local dotenv reading
is opt-in via `--load-env`; import no longer loads dotenv or creates results directories.

The scorer reads the generator's root results directory, or `--results-dir`. It rejects
legacy files without explicit episode metadata rather than inventing experiment/effort
identity. Raw all-attempts history is saved before final-attempt selection. Every replicate,
error and exact model remains visible; dry-run placeholder rows are excluded from scores.
Per-replicate tables are exported. Scoring labels are configuration keys, not independent
statistical samples. Historical study-model mappings are reference-only.

Prepared tests: `tests/test_run_contract.py` uses only the Python standard library and
invented values. `tests/test_episode_scoring.py` needs pandas/numpy/sklearn. They are not
executed. An action-specific offline test batch must pin final source/test hashes, deny
network, scrub credentials, mount source read-only, cap CPU/RAM/process/time/aggregate
scratch and retain outputs. No locked dependency environment is implied by this patch.

Official contract references: [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[reasoning](https://developers.openai.com/api/docs/guides/reasoning),
[count input tokens](https://developers.openai.com/api/docs/guides/token-counting).
