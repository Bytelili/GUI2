# Pipeline Status

## Implemented

- Strict official-test task preparation from formal execution references.
- Protocol and target-field provenance gates.
- Clean adapter provenance and hash gates.
- Model-by-condition matrix construction.
- Comparison-component task-ID alignment.
- LLaMA-Factory and replay inference backends.
- ADB and replay device backends.
- Deterministic per-app reset/cleanup hooks.
- Official action parsing and ADB action execution.
- Per-step screenshots, XML, actions, timing, tokens, and errors.
- Per-task resumability and retryable transient failures.
- Pre-model-load environment, task hash, adapter hash, ADB, hook, and rule
  preflight.
- Explicit final-state success rules and manual-review fallback.
- Official-format scored execution CSV files.
- Exact official fuzzy-text Sim2 formula plus strict action-list diagnostics.
- Official-reference Prompt mode and optional `uiautomator2` device mode.
- FingerTip source-contract and dataset-hash audit.
- Runtime App-hook/success-rule template generation.
- Device fingerprint, display-property, and installed-task-app preflight.
- Paired task bootstrap, user-cluster bootstrap, and per-user comparisons.
- Bundled official evaluator integration.
- Coverage, duplicate-ID, hash, retry, verification, and paper-eligibility
  audit.

## Intentionally not automatic

- Generic app-state reset without an authored app hook.
- Success inference from `finished()`.
- Success inference when no explicit final-state rule or manual annotation
  exists.
- Parallel ADB execution on one device.

These are excluded because automating them without environment-specific
evidence would make the experiment less reliable.
