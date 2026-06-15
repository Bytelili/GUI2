# FingerTip-20K Official Compatibility

Reference implementation:
`FingerTip-20K-main/personalized_execution.py`.

## Strictly reproduced

- Official action names and textual output format.
- Official malformed-output fallback to `wait()` when
  `invalid_action_policy=official_wait`.
- Official scroll direction semantics and default 500 px / 300 ms gesture.
- Official 500 ms long click and 2 second `wait()` behavior.
- Official `fuzzywuzzy.fuzz.ratio` complete-action-text similarity, rounded to
  two decimals. A standard-library fallback reproduces fuzzywuzzy's default
  `SequenceMatcher` calculation when the package is unavailable.
- The bound official source snapshot's literal U+FF0C fullwidth-comma action
  separator, verified by AST and Unicode codepoint rather than terminal glyph.
- Official `down_sim == 0 -> 0.4` denominator fallback and `up_sim / down_sim`
  personalized similarity.
- Official output columns consumed by the bundled evaluator.
- Official raw model-output history for similarity and `real_step`.
- Optional official-reference Prompt content and optional `uiautomator2`
  hierarchy/text-input backends.
- Official `sampled_test_execution.csv` task preparation when the file is
  present in the project's official-data directory.

## Intentionally strengthened

- Historical references use the strict temporal training partition instead of
  official `total.csv`, preventing test leakage.
- ADB commands use argument arrays rather than `shell=True`.
- Every app requires a deterministic reset hook by default.
- Empty screenshots, empty UI hierarchies, and empty required reset hooks are
  treated as hard execution failures.
- `finished()` is not treated as verified success.
- Success requires an explicit final-state rule or evidence-backed annotation.
- Model/device failures, duplicate task IDs, changed hashes, and stale
  manifests are hard failures.
- Results from Base, SFT, Listwise, and DPO are evaluated separately.

Install `requirements-official-compat.txt` on the evaluation server to use the
same named packages as the official source. The fallback exists for local code
tests, not as a reason to omit the official dependencies from the server.
The official-reference Prompt mode semantically transcribes the official
template instead of copying its mojibake natural-language fragments.

## Additional diagnostic metrics

The standard columns `up_sim`, `down_sim`, and `similarity` reproduce the
official formula. The candidate pipeline additionally exports
`strict_action_up_sim`, `strict_action_down_sim`, and
`strict_action_similarity`, which compare parsed action lists and should be
reported as diagnostic metrics rather than substituted for official Sim2.
