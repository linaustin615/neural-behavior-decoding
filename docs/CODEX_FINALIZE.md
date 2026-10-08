# Codex: finalize and publish the 2026-10-07 results

Repository: `/Users/austinlin/neuron_transformer`, remote `https://github.com/linaustin615/neural-behavior-decoding` (branch `main`). All work below is complete, audited and **uncommitted**. Your job is to update the README, commit and push. Do **not** retrain, rescore, or rerun any experiment stage: evaluation scripts refuse existing outputs, and rerunning would count as another look at the test data.

## The finding to publish

> Across 11 mice, including 4 never used in development, averaging 2–3 independently trained decoders consistently reduced test error by 5–8% and never made a mouse worse. Transformer and MLP decoders performed similarly; combining the two architectures added nothing beyond ordinary ensembling.

Present it as a robust, well-controlled practical result, **not** a discovery or a statistically significant claim. Ensembling is a known technique, and there are only 4 new mice: a sign test gives p = 0.125 on the 4 new mice and p ≈ 0.001 on all 11, but the 11-mouse figure is a post hoc analysis.

## Study chain (read each `ASSESSMENT.md` for details)

| Order | Folder (`experiments/`) | Outcome |
| --- | --- | --- |
| 1 | `2026-10-06_frozen_retrieval/` | Retrieval heads worse than original heads; all gates fail |
| 2 | `2026-10-06_movement_gate/` | Quiet periods improve, but running periods get worse; gates fail |
| 3 | `2026-10-07_bounded_hybrid/` | Blend + learned correction beats both originals on 7/7 development mice, but fails the gates against the blend (TX61 running-period harm). The gain is mostly the blend; the correction only helps on TX104/TX61 |
| 4 | `2026-10-07_correction_scaling/` | No-training strength/direction tuning gives +0.4%; still fails. Not blind |
| 5 | `2026-10-07_holdout_confirmation/` | **Pre-registered test on new mice D3/D4/D7/D9 (sensorimotor). The blend PASSES against both models (+5.64% vs transformer, +5.34% vs MLP, 4/4 mice).** The correction fails against the blend (+1.16%). A same-cost control shows that two seeds of one family match the blend, so the gain is ensembling |
| 6 | `2026-10-07_ensemble_check/` | **Two-seed average vs single model: +5.4–6.1% on 7/7 development and 4/4 holdout mice, both families, every gate passes, no mouse harmed. Three seeds: +7–8%** |

Holdout data notes, to mention briefly in the README:
- The public TX61/VR2 later sessions contain D7's running trace (a publisher packaging error), so they were excluded.
- D8 was excluded by a preset rule (flat training target).
- D3, D4, D7 and D9 have a 2-frame activity/running length difference. They were start-aligned under an amendment approved before any data value was read.

See `data_issues.json`, `amendment.json` and `exclusions.json` in the holdout folder.

## Already updated (keep these edits)

- `portfolio/RESEARCH_REPORT.md`: paragraphs for each study above, under "The evaluated architecture"
- `research/EXPERIMENTS.md`: index entries at the top
- `docs/CLAUDE_HANDOFF.md`: running state log (internal; fine to commit)
- `PROJECT_HANDOFF.md`: local and gitignored; leave it

## Steps

1. **Read first:** `README.md`, `experiments/2026-10-07_ensemble_check/ASSESSMENT.md` and `experiments/2026-10-07_holdout_confirmation/ASSESSMENT.md`.
2. **Edit `README.md` "Findings worth keeping":**
   - Add the ensembling finding as the lead result, with the small table from the ensemble check.
   - Add one line on the holdout process: protocol frozen before download, corrupted public files caught, audits passed.
   - Add one line of caveats: 11 mice, one lab, known technique, and the 11-mouse analysis is post hoc.
   - Keep the existing headline that the transformer did not beat the MLP. These results reinforce it.
   - Do not overstate. Do not call it significant, novel or state of the art.
3. **Check the diff.** Run `git status` and `git add -n docs experiments portfolio research README.md`. This should come to about 505 small text/JSON files, roughly 3.9 MB. `.gitignore` already excludes `*.npz`, `*.pt`, `*.npy`, `*.zip` and `data/`. Confirm no binary or `data/holdout` file is staged; the raw recordings are about 28 GB and CC-BY-NC.
4. **Commit on a branch and push** (the user asked to finalize for GitHub). Stage the specific paths rather than `git add -A`:
   ```sh
   git checkout -b results-2026-10-07
   git add README.md docs/CODEX_FINALIZE.md docs/CLAUDE_HANDOFF.md portfolio/RESEARCH_REPORT.md research/EXPERIMENTS.md \
     experiments/2026-10-06_frozen_retrieval experiments/2026-10-06_movement_gate experiments/2026-10-07_bounded_hybrid \
     experiments/2026-10-07_correction_scaling experiments/2026-10-07_holdout_confirmation experiments/2026-10-07_ensemble_check
   git commit -m "Add holdout confirmation and ensembling results; preserve failed hybrid studies"
   git push -u origin results-2026-10-07
   ```
   Then confirm with the user before merging into `main`, or open a PR with `gh pr create`.
5. **Do not touch** `decoding/` (the application code, unchanged by these studies), the root `train.py` and `data.py` (legacy files with side effects), any `run.py` stage, or locked JSON files (`protocol.json`, `*_lock.json`, `amendment.json`, `summary.json`, `audit.json`). Editing locked files breaks the recorded hashes.

## Verification already done (no need to repeat)

- Every study has an independent `audit.py`/`audit.json` that passed, with all source, checkpoint and data hashes verified.
- The ensemble check's single-model inputs matched the audited test records to 1e-12.
- `completion_manifest.json` in the bounded-hybrid and holdout folders lists the SHA-256 of every file.
- The holdout download was verified by archive md5 (`2ae0287e…`, `e00e27ee…`) and per-file size and CRC32 (`download_verification.json`, `data_issues.json`).

## If the user wants more later

The project's question is answered. Further confirmation would need a new dataset (Sensorium 2023 or the Allen Brain Observatory both include running speed), with a protocol frozen before download. Don't tune anything further on the current 11 mice: their test sets are used.
