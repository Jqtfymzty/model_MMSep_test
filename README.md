# MMSep AI Testing Practice

This repository contains the Module 2 "Test AI" practice for comparing
LLaVA v1.5 7B with and without MMSep inference-time compression.

The project specification is
[`docs/MODULE2_AI_TEST_SPEC.md`](docs/MODULE2_AI_TEST_SPEC.md). Read it before
changing code, cases, metrics, results, or report assets.

For the manually operated laboratory GPU server, follow
[`docs/SHARED_SERVER_RUNBOOK.md`](docs/SHARED_SERVER_RUNBOOK.md). It keeps the
environment isolated, records structured diagnostic logs, and defines a safe
resource-release procedure.

## Current status

The repository now contains the complete 15-case suite and one unified runner.
It loads LLaVA only once, runs the unchanged Baseline and the MMSep cache mode
with the same generation parameters, and writes raw JSONL evidence, summaries,
environment metadata, generated inputs, and a SHA-256 manifest. The agreed
split is preserved in the case files: `lcx` owns 8 cases and `wjn` owns 7.

No model weights, full datasets, API keys, raw run artifacts, course PDFs, or
paper PDFs belong in the GitHub repository.

## Execution modes

- `baseline`: LLaVA v1.5 7B with its original cache behavior.
- `mmsep`: the same model, weights, precision, inputs, and generation settings
  with MMSep enabled.
- The older `mock` and Baseline-only workflow remains available for lightweight
  development checks; it is not part of the formal 15-case result.

## One-command formal experiment

Before the run, select an idle GPU and set the two persistent model paths. A
performance run is valid only while that GPU is otherwise idle.

```bash
export CUDA_VISIBLE_DEVICES=<approved-idle-gpu>
export LLAVA_MODEL_PATH=/data/lichenxi/llava-7B/models/llava-v1.5-7b
export LLAVA_VISION_TOWER_PATH=/data/lichenxi/llava-7B/models/clip-vit-large-patch14-336
export MMSEP_RUNS_ROOT=/data/lichenxi/llava-7B/runs
export SOURCE_COMMIT=<reviewed-local-commit>
python scripts/run_module2_experiment.py --run-id module2-formal-001
```

The formal policy runs all 15 cases in both modes. Robustness, security, and
fairness cases use seeds `42`, `2026`, and `925`; each performance case uses 2
warmups followed by 5 measured runs. Use `--quick --case-id M2-FUN-201` only
for a non-formal smoke test. A completed process may still contain genuine
failed test verdicts; consult `summary/summary.json` instead of treating every
model-quality failure as an infrastructure crash.

CPU-only validation before transfer:

```powershell
python -m unittest tests.ai.test_all_cases_runner `
  tests.ai.test_lcx_cases tests.ai.test_mmsep_runtime -v
```

## Local configuration

Copy an example config instead of editing it in place. Keep machine-specific
paths outside versioned files.

```powershell
$env:LLAVA_MODEL_PATH = 'C:\path\to\llava-v1.5-7b'
$env:LLAVA_VISION_TOWER_PATH = 'C:\path\to\clip-vit-large-patch14-336'
# Optional when LLaVA is checked out instead of installed:
$env:LLAVA_CODE_PATH = 'C:\path\to\LLaVA'
$env:PYTHONPATH = "$PWD\src"
```

API credentials must be supplied only through environment variables. The
optional external judge uses `DEEPSEEK_API_KEY`; it is not used by the mock
workflow.

## Weight-independent checks

These commands require only Python's standard library:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests\unit -v
python -m mmsep_testkit.preflight --config configs\experiments\smoke.mock.json
python -m mmsep_testkit.runner `
  --config configs\experiments\smoke.mock.json `
  --cases tests\fixtures\mock_cases.jsonl `
  --output artifacts\raw\mock-run.jsonl `
  --log artifacts\raw\mock-run.runlog.jsonl
```

The JSONL run log records non-secret runtime versions, GPU state, preflight
checks, adapter lifecycle, per-case progress, output hashes, metrics, and full
exception tracebacks. Each result and log event is flushed to disk immediately
so a shared-server interruption does not discard earlier completed cases.

The fixture case is development-only and does not count toward the required 15
formal AI test cases.

## Real Baseline preflight

Record the ModelScope snapshot path and set it through the environment. Copy the
versioned example rather than storing an absolute path in Git:

```powershell
$env:LLAVA_MODEL_PATH = 'C:\Users\Administrator\.cache\modelscope\models\huangjianuo--llava-v1.5-7b\snapshots\master'
$env:LLAVA_VISION_TOWER_PATH = 'C:\path\reported\by\modelscope\for\clip-vit-large-patch14-336'
Copy-Item configs\experiments\baseline.llava.example.json `
  configs\experiments\baseline.llava.local.json
```

The checked-in configuration pins this snapshot using the SHA-256 digest of
its weight index. The vision tower is a separate required download:

```powershell
modelscope download --model openai-mirror/clip-vit-large-patch14-336
```

Use a dedicated Python 3.10 environment. Install the official inference
package versions from `requirements/llava-baseline.txt`, then run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mmsep_testkit.preflight `
  --config configs\experiments\baseline.llava.local.json
```

Do not attempt a formal run until every preflight check reports `ok: true`.
The local RTX 3050 has only 4 GB VRAM, below this project's 6 GB safety floor
for 4-bit loading. Use a cloud GPU with at least 8 GB VRAM for the actual run;
the same config and runner can be reused unchanged.

## Colab Baseline smoke test

Use [`notebooks/colab_baseline.ipynb`](notebooks/colab_baseline.ipynb) for the
free-Colab workflow. It checks for at least 12 GB GPU memory, creates an
isolated Python 3.10 environment, downloads both model components, runs the
same preflight, and executes one non-formal image case. Upload the reviewed
`artifacts/colab/mmsep_colab_source.zip` bundle when the notebook requests it.

The notebook does not contain API keys and does not push to GitHub. Model files
stay in Colab's temporary `/content` storage; only the small JSONL result should
be copied to Google Drive.

## Kaggle Baseline smoke test (recommended)

Use [`notebooks/kaggle_baseline.ipynb`](notebooks/kaggle_baseline.ipynb). In
Kaggle, enable a GPU and Internet, then add the reviewed
`artifacts/kaggle/mmsep_kaggle_source.zip` plus one test image as Notebook
inputs. The notebook stores the environment and both model components on the
non-persistent scratch disk. Only test evidence is written to
`/kaggle/working/mmsep_results`, so the large model does not consume the saved
Notebook Output allowance.

## GitHub safety gate

AI tools may prepare local changes and diffs but must never push, upload, open a
pull request, or merge without a fresh human review and explicit approval for
that exact action.
