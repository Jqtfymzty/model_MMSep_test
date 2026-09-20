# Shared Server Runbook

This runbook is for the manually operated Ubuntu 22.04 shared server. It does
not authorize use of a GPU or directory: confirm the laboratory rules and the
allowed path with the server owner before starting.

The commands deliberately avoid passwords and API keys. Never put credentials
in a command line, config file, result file, or log.

## 1. Confirm permission and reserve a GPU

Confirm that `/data/lichenxi/mmsep` may temporarily use about 40–50 GB. Ask how
GPU booking is coordinated because the server does not have Slurm.

After login, inspect the current state:

```bash
nvidia-smi
```

Choose only a GPU approved by the owner and currently showing enough free
memory. Functional tests may use an otherwise idle shared GPU. Performance
tests require an exclusive, quiet interval; otherwise latency and throughput
measurements are invalid.

Do not stop, signal, or inspect the files of another user's process.

## 2. Prepare and review the source locally

On Windows, run the unit tests before transferring the reviewed bundle:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests\unit -v
Get-FileHash -Algorithm SHA256 artifacts\kaggle\mmsep_kaggle_source.zip
```

The bundle must not contain model weights, datasets, result files, `.env`
files, passwords, API keys, or SSH material.

## 3. Create the isolated server workspace

Run these commands yourself only after permission is confirmed:

```bash
mkdir -p /data/lichenxi/mmsep/incoming
mkdir -p /data/lichenxi/mmsep/project
mkdir -p /data/lichenxi/mmsep/models
mkdir -p /data/lichenxi/mmsep/runs
mkdir -p /data/lichenxi/mmsep/tmp
mkdir -p /data/lichenxi/mmsep/cases
mkdir -p /data/lichenxi/mmsep/input
chmod 700 /data/lichenxi
chmod -R u+rwX,go-rwx /data/lichenxi/mmsep
```

The model and virtual environment belong under `/data`, not the home directory
or the root filesystem.

## 4. Upload the reviewed bundle

From local PowerShell, substitute the configured SSH host alias where possible:

```powershell
scp -P <port> artifacts\kaggle\mmsep_kaggle_source.zip `
  <user>@<server>:/data/lichenxi/mmsep/incoming/
```

On the server, record the received digest and compare it with the local value:

```bash
sha256sum /data/lichenxi/mmsep/incoming/mmsep_kaggle_source.zip
```

Extract it with the standard-library ZIP tool:

```bash
cd /data/lichenxi/mmsep/project
python3 -m zipfile -e \
  /data/lichenxi/mmsep/incoming/mmsep_kaggle_source.zip \
  /data/lichenxi/mmsep/project
```

## 5. Create the Python 3.10 environment

```bash
python3 -m venv /data/lichenxi/mmsep/venv
source /data/lichenxi/mmsep/venv/bin/activate
python -m pip --version
python -m pip install --upgrade pip setuptools wheel
```

If virtual-environment creation reports that `ensurepip` or `python3-venv` is
missing, stop and ask the administrator to install the operating-system
package. Do not use `sudo` without explicit authorization.

Install the pinned CUDA wheel and project dependencies:

```bash
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -m pip install \
  -r /data/lichenxi/mmsep/project/requirements/llava-cloud-linux.txt
```

```bash
python -m pip install modelscope-hub
```

The server does not need `nvcc`: these are prebuilt PyTorch and bitsandbytes
wheels. Do not compile CUDA code unless the project later proves it necessary.

## 6. Install the reviewed LLaVA source

```bash
git clone --depth 1 --branch v1.2.0 \
  https://github.com/haotian-liu/LLaVA.git \
  /data/lichenxi/mmsep/LLaVA-v1.2.0
```

```bash
python -m pip install --no-deps -e /data/lichenxi/mmsep/LLaVA-v1.2.0
```

Record the exact LLaVA revision:

```bash
git -C /data/lichenxi/mmsep/LLaVA-v1.2.0 rev-parse HEAD
```

## 7. Download and verify the two model components once

```bash
ms-hub download huangjianuo/llava-v1.5-7b \
  --revision master \
  --local-dir /data/lichenxi/mmsep/models/llava-v1.5-7b
```

```bash
ms-hub download openai-mirror/clip-vit-large-patch14-336 \
  --revision master \
  --local-dir /data/lichenxi/mmsep/models/clip-vit-large-patch14-336
```

Verify the LLaVA index fingerprint:

```bash
sha256sum \
  /data/lichenxi/mmsep/models/llava-v1.5-7b/pytorch_model.bin.index.json
```

The expected digest for the frozen snapshot is:

```text
487578d3f7ec21c057596d0f9dc71084acc349eb8f13bd8b5af86c26460f2d31
```

## 8. Select one approved GPU

Check the server again immediately before starting:

```bash
nvidia-smi
```

Set the physical GPU number approved for this run. The example `4` is not a
permanent allocation and must be replaced when necessary:

```bash
export CUDA_VISIBLE_DEVICES=4
```

The runner records this selection and the corresponding physical GPU metrics.
Inside PyTorch the selected physical card is normally exposed as `cuda:0`.

## 9. Set non-secret runtime paths

```bash
export PYTHONPATH=/data/lichenxi/mmsep/project/src
export LLAVA_MODEL_PATH=/data/lichenxi/mmsep/models/llava-v1.5-7b
export LLAVA_VISION_TOWER_PATH=/data/lichenxi/mmsep/models/clip-vit-large-patch14-336
export LLAVA_CODE_PATH=/data/lichenxi/mmsep/LLaVA-v1.2.0
```

If the reviewed source was transferred as a ZIP without `.git`, record its
bundle digest as the source revision:

```bash
export MMSEP_SOURCE_REVISION=bundle-sha256-<reviewed-bundle-digest>
```

This variable is not a credential. Never export an API key in commands that
will be copied into reports or shell history.

## 10. Run local unit tests and preflight

```bash
cd /data/lichenxi/mmsep/project
python -m unittest discover -s tests/unit -v
```

```bash
python -m mmsep_testkit.preflight \
  --config configs/experiments/baseline.llava.example.json
```

Do not load the model until all preflight checks report `ok: true`.

## 11. Prepare one smoke case

Create a machine-local case file outside the repository. Its `image_ref` must
be an absolute path to a reviewed test image on the server:

```json
{"case_id":"M2-FUN-SMOKE-001","title":"Single-image Baseline smoke test","dimension":"functional","input":{"text":"Describe the main objects and scene in this image briefly.","image_ref":"/data/lichenxi/mmsep/input/smoke.png"},"formal":false,"metadata":{"source":"manual-server-smoke"}}
```

Save the single JSON object as one line in:

```text
/data/lichenxi/mmsep/cases/baseline-smoke.jsonl
```

## 12. Run inside tmux with complete logs

Create a run directory whose name identifies the time and mode:

```bash
mkdir -p /data/lichenxi/mmsep/runs/baseline-smoke-001
tmux new -s mmsep-smoke
```

Inside tmux, reactivate the environment and repeat the path/GPU exports from
steps 8 and 9. Then run:

```bash
python -m mmsep_testkit.runner \
  --config /data/lichenxi/mmsep/project/configs/experiments/baseline.llava.example.json \
  --cases /data/lichenxi/mmsep/cases/baseline-smoke.jsonl \
  --output /data/lichenxi/mmsep/runs/baseline-smoke-001/results.jsonl \
  --log /data/lichenxi/mmsep/runs/baseline-smoke-001/runlog.jsonl \
  > /data/lichenxi/mmsep/runs/baseline-smoke-001/console.log 2>&1
```

Detach from tmux with `Ctrl+B`, then `D`. Reattach later with:

```bash
tmux attach -t mmsep-smoke
```

The runner writes each completed result immediately and flushes each structured
log event to disk. If SSH disconnects or a later case fails, completed evidence
is retained.

## 13. Inspect and send diagnostic evidence

```bash
tail -n 50 /data/lichenxi/mmsep/runs/baseline-smoke-001/console.log
```

```bash
tail -n 20 /data/lichenxi/mmsep/runs/baseline-smoke-001/runlog.jsonl
```

Send these three small files for diagnosis:

- `console.log`
- `runlog.jsonl`
- `results.jsonl`

The structured log contains environment versions, selected GPU status,
preflight checks, model-load duration, per-case timing and metrics, output
hashes, adapter cleanup events, and tracebacks. It intentionally omits prompts, model
answers, complete environment variables, passwords, API keys, and other users'
process commands. Model answers remain in `results.jsonl`.

## 14. Export and verify evidence

After a successful run:

```bash
cd /data/lichenxi/mmsep/runs/baseline-smoke-001
sha256sum results.jsonl runlog.jsonl console.log > SHA256SUMS
tar -czf baseline-smoke-001-evidence.tar.gz \
  results.jsonl runlog.jsonl console.log SHA256SUMS
```

Download the archive with `scp`, verify the hashes locally, and keep it outside
the Git repository unless the project specification explicitly approves a
sanitized summary.

## 15. Release resources safely

First let the Python process exit normally and confirm that your account has no
remaining model process:

```bash
ps -u "$(id -un)" -o pid,etime,cmd
nvidia-smi
```

Inspect exactly what would be removed:

```bash
du -sh /data/lichenxi/mmsep/*
```

Only after the evidence has been downloaded and verified, remove the explicit
project paths required by the owner's policy. If everything must be released,
the removable targets are:

```text
/data/lichenxi/mmsep/venv
/data/lichenxi/mmsep/models
/data/lichenxi/mmsep/LLaVA-v1.2.0
/data/lichenxi/mmsep/project
/data/lichenxi/mmsep/tmp
/data/lichenxi/mmsep/incoming
```

Keep `runs` until its downloaded archive and checksums have been verified. Do
not use globs, `killall`, `pkill`, or broad paths such as `/data`, `/home`, or
`~` in cleanup commands. Do not alter system logs.
