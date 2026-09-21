# MMSep AI 测试实践

本仓库用于完成模块二“测试 AI”实践，对比启用和未启用 MMSep 推理时压缩的
LLaVA v1.5 7B 模型。

项目规范见 [`docs/MODULE2_AI_TEST_SPEC.md`](docs/MODULE2_AI_TEST_SPEC.md)。修改代码、
测试用例、评价指标、实验结果或报告材料前，请先阅读该规范。

如需人工操作实验室 GPU 服务器，请遵循
[`docs/SHARED_SERVER_RUNBOOK.md`](docs/SHARED_SERVER_RUNBOOK.md)。该文档规定了隔离环境、
结构化诊断日志以及安全释放资源的操作流程。

## 当前状态

仓库现已包含完整的 15 条测试用例及统一运行器。运行器只加载一次 LLaVA，使用相同的
生成参数分别执行未经修改的 Baseline 模式和 MMSep 缓存模式，并输出原始 JSONL 证据、
汇总数据、环境元数据、生成的输入以及 SHA-256 清单。测试用例文件保留了约定的分工：
`lcx` 负责 8 条用例，`wjn` 负责 7 条用例。

模型权重、完整数据集、API 密钥、未整理的运行产物、课程 PDF 和论文 PDF 均不应提交到
GitHub 仓库。

## 运行模式

- `baseline`：采用原始缓存行为的 LLaVA v1.5 7B。
- `mmsep`：使用相同的模型、权重、精度、输入和生成参数，并启用 MMSep。
- 旧版 `mock` 和仅运行 Baseline 的流程仍可用于轻量级开发检查，但不计入正式的
  15 条用例实验结果。

## 一条命令运行正式实验

运行前请选择一张空闲 GPU，并配置两个持久化模型路径。只有在该 GPU 未被其他任务占用时，
性能实验结果才有效。

```bash
export CUDA_VISIBLE_DEVICES=<approved-idle-gpu>
export LLAVA_MODEL_PATH=/data/lichenxi/llava-7B/models/llava-v1.5-7b
export LLAVA_VISION_TOWER_PATH=/data/lichenxi/llava-7B/models/clip-vit-large-patch14-336
export MMSEP_RUNS_ROOT=/data/lichenxi/llava-7B/runs
export SOURCE_COMMIT=<reviewed-local-commit>
python scripts/run_module2_experiment.py --run-id module2-formal-001
```

正式实验会在两种模式下运行全部 15 条用例。鲁棒性、安全性和公平性用例使用随机种子
`42`、`2026` 和 `925`；每条性能用例先预热 2 次，再进行 5 次正式测量。
`--quick --case-id M2-FUN-201` 只能用于非正式冒烟测试。进程正常结束时仍可能存在真实的
测试判定失败；应查看 `summary/summary.json`，不要将模型质量不达标一律视为基础设施故障。

传输到服务器之前可执行以下纯 CPU 验证：

```powershell
python -m unittest tests.ai.test_all_cases_runner `
  tests.ai.test_lcx_cases tests.ai.test_mmsep_runtime -v
```

## 本地配置

请复制示例配置，不要直接修改示例文件。与具体机器相关的路径不应写入版本控制文件。

```powershell
$env:LLAVA_MODEL_PATH = 'C:\path\to\llava-v1.5-7b'
$env:LLAVA_VISION_TOWER_PATH = 'C:\path\to\clip-vit-large-patch14-336'
# 未安装 LLaVA、仅检出其源码时可选配：
$env:LLAVA_CODE_PATH = 'C:\path\to\LLaVA'
$env:PYTHONPATH = "$PWD\src"
```

API 凭据只能通过环境变量提供。可选的外部评审模型使用 `DEEPSEEK_API_KEY`；`mock` 流程
不会调用该密钥。

## 不依赖模型权重的检查

以下命令只需要 Python 标准库：

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

JSONL 运行日志记录不含敏感信息的运行时版本、GPU 状态、预检项目、适配器生命周期、
逐用例进度、输出哈希、评价指标和完整异常堆栈。每条结果和日志事件都会立即写入磁盘，
因此共享服务器任务即使中断，也不会丢失此前已经完成的用例。

夹具中的用例只用于开发，不计入要求的 15 条正式 AI 测试用例。

## 真实 Baseline 预检

记录 ModelScope 快照路径并通过环境变量配置。请复制纳入版本控制的示例文件，不要将绝对路径
写入 Git：

```powershell
$env:LLAVA_MODEL_PATH = 'C:\Users\Administrator\.cache\modelscope\models\huangjianuo--llava-v1.5-7b\snapshots\master'
$env:LLAVA_VISION_TOWER_PATH = 'C:\path\reported\by\modelscope\for\clip-vit-large-patch14-336'
Copy-Item configs\experiments\baseline.llava.example.json `
  configs\experiments\baseline.llava.local.json
```

仓库中的配置使用模型权重索引的 SHA-256 摘要固定该快照版本。视觉编码器需要单独下载：

```powershell
modelscope download --model openai-mirror/clip-vit-large-patch14-336
```

请使用独立的 Python 3.10 环境。根据 `requirements/llava-baseline.txt` 安装官方推理依赖版本，
然后运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mmsep_testkit.preflight `
  --config configs\experiments\baseline.llava.local.json
```

只有所有预检项目均显示 `ok: true` 后，才能开始正式实验。本地 RTX 3050 只有 4 GB 显存，
低于本项目进行 4 位量化加载所设定的 6 GB 安全下限。实际运行应使用至少 8 GB 显存的云端
GPU；配置文件和运行器无需修改即可复用。

## Colab Baseline 冒烟测试

免费 Colab 流程请使用
[`notebooks/colab_baseline.ipynb`](notebooks/colab_baseline.ipynb)。该笔记本会检查 GPU 显存
是否至少为 12 GB，创建隔离的 Python 3.10 环境，下载模型的两个组成部分，执行相同的预检，
并运行一条非正式图像用例。笔记本提示时，请上传已经审核的
`artifacts/colab/mmsep_colab_source.zip` 源码包。

该笔记本不包含 API 密钥，也不会向 GitHub 推送内容。模型文件保存在 Colab 临时的
`/content` 存储空间中；只需将小型 JSONL 结果复制到 Google Drive。

## Kaggle Baseline 冒烟测试（推荐）

请使用 [`notebooks/kaggle_baseline.ipynb`](notebooks/kaggle_baseline.ipynb)。在 Kaggle 中启用
GPU 和互联网访问，然后将已经审核的 `artifacts/kaggle/mmsep_kaggle_source.zip` 以及一张
测试图片添加为 Notebook 输入。笔记本会将环境和模型的两个组成部分存放在非持久化暂存盘中。
只有测试证据会写入 `/kaggle/working/mmsep_results`，因此大型模型不会占用已保存的
Notebook Output 配额。

## GitHub 安全门禁

AI 工具可以准备本地修改和差异，但在没有针对当前操作重新进行人工审核并获得明确许可前，
不得自行推送、上传、创建拉取请求或执行合并。
