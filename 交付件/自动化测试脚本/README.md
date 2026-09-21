# 模块二自动化测试脚本

`run_module2_tests.py` 是交付件中的统一入口。它调用项目根目录下的正式实验编排器，使用同一套
用例、适配器和配置依次运行 Baseline 与 MMSep，默认执行全部 15 条用例。

## 环境变量

```bash
export CUDA_VISIBLE_DEVICES=<空闲GPU编号>
export LLAVA_MODEL_PATH=/data/lichenxi/llava-7B/models/llava-v1.5-7b
export LLAVA_VISION_TOWER_PATH=/data/lichenxi/llava-7B/models/clip-vit-large-patch14-336
export MMSEP_RUNS_ROOT=/data/lichenxi/llava-7B/runs
export SOURCE_COMMIT=<已审核的提交号>
```

## 正式运行

在项目根目录执行：

```bash
python 交付件/自动化测试脚本/run_module2_tests.py --run-id module2-formal-001
```

脚本默认运行全部用例和两种模式。结果写入
`$MMSEP_RUNS_ROOT/<run-id>/`，其中包含原始 JSONL、环境快照、汇总和哈希清单。

## 冒烟测试

```bash
python 交付件/自动化测试脚本/run_module2_tests.py \
  --quick --case-id M2-FUN-201 --run-id module2-smoke-001
```

`--quick` 只用于验证环境和调用链，不计入正式实验结果。

## 不依赖模型权重的代码测试

```bash
PYTHONPATH=src python -m pytest -q
```

正式实验必须在 Baseline 与 MMSep 共享模型权重、精度、输入、随机种子和生成参数的条件下执行。
