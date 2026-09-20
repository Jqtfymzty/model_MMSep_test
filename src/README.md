# MMSep
Implementation instructions and details for MMSep

## Install
Follow your desired MLLM's repository for installation.

## Usage
Copy and paste functions in mm_separators.py to your desired MLLM's Language model class and call in layered transformer loop.
Copy and paste cache.py to your language model codebase, and replace the original Cache class with MMSepCache.

## Eval
Collect answers in LLaVA-v1.5 validation jsonl manner
and use auto_eval.py with your own evaluation model api.

More details are still under preperation.
