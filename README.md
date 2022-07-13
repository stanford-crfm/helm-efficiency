# Estimating Inference Efficiency using Megatron

This repository contains scripts to benchmark and analyze Megatron's text
generation functionality.


The main driver program is `generate_text.py`, which instantiates a model
of the passed-in dimensions, and then generates the desired number of
tokens given a prompt of a given size. The complete list of command line
arguments supported by `generate_text.py` can be produced using the
following command:
```bash
 PYTHONPATH=$PYTHONPATH:megatron_lm/ python generate_text.py -h
 ...
 text generation:
  --prompts-file PROMPTS_FILE
                        File with prompt (that can then be truncated)
  --all-num-input-tokens ALL_NUM_INPUT_TOKENS [ALL_NUM_INPUT_TOKENS ...]
                        Number of input tokens (must be > 0)
  --all-num-output-tokens ALL_NUM_OUTPUT_TOKENS [ALL_NUM_OUTPUT_TOKENS ...]
                        Number of tokens to generate
  --temperature TEMPERATURE
                        Sampling temperature.
  --top-p TOP_P         Top p sampling.
  --top-k TOP_K         Top k sampling.
  --out-seq-length OUT_SEQ_LENGTH
                        Size of the output generated text.
```

`scripts` contains some example scripts for various models of interest.

`logs` contains logfiles of experiment runs on V100 and A100 GPUs.

`notebooks/fit_runtimes.ipynb` is an IPython notebook that fits a linear regression
model to the collected raw runtimes to determine per-model per-output-token
runtimes as well as the cost of processing a prompt of a given number of
tokens. `fit_runtimes.ipynb` also dumps these parameters into JSON files
in `processed_jsons/`.
