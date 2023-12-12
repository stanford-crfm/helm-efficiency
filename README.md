# Cheaply Evaluating Inference Efficiency Metrics for Autoregressive Transformer Models

This repository contains the code used to produce the results in "Cheaply Evaluating
Inference Efficiency Metrics for Autoregressive Transformer Models", accepted to
NeurIPS 2023.

This repository contains the following:
- `megatron_lm` is a Git sub-module with Megatron-LM checked out to the commit hash
used in our experiments.
  - `scripts` contains example scripts to profile models of interest on various prompt sizes
  and numbers of output tokens.
  - `logs` contains logfiles of profiling runs on V100 and A100 GPUs for various models.

- `notebooks` contains various IPython notebooks.
  - `notebooks/fit_runtimes.ipynb` is an IPython notebook that fits a linear regression
  model to the collected raw runtimes to determine per-model runtime of processing a prompt of a size
  ($\alpha_p$ in Equation 1) and the per-output-token runtimes ($\beta in Equation 2). These raw runtimes
  could be measured on a dedicated server deployment, or could be measured using
  black-box text generation APIs like OpenAI's `davinci` offering (in this repository,
  we use results from [HELM's synthetic efficiency
  scenario](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/synthetic_efficiency_scenario.py)
  that allows us to control both the prompt size and the number of output tokens).
  `fit_runtimes.ipynb` also dumps the learnt parameters into JSON files in `processed_jsons/`. 
  - The other notebooks in this directory produce visualizations shown in the paper (e.g.,
  end-to-end runtime versus number of output tokens for different models and prompt sizes,
  or prompt runtime versus prompt size for different models). These visualizations are
  collected in `figures/`.

## Using Megatron-LM for autoregressive text generation

We use Megatron-LM in our experiments to generate text.

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

## Citation

If you found this repository or our paper useful, feel free to cite our work:

```
@inproceedings{narayanan2023cheaply,
    title={{Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models}},
    author={Deepak Narayanan and Keshav Santhanam and Peter Henderson and Rishi Bommasani and Tony Lee and Percy Liang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=RJpAz15D0S}
}
```
