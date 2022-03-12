#!/bin/bash

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

VOCAB_FILE=../data/megatron/data/gpt2/gpt2_vocab.json
MERGE_FILE=../data/megatron/data/gpt2/gpt2_merges.txt

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS tools/generate_text.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 24  \
       --hidden-size 1024  \
       --num-attention-heads 16  \
       --max-position-embeddings 1024  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 1024  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top-p 0.9  \
       --seed 42 \
       --prompts-file example_text/alice_in_wonderland.txt \
       --all-tokens-to-generate 0 1 2 4 8 12 16 24 32 48 64 \
       --all-num-input-tokens 1 16 256
