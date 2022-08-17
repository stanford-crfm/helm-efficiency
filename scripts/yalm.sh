#!/bin/bash

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

VOCAB_FILE=data/gpt2/gpt2_vocab.json
MERGE_FILE=data/gpt2/gpt2_merges.txt

export PYTHONPATH=$PYTHONPATH:megatron_lm/

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS generate_text.py   \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 1  \
       --num-layers 80  \
       --hidden-size 10240  \
       --ffn-hidden-size 27308  \
       --num-attention-heads 128  \
       --max-position-embeddings 2048  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top-p 0.9  \
       --seed 42  \
       --prompts-file data/alice_in_wonderland.txt  \
       --all-num-output-tokens 1 2 4 8 12 16 24 32 48 64  \
       --all-num-input-tokens 1 16 32 64 128 192 256 320 384 428 472 512 576 640 704 768 832 896 960 1024 1152 1280 1408 1536 1664 1792 1920 2048
