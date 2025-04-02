#!/bin/bash

MODEL_BASE="/persistent/app_user_data/models"
DIT_WEIGHT="$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"


torchrun --nproc_per_node=8 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=4 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=2 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=1 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=8 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 69 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=4 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 69 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=2 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 69 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=1 --master_port=29503 sample_video.py \
    --video-size 720 1280 \
    --video-length 69 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=8 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=4 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=2 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=1 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=8 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 49 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=4 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 49 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=2 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 49 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache

torchrun --nproc_per_node=1 --master_port=29503 sample_video.py \
    --video-size 360 640 \
    --video-length 49 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --model-base "$MODEL_BASE" \
    --dit-weight "$DIT_WEIGHT" \
    --save-path ./results \
    --delta_cache
