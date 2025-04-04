# ProfilingDiT

## Official Implementation of ["Model Reveals What to Cache: Profiling-Based Feature Reuse for Video Diffusion Models"]
## [📄 Paper](docs/Model_Reveals_What_to_Cache__Profiling_Based_Feature_Reuse_for_Video_Diffusion_Models.pdf)

This repository contains the official implementation of our paper: *Model Reveals What to Cache: Profiling-Based Feature Reuse for Video Diffusion Models*. Please follow the official link for setting up the environment.

## Installation

Follow the official [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [WAN 2.1](https://github.com/Wan-Video/Wan2.1) environment setup guide.

## Running the Code

### HunyuanVideo

```sh
cd HunyuanVideo
python3 sample_video.py \
    --video-size 360 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "cat walk on grass" \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results \
    --seed 42 \
    --model-base "ckpts" \
    --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --delta_cache
```

### WAN 2.1

```sh
cd Wan2.1
python generate.py \
    --task t2v-14B \
    --size 832*480 \
    --frame_num 81 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --delta_cache
```

