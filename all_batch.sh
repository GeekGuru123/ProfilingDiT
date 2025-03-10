#!/bin/bash

BASE_DIR="/home/xuran/ProfilingDiT"
RESULTS_DIR="$BASE_DIR/HunyuanVideo/alternate"

bash "$BASE_DIR/cal_vbench.sh" "$RESULTS_DIR"
python "$BASE_DIR/parallel_cal_psnr_ssim_lpips_fid.py" --cmp_dir "$RESULTS_DIR"
python "$BASE_DIR/collect_all_vench.py"
rm -rf /home/xuran/ProfilingDiT/evaluation_results