conda create -n wan python==3.10.9 -y
cd Wan2.1
# Ensure torch >= 2.4.0
pip install -r requirements.txt
pip install einops flash-attn