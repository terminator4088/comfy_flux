#!/bin/bash

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/runpod/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

pip install huggingface_hub
# Log Into Huggingface
git config --global credential.helper store
huggingface-cli login --token $HFTK --add-to-git-credential

cd /workspace

### Downloads
mkdir downloads
cd downloads
(huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors --local-dir /workspace/downloads
huggingface-cli download black-forest-labs/FLUX.1-dev text_encoder/model.safetensors vae/diffusion_pytorch_model.safetensors --local-dir /workspace/downloads
touch /workspace/downloads/hf_download_1.fin) &> hf_download_1.log &

(huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir /workspace/downloads
touch /workspace/downloads/hf_download_2.fin) &> hf_download_2.log &

(huggingface-cli download XLabs-AI/flux-RealismLora lora.safetensors --local-dir /workspace/downloads
touch /workspace/downloads/hf_download_3.fin) &> hf_download_3.log &

(huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Union diffusion_pytorch_model.safetensors --local-dir /workspace/downloads
touch /workspace/downloads/hf_download_4.fin) &> hf_download_4.log &

cd /workspace

apt update
apt -y install vim python3-pip curl wget git-lfs



# Install ComfyUI
git clone --branch=v0.2.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install Nodes Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
