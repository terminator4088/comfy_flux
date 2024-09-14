#!/bin/bash
set -x

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/flux_lora/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

pip install huggingface_hub
# Log Into Huggingface
git config --global credential.helper store
huggingface-cli login --token $HFTK --add-to-git-credential

cd /workspace

### Downloads
mkdir downloads
cd downloads
mkdir text_encoder
mkdir vae
mkdir unet
mkdir lora
mkdir controlnet

(huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors --local-dir /workspace/downloads
huggingface-cli download black-forest-labs/FLUX.1-dev text_encoder/model.safetensors vae/diffusion_pytorch_model.safetensors --local-dir /workspace/downloads
mv t5xxl_fp8_e4m3fn.safetensors text_encoder/t5xxl_fp8_e4m3fn.safetensors
mv text_encoder/model.safetensors text_encoder/clip_l.safetensors
mv vae/diffusion_pytorch_model.safetensors vae/flux_vae.safetensors
touch /workspace/downloads/hf_download_1.fin) &> hf_download_1.log &

(huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors --local-dir /workspace/downloads/unet
mv flux1-dev.safetensors unet/flux1-dev.safetensor
mv ae.safetensors vae/ae.safetensor
touch /workspace/downloads/hf_download_2.fin) &> hf_download_2.log &

(huggingface-cli download XLabs-AI/flux-RealismLora lora.safetensors --local-dir /workspace/downloads
mv lora.safetensors lora/realism.safetensors
touch /workspace/downloads/hf_download_3.fin) &> hf_download_3.log &

(huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Union diffusion_pytorch_model.safetensors --local-dir /workspace/downloads/controlnet
mv controlnet/diffusion_pytorch_model.safetensors controlnet/Controlnet-Union.safetensors
touch /workspace/downloads/hf_download_4.fin) &> hf_download_4.log &

cd /workspace

apt update
apt -y install vim python3-pip curl wget git-lfs


cat >/dev/null <<ignore
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
ignore

git config --global http.version HTTP/1.1
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git


cd stable-diffusion-webui-forge/models
rm -rf text_encoder VAE Stable-diffusion controlnet loras
ln -s /workspace/downloads/text_encoder text_encoder
ln -s /workspace/downloads/vae VAE
ln -s /workspace/downloads/unet Stable-diffusion
ln -s /workspace/downloads/controlnet controlnet
ln -s /workspace/downloads/lora lora
cd ..

python3 -m venv venv
source venv/bin/activate
pip install -r requirements_versions.txt
./webui.sh -f --listen
 