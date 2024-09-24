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
mkdir llm_gguf

downloads=(
    "comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors text_encoder/t5xxl_fp8_e4m3fn.safetensors"
    "black-forest-labs/FLUX.1-dev text_encoder/model.safetensors text_encoder/clip_l.safetensors"
    "black-forest-labs/FLUX.1-dev vae/diffusion_pytorch_model.safetensors vae/flux_vae.safetensors"
    "black-forest-labs/FLUX.1-dev flux1-dev.safetensors unet/flux1-dev.safetensor"
    "black-forest-labs/FLUX.1-dev ae.safetensors vae/ae.safetensor"
    "XLabs-AI/flux-RealismLora lora.safetensors lora/realism.safetensors"
    "InstantX/FLUX.1-dev-Controlnet-Union diffusion_pytorch_model.safetensors controlnet/Controlnet-Union.safetensors"
    "cognitivecomputations/dolphin-2.9.4-llama3.1-8b-gguf dolphin-2.9.4-llama3.1-8b-Q6_K.gguf llm_gguf/dolphin-2.9.4-llama3.1-8b-Q6_K.gguf"
)

download_repo_files() {
    local repo=$1
    local file=$2
    local new_location=$3

    local dest_dir="/workspace/downloads"
    local folder
    folder=$(dirname "$new_location")

    mkdir -p "$dest_dir/$folder"
    huggingface-cli download "$repo" "$file" --local-dir "$dest_dir" && mv "$dest_dir/$file" "$dest_dir/$new_location"
}

( max_jobs=3
declare -A cur_jobs

for entry in "${downloads[@]}"; do
    IFS=' ' read -r repo_path file_name new_location <<< "$entry"

    echo "Downloading: $file_name"

    if (( ${#cur_jobs[@]} >= max_jobs )); then
        wait -n
        for pid in "${!cur_jobs[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                unset cur_jobs[$pid]
            fi
        done
    fi

    download_repo_files "$repo_path" "$file_name" "$new_location" & cur_jobs[$!]=1
done
touch /workspace/download.fin ) &
# Wait for all jobs to complete


#mkdir text_encoder
#mkdir vae
#mkdir unet
#mkdir lora
#mkdir controlnet
#
#(huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors --local-dir /workspace/downloads
#huggingface-cli download black-forest-labs/FLUX.1-dev text_encoder/model.safetensors vae/diffusion_pytorch_model.safetensors --local-dir /workspace/downloads
#mv t5xxl_fp8_e4m3fn.safetensors text_encoder/t5xxl_fp8_e4m3fn.safetensors
#mv text_encoder/model.safetensors text_encoder/clip_l.safetensors
#mv vae/diffusion_pytorch_model.safetensors vae/flux_vae.safetensors
#touch /workspace/downloads/hf_download_1.fin) &> hf_download_1.log &
#
#(huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors --local-dir /workspace/downloads/unet
#mv flux1-dev.safetensors unet/flux1-dev.safetensor
#mv ae.safetensors vae/ae.safetensor
#touch /workspace/downloads/hf_download_2.fin) &> hf_download_2.log &
#
#(huggingface-cli download XLabs-AI/flux-RealismLora lora.safetensors --local-dir /workspace/downloads
#mv lora.safetensors lora/realism.safetensors
#touch /workspace/downloads/hf_download_3.fin) &> hf_download_3.log &
#
#(huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Union diffusion_pytorch_model.safetensors --local-dir /workspace/downloads/controlnet
#mv controlnet/diffusion_pytorch_model.safetensors controlnet/Controlnet-Union.safetensors
#touch /workspace/downloads/hf_download_4.fin) &> hf_download_4.log &


apt update
apt -y install vim python3-pip curl wget git-lfs
git config --global http.version HTTP/1.1

# Install ComfyUI
cd /workspace
git clone --branch=v0.2.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install Nodes Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/SeargeDP/ComfyUI_Searge_LLM.git
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ..


cd /workspace/ComfyUI/models
rm -rf text_encoder VAE Stable-diffusion controlnet loras
ln -s /workspace/downloads/text_encoder text_encoder
ln -s /workspace/downloads/vae VAE
ln -s /workspace/downloads/unet Stable-diffusion
ln -s /workspace/downloads/controlnet controlnet
ln -s /workspace/downloads/lora lora
ln -s /workspace/downloads/llm_gguf llm_gguf

cd ..
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py --listen 
 
