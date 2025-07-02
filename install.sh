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
    "Comfy-Org/flux1-kontext-dev_ComfyUI split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors unet/flux-kontext-fp8.safetensor"
    "comfyanonymous/flux_text_encoders clip_l.safetensors unet/flux1-kontext.safetensor"
    "comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn_scaled.safetensors text_encoder/t5xxl_fp8_e4m3fn_scaled.safetensors"
    "Comfy-Org/Lumina_Image_2.0_Repackaged split_files/vae/ae.safetensors vae/flux-kontext-ae.safetensor"
)

download_repo_files() {
    local repo=$1
    local file=$2
    local new_location=$3

    local dest_dir="/workspace/downloads"
    local folder
    folder=$(dirname "$new_location")

    mkdir -p "$dest_dir/$folder"

    if ( $repo -ne "http"); then
        echo "OK"
        echo "huggingface-cli download \"$repo\" \"$file\" --local-dir \"$dest_dir\" && mv \"$dest_dir/$file\" \"$dest_dir/$new_location\""
        huggingface-cli download "$repo" "$file" --local-dir "$dest_dir" && mv "$dest_dir/$file" "$dest_dir/$new_location"
    else
        curl -o "$dest_dir/$new_location" "$file"
    fi
    
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


apt update
apt -y install vim python3-pip curl wget git-lfs
git config --global http.version HTTP/1.1

# Install ComfyUI
cd /workspace
git clone --branch=v0.3.43 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install Nodes Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
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
 
