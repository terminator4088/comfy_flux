#!/bin/bash
set -x

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/flux_lora/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

if [ -d "/workspace/ComfyUI" ]; then
  echo "Peter: Skipping Installation"
  cd /workspace/ComfyUI
  source venv/bin/activate
  python3 main.py --listen
  echo "Peter: Now exiting"
  exit 0
fi


pip install huggingface_hub
# Log Into Huggingface
git config --global credential.helper store
huggingface-cli login --token $HFTK --add-to-git-credential

cd /workspace

### Downloads
mkdir downloads
mkdir cache_downloads
cd downloads

mkdir text_encoder
mkdir vae
mkdir unet
mkdir lora
mkdir controlnet
mkdir llm_gguf

downloads=(
    "black-forest-labs/FLUX.1-Kontext-dev vae/diffusion_pytorch_model.safetensors vae/flux_vae.safetensors"
    "Kijai/flux-fp8 flux1-dev-fp8-e5m2.safetensors diffusion_models/flux-dev-fp8-e5m2.safetensors"
    #"XLabs-AI/flux-dev-fp8 flux-dev-fp8.safetensors diffusion_models/flux-dev-fp8.safetensors"
    "comfyanonymous/flux_text_encoders clip_l.safetensors text_encoders/clip_l.safetensors"
    "comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn_scaled.safetensors text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"
    "http https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/289798/redKFm00NSFWEditorFP8.Wtdk.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22redcraftCADSUpdatedJUN29_redKKingOfHearts.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20250702/us-east-1/s3/aws4_request&X-Amz-Date=20250702T145948Z&X-Amz-SignedHeaders=host&X-Amz-Signature=871241f362ddd7804ec6903608af15000529f463e2e53aca51e44147dc59329f diffusion_models/nsfw.safetensors"
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta diffusion_pytorch_model.safetensors controlnet/flux-inpaint.safetensors"
    #"Comfy-Org/flux1-kontext-dev_ComfyUI split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors diffusion_models/flux-kontext-fp8.safetensors"
    #"Comfy-Org/Lumina_Image_2.0_Repackaged split_files/vae/ae.safetensors vae/flux-kontext-ae.safetensors"
    #"black-forest-labs/FLUX.1-Fill-dev flux1-fill-dev.safetensor diffusion_models/flux1-fill-dev.safetensors"
)

download_repo_files() {
    local repo=$1
    local file=$2
    local new_location=$3

    local cache_dir="/workspace/cache_downloads"
    local dest_dir="/workspace/downloads"
    local folder
    folder=$(dirname "$new_location")

    mkdir -p "$dest_dir/$folder"
    mkdir -p "$cache_dir/$folder"

    if [ "$repo" != "http" ]; then
        huggingface-cli download "$repo" "$file" --local-dir "$cache_dir" && mv "$cache_dir/$file" "$dest_dir/$new_location"
    else
        curl -o "$dest_dir/$new_location" "$file"
    fi
    
}

( max_jobs=4
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

cat > /etc/apt/apt.conf.d/99mytimeout <<EOF
Acquire::http::Timeout "9";
Acquire::https::Timeout "9";
EOF

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
git clone https://github.com/alimama-creative/FLUX-Controlnet-Inpainting.git # FLUX Inpainting
git clone https://github.com/kijai/ComfyUI-FluxTrainer.git # FluxTrainer
git clone https://github.com/fofr/comfyui-basic-auth # Basic Auth
cd ..


cd /workspace/ComfyUI/models
rm -rf text_encoder VAE Stable-diffusion controlnet loras
ln -s /workspace/downloads/text_encoders text_encoders
ln -s /workspace/downloads/vae vae
ln -s /workspace/downloads/diffusion_models diffusion_models
ln -s /workspace/downloads/controlnet controlnet
ln -s /workspace/downloads/lora lora
ln -s /workspace/downloads/llm_gguf llm_gguf

cd ..
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py --listen 
 
