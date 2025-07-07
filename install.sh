#!/bin/bash
set -x

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/flux_lora/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

pip install huggingface_hub
# Log Into Huggingface
git config --global credential.helper store
huggingface-cli login --token $HFTK --add-to-git-credential

cd /workspace

### Downloads
mkdir -p /workspace/downloads
mkdir -p /workspace/cache_downloads

downloads=(
    # Diffusion
    "black-forest-labs/FLUX.1-dev flux1-dev.safetensors diffusion_models/flux1-dev_orig.safetensors"
    #"lllyasviel/flux1_dev flux1-dev-fp8.safetensors diffusion_models/flux1-dev-fp8-illyasviel.safetensors"
    "bdsqlsz/flux1-dev2pro-single flux1-dev2pro.safetensors diffusion_models/flux-dev-2pro.safetensors"
    #"XLabs-AI/flux-dev-fp8 flux-dev-fp8.safetensors diffusion_models/flux-dev-fp8-xlab.safetensors"
    #"Kijai/flux-fp8 flux1-dev-fp8-e5m2.safetensors diffusion_models/flux-dev-fp8-e5m2.safetensors"
    # VAE
    "black-forest-labs/FLUX.1-dev ae.safetensors vae/ae.safetensors"
    "black-forest-labs/FLUX.1-dev vae/diffusion_pytorch_model.safetensors vae/vae_diffusion.safetensors"
    # Text Encoders
    "comfyanonymous/flux_text_encoders clip_l.safetensors text_encoders/clip_l.safetensors"
    "comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn_scaled.safetensors text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"
    # Other
    # "http https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/127658/acorn20is20spinning.htuc.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22acornIsSpinningFLUX_aisfV169.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20250706/us-east-1/s3/aws4_request&X-Amz-Date=20250706T191609Z&X-Amz-SignedHeaders=host&X-Amz-Signature=4e11233e9bfac9b0a8f3cbc3730f23f8e604855794dadcd14777e9d04cce10c8 diffusion_models/nsfw.safetensors"
    # "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta diffusion_pytorch_model.safetensors controlnet/flux-inpaint.safetensors"
)

download_repo_files() {
    local repo_name=$1
    local repo_file=$2
    local new_location=$3
    local new_location_folder=$(dirname "$new_location")

    local cache_dir="/workspace/cache_downloads"
    local dest_dir="/workspace/downloads"

    
    # Break if File already exists
    if [ -e "$dest_dir/$new_location" ]; then
      echo "File Already Downloaded: $new_location"
      return 0
    fi

    mkdir -p "$dest_dir/$new_location_folder"

    if [ "$repo" != "http" ]; then
        huggingface-cli download "$repo_name" "$repo_file" --local-dir "$cache_dir" && mv "$cache_dir/$repo_file" "$dest_dir/$new_location"
    else
        curl -o "$dest_dir/$new_location" "$repo_file"
    fi

    # Create SymLink to ComfyUI
    local comfy_models="/workspace/ComfyUI/models"
    if [ ! -L "$comfy_models/$new_location_folder" ]; then
      while [ ! -d "$comfy_models" ]; do
        sleep 5
      done
      echo "Creating SymLink for $dest_dir/$new_location_folder"
      rm -rf $comfy_models/$new_location_folder
      ln -s $dest_dir/$new_location_folder $comfy_models
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


if [ -d "/workspace/ComfyUI" ]; then
  echo "Skipping Installation; Start ComfyUI"
  cd /workspace/ComfyUI
  source venv/bin/activate
  python3 main.py --listen
  echo "Now exiting ComfyUI"
  exit 0
fi

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
git clone https://github.com/kijai/ComfyUI-KJNodes # Dependency
git clone https://github.com/rgthree/rgthree-comfy.git # Dependency
cd ..


python3 -m venv venv

pip install -r /workspace/ComfyUI/custom_nodes/custom_nodes/ComfyUI-KJNodes/requirements.txt
pip install -r /workspace/ComfyUI/custom_nodes/custom_nodes/rgthree-comfy/requirements.txt
pip install -r /workspace/ComfyUI/custom_nodes/custom_nodes/ComfyUI-FluxTrainer/requirements.txt

source venv/bin/activate
pip install -r requirements.txt
python3 main.py --listen 
 
