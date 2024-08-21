#!/bin/bash

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/runpod/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

apt update
apt -y install vim
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx

# Install SimpleTuner
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

python3 -m venv .venv
source .venv/bin/activate
pip install -U poetry pip
pip install huggingface_hub

# Log Into Huggingface
huggingface-cli login --token $HFTK --add-to-git-credential

# Install Dependencie
poetry install --no-root

# May Be needed
pip uninstall -y deepspeed bitsandbytes

# Download working Diffusers
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers

#
cat >config/config.env <<'EOF'
export MODEL_TYPE='lora'

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export FLUX=true
export FLUX_GUIDANCE_VALUE=1.0
export FLUX_LORA_TARGET=all # options: 'all+ffs', 'all', 'context', 'mmdit', 'ai-toolkit'

export CONTROLNET=false

export USE_DORA=false

export RESUME_CHECKPOINT="latest"

export CHECKPOINTING_STEPS=150
export CHECKPOINTING_LIMIT=2

export LEARNING_RATE=1e-5 #@param {type:"number"}

# Max number of steps OR epochs can be used. Not both.
export MAX_NUM_STEPS=30000
# Will likely overtrain, but that's fine.
export NUM_EPOCHS=0

export DATALOADER_CONFIG="config/multidatabackend.json"
export OUTPUT_DIR="output/models"

# Set this to "true" to push your model to Hugging Face Hub.
export PUSH_TO_HUB="true"
export PUSH_CHECKPOINTS="true"
# This will be the model name for your final hub upload, eg. "yourusername/yourmodelname"
# It defaults to the wandb project name, but you can override this here.
export HUB_MODEL_NAME=$TRACKER_PROJECT_NAME

export RESOLUTION=1024
export RESOLUTION_TYPE="pixel"
export MINIMUM_RESOLUTION=$RESOLUTION

#export ASPECT_BUCKET_ROUNDING=2

export VALIDATION_PROMPT="ethnographic photography of teddy bear at a picnic"
export VALIDATION_GUIDANCE=7.5
export VALIDATION_GUIDANCE_RESCALE=0.0
# For flux training, you may want to do validation with classifier free guidance.
# You can set this to be >1.0 to do so. If you don't enable CFG, results may be unreliable or look very bad.
# Flux LoRAs have a side-effect of requiring a blank negative prompt, so be sure to set VALIDATION_NEGATIVE_PROMPT="" as well.
export VALIDATION_GUIDANCE_REAL=1.0
# Skip CFG during validation sampling with CFG (flux only, default=2).
export VALIDATION_NO_CFG_UNTIL_TIMESTEP=2
# How frequently we will save and run a pipeline for validations.
export VALIDATION_STEPS=30
export VALIDATION_NUM_INFERENCE_STEPS=20
export VALIDATION_NEGATIVE_PROMPT="blurry, cropped, ugly"
export VALIDATION_SEED=42
export VALIDATION_RESOLUTION=$RESOLUTION


export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=2
export VAE_BATCH_SIZE=4

# Use any standard scheduler type. constant, polynomial, constant_with_warmup
export LR_SCHEDULE="constant"
export LR_WARMUP_STEPS=10

# Caption dropout probability. Set to 0.1 for 10% of captions dropped out. Set to 0 to disable.
# You may wish to disable dropout if you want to limit your changes strictly to the prompts you show the model.
# You may wish to increase the rate of dropout if you want to more broadly adopt your changes across the model.
export CAPTION_DROPOUT_PROBABILITY=0.1

export METADATA_UPDATE_INTERVAL=65

# How many workers to use for VAE caching.
export MAX_WORKERS=32
# Read and write batch sizes for VAE caching.
export READ_BATCH_SIZE=25
export WRITE_BATCH_SIZE=64
# How many images to process at once (resize, crop, transform) during VAE caching.
export IMAGE_PROCESSING_BATCH_SIZE=32
# When using large batch sizes, you'll need to increase the pool connection limit.
export AWS_MAX_POOL_CONNECTIONS=128
# For very large systems, setting this can reduce CPU overhead of torch spawning an unnecessarily large number of threads.
export TORCH_NUM_THREADS=8

# If this is set, any images that fail to open will be DELETED to avoid re-checking them every time.
export DELETE_ERRORED_IMAGES=0
# If this is set, any images that are too small for the minimum resolution size will be DELETED.
export DELETE_SMALL_IMAGES=0

# Bytedance recommends these be set to "trailing" so that inference and training behave in a more congruent manner.
# To follow the original SDXL training strategy, use "leading" instead, though results are generally worse.
export TRAINING_SCHEDULER_TIMESTEP_SPACING="trailing"
export INFERENCE_SCHEDULER_TIMESTEP_SPACING="trailing"

# Removing this option or unsetting it uses vanilla training. Setting it reweights the loss by the position of the timestep in the noise schedule.
# A value "5" is recommended by the researchers. A value of "20" is the least impact, and "1" is the most impact.
export MIN_SNR_GAMMA=5

# Set this to an explicit value of "false" to disable Xformers. Probably required for AMD users.
export USE_XFORMERS=false

# There's basically no reason to unset this. However, to disable it, use an explicit value of "false".
# This will save a lot of memory consumption when enabled.
export USE_GRADIENT_CHECKPOINTING=true

##
# Options below here may require a bit more complicated configuration, so they are not simple variables.
##

# TF32 is great on Ampere or Ada, not sure about earlier generations.
export ALLOW_TF32=true

export OPTIMIZER="optimi-adamw"


# EMA is a strong regularisation method that uses a lot of extra VRAM to hold two copies of the weights.
# This is worthwhile on large training runs, but not so much for smaller training runs.
# NOTE: EMA is not currently applied to LoRA.
export USE_EMA=false
export EMA_DECAY=0.999

export TRAINER_EXTRA_ARGS=""
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --i_know_what_i_am_doing"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_rank=4"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --max_grad_norm=1.0 --gradient_precision=fp32"
# x-flux only trains the mmdit blocks but you can change lora_target to all or context to experiment.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16 --lora_init_type=default --flux_lora_target=mmdit"

# Reproducible training. Set to -1 to disable.
export TRAINING_SEED=42

# Mixed precision is the best. You honestly might need to YOLO it in fp16 mode for Google Colab type setups.
export MIXED_PRECISION="bf16"                # Might not be supported on all GPUs. fp32 will be needed for others.
export PURE_BF16=true

# This has to be changed if you're training with multiple GPUs.
export TRAINING_NUM_PROCESSES=1
export TRAINING_NUM_MACHINES=1
export ACCELERATE_EXTRA_ARGS=""                          # --multi_gpu or other similar flags for huggingface accelerate

# Some models just simply won't work with torch.compile or will not experience any speedup. YMMV.
export TRAINING_DYNAMO_BACKEND='no'                # or 'no' if you want to disable torch compile in case of performance issues or lack of support (eg. AMD)

# Workaround a bug in the tokenizers platform.
export TOKENIZERS_PARALLELISM=false
EOF

cat > config/multidatabackend.json <<EOF
[
  {
    "id": "blackswan-flux",
    "type": "local",
    "crop": false,
    "crop_aspect": "preserve",
    "crop_style": "random",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux/blackswan",
    "instance_data_dir": "/workspace/SimpleTuner/output/blackswan/dataset/",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "textfile",
    "metadata_backend": "json"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux/blackswan",
    "disabled": false,
    "write_batch_size": 128
  }
]
EOF

cd /workspace



apt -y install git-lfs

##Download orig_backup
apt -y install rclone

##Fresh SD Install
mkdir /workspace/stable-diffusion-webui
cd /workspace/stable-diffusion-webui

if [ -z "$A1111" ]; then
	echo "Installing VLAD"
	git clone https://github.com/vladmandic/automatic.git ./
else
	echo "Installing A1111"
	git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git ./
 	#git checkout tags/v1.6.0

  	cd extensions
	git clone https://github.com/Mikubill/sd-webui-controlnet.git
	#git clone https://github.com/ahgsql/StyleSelectorXL
	#git clone https://github.com/imrayya/stable-diffusion-webui-Prompt_Generator.git
	#git clone https://github.com/IDEA-Research/DWPose
	git clone https://github.com/Uminosachi/sd-webui-inpaint-anything.git
	#git clone https://github.com/d8ahazard/sd_dreambooth_extension.git ./extensions/sd_dreambooth_extension
fi

if [ -z "$A1111" ]; then
	controlnet_path='/workspace/stable-diffusion-webui/extensions-builtin/sd-webui-controlnet/models/'
else
	controlnet_path='/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/models/'
fi

(
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_xl.pth" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_xl_openpose.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_256lora.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_sketch_256lora.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_recolor_256lora.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_depth_256lora.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors" -P $controlnet_path ;
wget "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin" -P $controlnet_path ;

sleep 10
touch /workspace/download/control_finish) &> control_download.log &

(
cd /workspace/stable-diffusion-webui/models;
mkdir Stable-diffusion;
mkdir Lora;
mkdir embeddings;
mkdir VAE;
wget "https://civitai.com/api/download/models/198246?type=Model&format=SafeTensor&size=pruned&fp=fp16" -O "Stable-diffusion/TimeLessXL.safetensors";
wget "https://civitai.com/api/download/models/169740?type=Model&format=SafeTensor&size=full&fp=fp16" -O "Stable-diffusion/ZavyXL.safetensors";
wget "https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors?download=true" -O "Stable-diffusion/SSD-1B.safetensors";

wget "https://civitai.com/api/download/models/131960?type=VAE&format=SafeTensor" -O "VAE/TalmendoXL.safetensors";

wget "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true" -O "Lora/LCM-SDXL.safetensors";
wget "https://huggingface.co/latent-consistency/lcm-lora-ssd-1b/resolve/main/pytorch_lora_weights.safetensors?download=true" -O "Lora/LCM-SSD-1B.safetensors";

wget "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor" -O "Lora/Detail_Tweaker_XL.safetensors" ;
wget "https://civitai.com/api/download/models/169002?type=Model&format=SafeTensor" -O "Lora/Artfull_Fractal.safetensors" ;
wget "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor" -O "Lora/Artfull_Base.safetensors" ;
wget "https://civitai.com/api/download/models/169041?type=Model&format=SafeTensor" -O "Lora/Schematics.safetensors" ;


sleep 10
touch /workspace/download/finish) &> download.log &

#Define Copy Job
(cd /workspace
mkdir stable-diffusion-webui/models/Stable-diffusion/
mkdir stable-diffusion-webui/models/embeddings/
mkdir stable-diffusion-webui/models/VAE/
mkdir stable-diffusion-webui/models/Lora/

if [ -z "$A1111" ]; then
	controlnet_path='stable-diffusion-webui/extensions-builtin/sd-webui-controlnet/models/'
else
	controlnet_path='stable-diffusion-webui/extensions/sd-webui-controlnet/models/'
fi

while [ ! -f /workspace/download/finish ]; do
	mv download/Stable-diffusion/* stable-diffusion-webui/models/Stable-diffusion/ &
	mv download/embeddings/* stable-diffusion-webui/models/embeddings/ &
	mv download/Lora/* stable-diffusion-webui/models/Lora/ &
	mv download/VAE/* stable-diffusion-webui/models/VAE/ &
done

while [ ! -f /workspace/download/control_finish ]; do
	if [ -d $controlnet_path ]; then
		mv download/controlnet_models/* $controlnet_path &
	fi

	sleep 4
done
) &> copy.log &


#Write necessary files
cd /workspace/stable-diffusion-webui
cat  <<EOT > relauncher.py
#!/usr/bin/python3
import os, time

n = 0
while True:
    print('Relauncher: Launching...')
    if n > 0:
        print(f'\tRelaunch count: {n}')
    launch_string = "/workspace/stable-diffusion-webui/webui.sh -f --listen"
    os.system(launch_string)
    print('Relauncher: Process is ending. Relaunching in 2s...')
    n += 1
    time.sleep(2)
EOT
chmod +x relauncher.py

#source /workspace/venv/bin/activate
if [ -z "$A1111" ]; then
	python3 -u /workspace/stable-diffusion-webui/relauncher.py | while IFS= read -r line
	do
		echo "--$line"
		if [[ "$line" == *"Available VAEs"* ]]; then
			pkill relauncher.py
			echo "Killed Relauncher as it was stuck at no models"
			
			while [[ ! -e /workspace/download/download_finished ]];do
				sleep 1
			done
			
			/workspace/copy_downloaded_models.sh
			echo "Copied Models"
	
			echo "Setup finished, launching SD :)"  
			python3 /workspace/stable-diffusion-webui/relauncher.py
		fi
	done
else
	python3 -u /workspace/stable-diffusion-webui/relauncher.py
fi
