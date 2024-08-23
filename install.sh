#!/bin/bash

#####	Start Docker Cmd: /bin/bash -c 'if [ ! -f /setup.sh ]; then wget "https://raw.githubusercontent.com/terminator4088/runpod/main/install.sh" -O /setup.sh && chmod +x /setup.sh && /setup.sh; fi'

cd /workspace

apt update
apt -y install vim
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx

# Install SimpleTuner
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# Regularization Images
apt -y install git-lfs

(mkdir -p /workspace/SimpleTuner/datasets
cd /workspace/SimpleTuner/datasets
mkdir $SUBJECT
git clone https://huggingface.co/datasets/ptx0/pseudo-camera-10k  pseudo-camera-10k) &>/workspace/download.log &

python3 -m venv .venv
source .venv/bin/activate
pip install -U poetry pip

# Install Dependencie
pip install huggingface_hub
pip install wandb

# Log Into Huggingface
git config --global credential.helper store
huggingface-cli login --token $HFTK --add-to-git-credential

# Log Into Wandb
WANDB_API_KEY="$WANDB_API_KEY" wandb login

# Install Dependencie
poetry install --no-root

# May Be needed
pip uninstall -y deepspeed bitsandbytes

# Download working Diffusers
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers

# Less VRAM Usage
pip install optimum-quanto

#
cat >config/config.env <<EOF
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
export HUB_MODEL_NAME=$SUBJECT

# Make DEBUG_EXTRA_ARGS empty to disable wandb.
export DEBUG_EXTRA_ARGS="--report_to=wandb"
export TRACKER_PROJECT_NAME="$SUBJECT-training"
export TRACKER_RUN_NAME="simpletuner-fluxx"

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
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16 --lora_init_type=default --flux_lora_target=mmdit"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --user_prompt_library=config/user_prompt_library.json"

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
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/workspace/SimpleTuner/cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "/workspace/SimpleTuner/datasets/pseudo-camera-10k",
    "ignore_epochs": true,
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "json"
  },
  {
    "id": "$SUBJECT-flux",
    "type": "local",
    "crop": false,
    "crop_aspect": "preserve",
    "crop_style": "random",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "/workspace/SimpleTuner/cache/vae/flux/$SUBJECT",
    "instance_data_dir": "/workspace/SimpleTuner/output/$SUBJECT/dataset/",
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
    "cache_dir": "/workspace/SimpleTuner/cache/text/flux/blackswan",
    "disabled": false,
    "write_batch_size": 128
  }
]
EOF

cat > config/user_prompt_library.json <<EOF
{
    "anime_$SUBJECT": "a breathtaking anime-style portrait of $SUBJECT, capturing her essence with vibrant colors and expressive features",
    "chef_$SUBJECT": "a high-quality, detailed photograph of $SUBJECT as a sous-chef, immersed in the art of culinary creation",
    "just_$SUBJECT": "a lifelike and intimate portrait of $SUBJECT, showcasing her unique personality and charm",
    "cinematic_$SUBJECT": "a cinematic, visually stunning photo of $SUBJECT, emphasizing her dramatic and captivating presence",
    "elegant_$SUBJECT": "an elegant and timeless portrait of $SUBJECT, exuding grace and sophistication",
    "adventurous_$SUBJECT": "a dynamic and adventurous photo of $SUBJECT, captured in an exciting, action-filled moment",
    "mysterious_$SUBJECT": "a mysterious and enigmatic portrait of $SUBJECT, shrouded in shadows and intrigue",
    "vintage_$SUBJECT": "a vintage-style portrait of $SUBJECT, evoking the charm and nostalgia of a bygone era",
    "artistic_$SUBJECT": "an artistic and abstract representation of $SUBJECT, blending creativity with visual storytelling",
    "futuristic_$SUBJECT": "a futuristic and cutting-edge portrayal of $SUBJECT, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
EOF

mkdir -p cache/vae/$SUBJECT
mkdir -p cache/text/$SUBJECT

cd /workspace/SimpleTuner

cat <<EOF
---
DONE
---
EOF
