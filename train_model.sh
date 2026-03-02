# Define variables for columns
export MODEL_NAME="black-forest-labs/FLUX.2-klein-4B"
export LR=1e-4
export RANK=128
export SIZE=512
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="./runs/${TIMESTAMP}_r${RANK}_lr${LR}_s${SIZE}"
export WANDB_NAME=$(basename $OUTPUT_DIR)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --rank=$RANK \
  --learning_rate=$LR \
  --height=$SIZE \
  --width=$SIZE \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30050 \
  --num_train_epochs=10 \
  --seed="42" \
  --checkpointing_steps=2000  \
  --validation_check \
  --validation_steps=400 \
  --report_to="wandb" \
  # --resume_from_checkpoint="latest"  \ # remember change $OUTPUT_DIR when resuming
