PYTHONPATH=$(pwd) \
IMAGE_TRAIN_SIZE=252 \
USE_IMG_EMBED_AS_INPUT=false \
TOKEN_LOSS_WIGHT=0.1 \
MAX_PIXELS=262640 \
nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model 'DiffSynth-Studio/Nexus-Gen' \
    --model_type 'qwenall2all_custom' \
    --template 'qwenall2all_custom' \
    --custom_register_path train/ar/model.py \
                           train/ar/template.py \
    --dataset 'path_to_you_dataset.jsonl' \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --freeze_vit true \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps 3000 \
    --save_steps 3000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --max_length 4096 \
    --deepspeed zero3 \
