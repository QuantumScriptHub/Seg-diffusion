LAUNCH_TRAINING(){

cd ..
cd training
pretrained_model_name_or_path='path for stable-diffusion-2'
train_img_list='img_list.txt'
train_gt_list='gt_list.txt'
train_text_list='train_text.json'
val_img='path for validation of image'
val_gt='path for validation of ground truth'
val_text_list='val_text.json'
output_dir='../outputs'
train_batch_size=1
num_train_epochs=500
gradient_accumulation_steps=4
learning_rate=3e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='open-vocabulary semantic segmentation'
multires_noise_iterations=6
main_process_port=26777

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="no"  --multi_gpu --main_process_port $main_process_port run_train.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --train_img_list $train_img_list \
                  --train_gt_list $train_gt_list \
                  --train_text_list $train_text_list \
                  --val_img $val_img --val_gt $val_gt \
                  --val_text_list $val_text_list \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention \
                  --multires_noise_iterations $multires_noise_iterations \


}

LAUNCH_TRAINING
