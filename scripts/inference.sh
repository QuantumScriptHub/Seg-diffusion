inference_single_image(){
input_rgb_path='path for input rgb'
output_dir='../output'
class_name='a list with class name for open-vocabulary semantic segmentation'
stable_diffusion_repo_path='path for stable-diffusion-2'
pretrained_model_path='path for unet'
ensemble_size=10

cd ..
cd Inference

CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --input_rgb_path $input_rgb_path \
    --output_dir $output_dir \
    --class_name $class_name \
    --stable_diffusion_repo_path $stable_diffusion_repo_path \
    --pretrained_model_path $pretrained_model_path \
    --ensemble_size $ensemble_size \

}

inference_single_image
