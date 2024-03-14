import os
import sys
import cv2
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

sys.path.append("../")
from Inference.semantic_pipeline import SemanticEstimationPipeline
from Inference.predictor import VisualizationDemo
from utils.seed_all import seed_all


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run semantic Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--stable_diffusion_repo_path",
        type=str,
        default='stable-diffusion-2',
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='None',
        help="path for unet",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="a list with class name for open-vocabulary semantic segmentation"
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=384,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 384.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output mask at resized operating resolution. Default: False.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )

    args = parser.parse_args()
    checkpoint_path = args.pretrained_model_path
    input_image_path = args.input_rgb_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    seed = args.seed
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = 1
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}".format(input_image_path))

    # -------------------- Model --------------------
    vae = AutoencoderKL.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'vae'))
    scheduler = DDIMScheduler.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'scheduler'))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'text_encoder'))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'tokenizer'))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, 'unet'),
                                                in_channels=8, sample_size=48,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)

    pipe = SemanticEstimationPipeline(unet=unet,
                                      vae=vae,
                                      scheduler=scheduler,
                                      text_encoder=text_encoder,
                                      tokenizer=tokenizer)

    logging.info("loading pipeline whole successfully.")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        cfg = setup_cfg()
        demo = VisualizationDemo(cfg)
        for img_name in os.listdir(input_image_path):
            input_image_pil = Image.open(os.path.join(input_image_path, img_name))
            input_prompt = [f'a photo of a {x}' for x in args.class_name]
            pipe_out = pipe(input_image_pil,
                            input_prompt,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=batch_size,
                            show_progress_bar=True,
                            )
            semantic_np = pipe_out.semantic_np
            pred_save_path = os.path.join(output_dir, img_name.split('.jpg')[0] + '.png')
            semantic_np = (semantic_np * 255).astype(np.uint8)

            predictions = {}
            predictions['sem_seg'] = semantic_np
            _, visualized_output = demo.run_on_image(cv2.imread(os.path.join(input_image_path, img_name)), predictions, args.class_name)
            visualized_output.save(pred_save_path)


