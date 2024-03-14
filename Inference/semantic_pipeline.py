from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer


from utils.image_util import resize_max_res, chw2hwc, colorize_semantic_maps
from utils.ensemble import ensemble_masks


class SemanticPipelineOutput(BaseOutput):
    """
    Output class for semantic pipeline.
    Args:
        semantic_np (`np.ndarray`):
            semantic array, with values in the range of [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    semantic_np: np.ndarray
    semantic_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class SemanticEstimationPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    semantic_latent_scale_factor = 0.18215

    def __init__(self,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 scheduler: DDIMScheduler,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    @torch.no_grad()
    def __call__(self,
                 input_image: Image,
                 prompt: str = None,
                 denoising_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 384,
                 match_input_res: bool = True,
                 batch_size: int = 0,
                 show_progress_bar: bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> SemanticPipelineOutput:

        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size

        # adjust the input resolution.
        if not match_input_res:
            assert (
                    processing_res is not None
            ), " Value Error: `resize_output_back` is only valid with "

        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        # Convert the image to RGB
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        # Normalize RGB Values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, c] -> [c, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # ----------------- predicting semantic -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        # find the batch size
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = 1
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # predict the semantic
        semantic_pred_ls = []
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader

        for batch in iterable_bar:
            (batched_image,) = batch
            semantic_pred_raw = self.single_infer(
                rgb_in=batched_image,
                prompt=prompt,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            semantic_pred_ls.append(semantic_pred_raw.detach().clone())

        semantic_preds = torch.concat(semantic_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            semantic_pred, semantic_uncert = ensemble_masks(semantic_preds, **(ensemble_kwargs or {}))
        else:
            semantic_pred = semantic_preds
            semantic_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(semantic_pred)
        max_d = torch.max(semantic_pred)
        semantic_pred = (semantic_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        semantic_pred = semantic_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(semantic_pred)
            pred_img = pred_img.resize(input_size)
            semantic_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        semantic_pred = semantic_pred.clip(0, 1)

        # Colorize
        semantic_colored = colorize_semantic_maps(semantic_pred, 0, 1, cmap="Spectral").squeeze()
        semantic_colored = (semantic_colored * 255).astype(np.uint8)
        semantic_colored_hwc = chw2hwc(semantic_colored)
        semantic_colored_img = Image.fromarray(semantic_colored_hwc)

        return SemanticPipelineOutput(
            semantic_np=semantic_pred,
            semantic_colored=semantic_colored_img,
            uncertainty=semantic_uncert)

    @torch.no_grad()
    def single_infer(self, rgb_in: torch.Tensor, prompt: str, num_inference_steps: int, show_pbar: bool):
        device = rgb_in.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # encode prompt
        text_embed = self.encode_text(prompt)

        # Initial semantic (noise)
        semantic_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype)

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, semantic_latent], dim=1)
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=text_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            semantic_latent = self.scheduler.step(noise_pred, t, semantic_latent).prev_sample

        torch.cuda.empty_cache()
        semantic = self.decode_semantic(semantic_latent)
        # clip prediction
        semantic = torch.clip(semantic, -1.0, 1.0)
        # shift to [0, 1]
        semantic = (semantic + 1.0) / 2.0

        return semantic

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor

        return rgb_latent

    def encode_text(self, prompt):
        """
        Encode text embedding for prompt
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        text_embed = self.text_encoder(text_input_ids)[0]
        text_embed = text_embed.mean(dim=0, keepdim=True)

        return text_embed

    def decode_semantic(self, semantic_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into semantic map.
        """
        semantic_latent = semantic_latent / self.semantic_latent_scale_factor
        z = self.vae.post_quant_conv(semantic_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        semantic_mean = stacked.mean(dim=1, keepdim=True)

        return semantic_mean
