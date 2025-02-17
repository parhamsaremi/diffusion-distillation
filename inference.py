"""
A script for evaluating a diffusion model on the Smithsonian Butterflies dataset by computing FID. 
We assume the model has been trained (weights available in `checkpoint_path`) and that the code for
'simpleDiffusion' includes the `inference` method to compute FID and save samples.

This script uses the UNet2D model from the diffusion library and the simpleDiffusion model from the simple_diffusion library.
"""

import os
import torch
from nets.unet import UNetCondition2D
from nets.uvit import UViT
from utils.wavelet import wavelet_dec_2
from utils.plotter import plot_rgb
from diffusion.simple_diffusion import simpleDiffusion

from datasets import load_dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance


class InferenceConfig:
    # Checkpoint path to load from
    checkpoint_path = "/home/mila/p/parham.saremi/simpleDiffusion/ddpm-butterflies-128/distill_checkpoints/distilled_stage_5"

    image_size = 128  
    backbone = "unet"
    pred_param = "v"
    schedule = "shifted_cosine"
    noise_d = 64
    sampling_steps = 8
    ema_beta = 0.9999
    ema_warmup = 500
    ema_update_freq = 10
    batch_size = 16

    # Same environment / run settings
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    experiment_path = "/home/mila/p/parham.saremi/simpleDiffusion/ddpm-butterflies-128"

    # How many images to generate for FID evaluation
    num_images_to_eval = 900
    num_images_to_save = 16
    seed = 0

def main():
    config = InferenceConfig

    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    def transform_fn(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        # Wavelet decomposition
        # images = [wavelet_dec_2(image) / 2 for image in images]
        return {"images": images}

    dataset.set_transform(transform_fn)

    # Using a small batch size so we can gather images gradually
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,  
        shuffle=True,
    )

    if config.backbone == "unet":
        backbone = UNetCondition2D(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=(1,2,2,8,2),  
            block_out_channels=(128,128,256,512,768),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
        )
    elif config.backbone == "uvit":
        raise ValueError("UViT not supported in this example.")
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")

    diffusion_model = simpleDiffusion(
        backbone=backbone,
        config=config,
    )

    def fid_calculator(real_images, generated_images):
        """
        Compute FID between real_images and generated_images,
        both expected as [N, C, H, W] Tensors in the same domain & range.
        """
        fid_metric = FrechetInceptionDistance().to("cuda")

        # Real
        fid_metric.update(real_images.to("cuda"), real=True)
        # Fake
        fid_metric.update(generated_images.to("cuda"), real=False)

        fid_score = fid_metric.compute().item()
        return fid_score

    fid_value = diffusion_model.inference(
        val_dataloader=val_loader,
        checkpoint_path=config.checkpoint_path,
        fid_calculator=fid_calculator,
        num_images_to_eval=config.num_images_to_eval,
        num_images_to_save=config.num_images_to_save,
        plot_function=plot_rgb,
        use_ema=False
    )

    print(f"Final FID on {config.num_images_to_eval} images: {fid_value:.4f}")

if __name__ == "__main__":
    main()