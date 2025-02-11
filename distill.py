"""
A script for progressive distillation of a previously trained diffusion model
on the Smithsonian Butterflies dataset. The student model learns to match the
teacher’s multiple-step DDIM sampling with fewer steps, following Algorithm 2
from the progressive distillation papers.
"""

import os
import torch
from torchvision import transforms
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset

from nets.unet import UNetCondition2D
from nets.uvit import UViT
from utils.wavelet import wavelet_dec_2
from utils.plotter import plot_rgb
from diffusion.simple_diffusion import simpleDiffusion

class DistillationConfig:
    # Optimization parameters
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    train_batch_size = 4
    gradient_accumulation_steps = 1
    ema_beta = 0.9999
    ema_warmup = 500
    ema_update_freq = 10

    # Distillation parameters
    distill_stages = 5      # K: how many times we halve the teacher’s sampling steps
    initial_N = 128           # initial teacher sampling steps
    distill_epochs = 100      # how many epochs to distill per stage

    # Experiment parameters
    resume = False  # not strictly used here; can adapt if you want to load teacher differently
    num_epochs = 400 # original training epochs (not used in distill loop, but kept for reference)
    save_image_epochs = 50
    evaluation_batches = 1
    mixed_precision = "fp16"
    experiment_path = "/home/mila/p/parham.saremi/simpleDiffusion/ddpm-butterflies-wavelet"

    # Model/backbone parameters
    image_size = 128
    backbone = "unet"

    # Diffusion parameters
    pred_param = "v" 
    schedule = "shifted_cosine"
    noise_d = 64
    sampling_steps = 128  # not used in teacher sampling, since teacher steps are progressive
    seed = 0
    
def main():
    config = DistillationConfig

    # Load and transform dataset, same as in train.py
    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def transform(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        # wavelet decomposition
        images = [wavelet_dec_2(image) / 2 for image in images]
        return {"images": images}

    dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    # Set up the same backbone architecture
    backbone = UNetCondition2D(
        sample_size=config.image_size,  # the target image resolution
        in_channels=12,  # the number of input channels, 3 for RGB images
        out_channels=12,  # the number of output channels
        layers_per_block=(1,2,2,8,2),  # how many ResNet layers to use per UNet block
        block_out_channels=(128,128,256,512,768),  # the number of output channels for each UNet block
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


    student_diffusion_model = simpleDiffusion(
        backbone=backbone,
        config=config,
    )

    # optimizer = torch.optim.Adam(backbone.parameters(), lr=config.learning_rate)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=len(train_loader) * config.distill_epochs,
    # )

    student_diffusion_model.distill(
        # optimizer=optimizer,
        train_dataloader=train_loader,
        # lr_scheduler=lr_scheduler,
        K=config.distill_stages,
        initial_N=config.initial_N,
        distill_epochs=config.distill_epochs,
    )


if __name__ == "__main__":
    main()
