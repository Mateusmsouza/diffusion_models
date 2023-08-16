#@title training
import os
import torch
import torch.nn as nn
import argparse

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
#from modules import UNet

def train(args):
    run_name = args["run_name"]
    device = args["device"]
    lr = args["lr"]
    image_size = args["image_size"]
    epochs = args["epochs"]
    run_name = args["run_name"]
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=image_size, device=device)
    l = len(dataloader)

    for epoch in range(epochs):
        print(f"Training epoch {0}")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise_to_image(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
        #torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def launch():
    run_name = "DDPM_Uncondtional"
    epochs = 500
    batch_size = 12
    image_size = 64
    dataset_path = r"/content/cats-faces-64x64-for-generative-models"
    device = "cuda"
    lr = 3e-4
    os.makedirs(f"results/{run_name}", exist_ok=True)
    train({
      "run_name": run_name,
      "epochs": epochs,
      "batch_size": batch_size,
      "image_size": image_size,
      "dataset_path": dataset_path,
      "device": device,
      "lr": lr
    })

if __name__ == '__main__':
    launch()