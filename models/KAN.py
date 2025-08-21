import torch
from torch import nn
from convkan import ConvKAN, LayerNorm2D
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

device="cuda:0" if torch.cuda.is_available() else "cpu"

#--------------------MODEL------------------------
def create_model_KAN():
    model=nn.Sequential(
        ConvKAN(3,5,padding=1, kernel_size=3, stride=1,spline_order = 3, grid_size=5, grid_range=(-10,10)),
        ConvKAN(5,64,padding=2, kernel_size=5, stride=1,spline_order = 1, grid_size=5, grid_range=(-10,10)),
        LayerNorm2D(64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).to(device)
    return model
#-------------------utils----------------------------


def postactivations_KAN(model, test_loader):
    torch.cuda.empty_cache()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    idx = random.randint(0, images.size(0) - 1)
    input_img = images[idx:idx+1].to('cuda:0')

    post_activations = []

    def save_activation(module, input, output):
        post_activations.append(output.detach().cpu())

    hooks = []
    for layer in model:
        if isinstance(layer, ConvKAN):
            hooks.append(layer.register_forward_hook(save_activation))

    with torch.no_grad():
        _ = model(input_img)

    for h in hooks:
        h.remove()

    plt.figure()
    plt.imshow(input_img.cpu()[0].permute(1, 2, 0))
    plt.title("Входное изображение")
    plt.axis("off")
    plt.show()

    def padded_grid(n_items: int):
        if n_items <= 0:
            return 0, 0
        rows = math.isqrt(n_items)
        cols = math.ceil(n_items / rows)
        if rows > cols:  
            rows, cols = cols, rows
        return rows, cols

    for layer_idx, act in enumerate(post_activations, 1):
        act = act.squeeze(0)  
        if act.dim() == 4 and act.size(0) == 1:
            act = act.squeeze(0)           
        elif act.dim() == 4:
            act = act[0]

        num_channels = act.shape[0]
        rows, cols = padded_grid(num_channels)

        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        fig.suptitle(f"Постактивации ConvKAN слой {layer_idx}")
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        flat_axes = axes.ravel()
        for ch in range(num_channels):
            ax = flat_axes[ch]
            ax.imshow(act[ch], cmap="viridis")
            ax.axis("off")

        for k in range(num_channels, rows*cols):
            flat_axes[k].axis("off")

        plt.tight_layout()
        plt.show()


def plot_convkan_splines(layer, kernel_size, in_channels, out_channels, title_prefix="Layer"):

    grid = layer.kan_layer.grid.detach().cpu().numpy()
    if grid.ndim > 1:
        grid_1d = grid[0][0]
    else:
        grid_1d = grid
    k = int(layer.kan_layer.spline_order)

    weights = layer.kan_layer.scaled_spline_weight.detach().cpu().numpy()
    n_coeffs = weights.shape[-1]
    try:
        W = weights.reshape(out_channels, in_channels, kernel_size, kernel_size, n_coeffs)
    except Exception as e:
        raise RuntimeError(
            f"Нельзя сделать reshape в (out={out_channels}, in={in_channels}, kh={kernel_size}, kw={kernel_size}, coeffs={n_coeffs}). "
            f"Исходная форма {weights.shape}. Проверь порядок индексов/параметры."
        ) from e

    x = np.linspace(grid_1d[0], grid_1d[-1], 200)

    for ic in range(in_channels):
        fig, axes = plt.subplots(
            kernel_size, kernel_size,
            figsize=(kernel_size*2.6, kernel_size*2.1),
            squeeze=False
        )
        fig.suptitle(f"{title_prefix} | in_ch={ic}")

        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ax = axes[kh, kw]

                for oc in range(out_channels):
                    coeffs = W[oc, ic, kh, kw, :]
                    spline = BSpline(grid_1d, coeffs, k=k)
                    y = spline(x)
                    ax.plot(x, y, linewidth=0.9)

                ax.set_title(f"({kh},{kw})", fontsize=8)
                ax.grid(True, alpha=0.3)


        plt.tight_layout()
        plt.show()