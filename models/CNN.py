import torch
from torch import nn
from torch.nn import Conv2d
import math
import numpy as np
import matplotlib.pyplot as plt
import random

device="cuda:0" if torch.cuda.is_available() else "cpu"

#--------------------MODEL------------------------
def create_model_CNN():
    model=nn.Sequential(
        Conv2d(3,5,padding=1, kernel_size=3, stride=1),
        nn.ReLU(),
        Conv2d(5,64,padding=2, kernel_size=5, stride=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).to(device)
    return model


#----------------utils-----------------------------

def _padded_grid(n_items: int):
    if n_items <= 0:
        return 0, 0
    rows = math.isqrt(n_items)
    cols = math.ceil(n_items / rows)
    if rows > cols:
        rows, cols = cols, rows
    return rows, cols


def viz_kernels_per_in_channel(conv: nn.Conv2d, in_channels_to_show=None, title_prefix="Conv2d"):

    assert isinstance(conv, nn.Conv2d)
    W = conv.weight.detach().cpu().numpy()  
    out_ch, in_ch, kh, kw = W.shape

    if in_channels_to_show is None:
        in_channels_to_show = list(range(in_ch))
    else:
        in_channels_to_show = [c for c in in_channels_to_show if 0 <= c < in_ch]
        if not in_channels_to_show:
            return

    for ic in in_channels_to_show:
        kernels = W[:, ic, :, :] 
        rows, cols = _padded_grid(out_ch)

        # единая симметричная шкала по модулю
        vmax = float(np.abs(kernels).max()) if kernels.size else 1.0
        vmin = -vmax
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2), squeeze=False)
        fig.suptitle(f"{title_prefix} | per in_ch={ic} | kernel={kh}x{kw} | out={out_ch}")

        flat_axes = axes.ravel()
        for oc in range(out_ch):
            ax = flat_axes[oc]
            ax.imshow(kernels[oc], cmap="bwr", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"out {oc}", fontsize=8)
            ax.axis("off")

        for k in range(out_ch, rows*cols):
            flat_axes[k].axis("off")

        plt.tight_layout()
        plt.show()


def viz_kernels_aggregated_over_in(conv: nn.Conv2d, reduce="l2", title_prefix="Conv2d"):

    assert isinstance(conv, nn.Conv2d)
    W = conv.weight.detach().cpu().numpy() 
    out_ch, in_ch, kh, kw = W.shape

    if reduce == "l2":
        agg = np.sqrt((W ** 2).sum(axis=1))        
    elif reduce == "sum":
        agg = W.sum(axis=1)                          
    elif reduce == "abs":
        agg = np.abs(W).sum(axis=1)                  
    else:
        raise ValueError("reduce must be one of {'l2','sum','abs'}")

    rows, cols = _padded_grid(out_ch)
    # шкала по всей фигуре:
    vmax = float(np.abs(agg).max()) if agg.size else 1.0
    vmin = -vmax if reduce != "l2" and reduce != "abs" else 0.0
    cmap = "bwr" if vmin < 0 else "viridis"

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2), squeeze=False)
    fig.suptitle(f"{title_prefix} | aggregated over in ({reduce}) | kernel={kh}x{kw} | out={out_ch}")

    flat_axes = axes.ravel()
    for oc in range(out_ch):
        ax = flat_axes[oc]
        ax.imshow(agg[oc], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"out {oc}", fontsize=8)
        ax.axis("off")

    for k in range(out_ch, rows*cols):
        flat_axes[k].axis("off")

    plt.tight_layout()
    plt.show()


def viz_rgb_first_layer(conv: nn.Conv2d, title_prefix="Conv2d", clip_mode="minmax"):
 
    assert isinstance(conv, nn.Conv2d)
    W = conv.weight.detach().cpu().numpy()  
    out_ch, in_ch, kh, kw = W.shape
    if in_ch != 3:
        raise ValueError("viz_rgb_first_layer: ожидается in_channels == 3")

 
    filters = np.transpose(W, (0, 2, 3, 1))

    if clip_mode == "global":
        fmin, fmax = filters.min(), filters.max()
        def norm(x):
            return (x - fmin) / (fmax - fmin + 1e-12)
    elif clip_mode == "minmax":
        def norm(x):
            a, b = x.min(), x.max()
            return (x - a) / (b - a + 1e-12)
    else:
        raise ValueError("clip_mode must be {'minmax','global'}")

    rows, cols = _padded_grid(out_ch)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.4, rows*2.4), squeeze=False)
    fig.suptitle(f"{title_prefix} | RGB first layer | kernel={kh}x{kw} | out={out_ch}")

    flat_axes = axes.ravel()
    for oc in range(out_ch):
        ax = flat_axes[oc]
        rgb = norm(filters[oc])
        rgb = np.clip(rgb, 0.0, 1.0)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"out {oc}", fontsize=8)
        ax.axis("off")

    for k in range(out_ch, rows*cols):
        flat_axes[k].axis("off")

    plt.tight_layout()
    plt.show()

def postactivations_CNN(model: nn.Module, test_loader, device="cuda:0", max_layers=None):

    model.eval().to(device)
    torch.cuda.empty_cache()

    # возьмем одно изображение
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    idx = random.randint(0, images.size(0) - 1)
    input_img = images[idx:idx+1].to(device)

    post_activations = []
    conv_modules = []

    def save_activation(module, inp, out):
        post_activations.append(out.detach().cpu())

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_modules.append(m)
            hooks.append(m.register_forward_hook(save_activation))
            if max_layers is not None and len(conv_modules) >= max_layers:
                break

    with torch.no_grad():
        _ = model(input_img)

    for h in hooks:
        h.remove()

    img_np = input_img.detach().cpu()[0]
    if img_np.shape[0] == 3:
        show_img = img_np.permute(1, 2, 0)
    else:
        show_img = img_np[0]

    plt.figure()
    if show_img.ndim == 3:
        plt.imshow((show_img - show_img.min()) / (show_img.max() - show_img.min() + 1e-12))
    else:
        plt.imshow(show_img, cmap="gray")
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
        if act.dim() == 4:
            act = act[0]  
        elif act.dim() == 3:
            pass
        else:
            continue

        num_channels = act.shape[0]
        rows, cols = padded_grid(num_channels)

        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
        fig.suptitle(f"Постактивации Conv2d слой {layer_idx}")
        axes = np.atleast_2d(axes)
        flat_axes = axes.ravel()

        for ch in range(num_channels):
            ax = flat_axes[ch]
            m = act[ch].numpy()
            if np.allclose(m.std(), 0):
                ax.imshow(m, cmap="viridis")
            else:
                m_norm = (m - m.min()) / (m.max() - m.min() + 1e-12)
                ax.imshow(m_norm, cmap="viridis")
            ax.axis("off")

        for k in range(num_channels, rows*cols):
            flat_axes[k].axis("off")

        plt.tight_layout()
        plt.show()