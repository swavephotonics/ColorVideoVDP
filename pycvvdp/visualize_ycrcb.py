import numpy as np
import matplotlib.pyplot as plt
import torch

def show_dkl_channels(abc, title='cvvdp opponent channels'):
    """
    Visualise ΔDKL channels.

    Parameters
    ----------
    abc   : (H, W, 3) or (3, H, W) torch.Tensor | np.ndarray
            Channel order = [Achromatic, RG, YV], linear (not gamma-corrected).
    title : str, optional
            Figure window title.

    Notes
    -----
    • Achromatic channel is displayed as grayscale.  
    • RG channel is rendered with red for +RG, green for –RG.  
    • YV channel is rendered with yellow for +YV, violet for –YV.  
    """
    # --- convert to numpy & canonical shape H×W×3 -------------------------
    if torch.is_tensor(abc):
        abc = abc.detach().cpu().numpy()

    if abc.shape[0] == 3 and abc.ndim == 3:          # 3×H×W → H×W×3
        abc = np.transpose(abc, (1, 2, 0))

    assert abc.ndim == 3 and abc.shape[2] == 3, \
        'Expect input with 3 opponent channels.'

    # split channels -------------------------------------------------------
    ach = abc[:, :, 0]
    rg  = abc[:, :, 1]
    yv  = abc[:, :, 2]

    # --- build pseudo-RGB images -----------------------------------------
    rg_img = np.zeros((*rg.shape, 3), dtype=np.float32)
    rg_img[..., 0] = np.clip(rg,  0, None)   # +RG → red
    rg_img[..., 1] = np.clip(-rg, 0, None)   # –RG → green

    yv_img = np.zeros((*yv.shape, 3), dtype=np.float32)
    # +YV → yellow  (R+G) ;  –YV → violet (B)
    yv_img[..., 0] = np.clip(yv,  0, None)               # red
    yv_img[..., 1] = np.clip(yv,  0, None)               # green
    yv_img[..., 2] = np.clip(-yv, 0, None)               # blue

    # --- normalise for display -------------------------------------------
    def to_8bit(img):
        img_norm = img - img.min()
        img_norm /= (img_norm.max() + 1e-8)
        return (img_norm * 255).astype(np.uint8)

    ach_vis = to_8bit(ach)
    rg_vis  = to_8bit(rg_img)
    yv_vis  = to_8bit(yv_img)

    # --- plot -------------------------------------------------------------
    plt.figure(figsize=(10, 4), num=title)
    # Achromatic
    plt.subplot(1, 3, 1)
    plt.imshow(ach_vis, cmap='gray')
    plt.title('Ach (luminance)')
    plt.axis('off')
    # RG opponent
    plt.subplot(1, 3, 2)
    plt.imshow(rg_vis)
    plt.title('RG (red–green)')
    plt.axis('off')
    # YV opponent
    plt.subplot(1, 3, 3)
    plt.imshow(yv_vis)
    plt.title('YV (yellow–violet)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()