import imageio
import torch
from PIL import Image           
import numpy as np
from .numpy import tensor_img_to_numpy

def save_img(img, path):
    if torch.is_tensor(img):
        img = convert_img_tensor_to_np_img(img)
    if not (path.endswith('.jpg') or path.endswith('.png') or path.endswith('.jpeg')):
        path += '.jpeg'
    imageio.imwrite(path, img)
    
def save_gif(frames, fps=30, name=None):
    if torch.is_tensor(frames):
        frames = tensor_img_to_numpy(frames)
    if name is None:
        name = './images/movie.gif'
    if not name.endswith(".gif"):
        name += ".gif"
    imageio.mimsave(name, frames, fps=fps)
    return name
    
    
def convert_img_tensor_to_np_img(tensor):
    assert len(tensor.shape) == 3
    tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    return img


def load_image(img_path):
    img = Image.open(img_path)
    img = np.array(img)  # in range(0, 255),
    return img  # shape (H,W, 3)