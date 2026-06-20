import torch
import os

def get_latest_model_path(ckpnt_path):
    files = [f for f in os.listdir(ckpnt_path) if f.startswith('checkpoint_')]
    epoch = [int(f.split('_val')[0].replace('checkpoint_', '').replace('.pt', '')) for f in files]
    corr = {m:f for m,f in zip(epoch, files)}
    epoch.sort()
    return os.path.join(ckpnt_path, corr[epoch[-1]])
    
def load_model_checkpoint(load_path):
    if not load_path.endswith(".pt"): # support for models stored in ".p" format
        print("Loading from a '.p' checkpoint. Only evaluation is supported. Only model weights will be loaded.")
        checkpoint = torch.load(load_path, weights_only=False)
    else: # ".pth" format
        checkpoint = torch.load(load_path, weights_only=False)#, map_location=device)
    return checkpoint

def load_autoencoder(model, **kwargs):
    checkpoint = torch.load(kwargs['pretrained_autoencoder_path'], weights_only=False)
    model.load_state_dict(checkpoint['model'])
    # freeze weights
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
