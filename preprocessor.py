from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image).resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    
    np_image = np.array(img)/255
    np_image = np_image - np.array([0.485, 0.456, 0.406])
    np_image = np_image / np.array([0.229, 0.224, 0.225])
    return torch.tensor(np_image.transpose(2,0,1))