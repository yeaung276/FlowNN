from argparse import ArgumentParser
import numpy as np
import torch
from preprocessor import process_image
from model import load_model_from_path
from utils import predict, get_flower_name

parser = ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', required=False, default=1, type=int)
parser.add_argument('--category_names', required=False, default='cat_to_name.json', type=str)
parser.add_argument('--gpu', required=False, default=True, action="store_true")

def main():
    args = parser.parse_args()
    device = 'cuda' if args.gpu else 'cpu'

    image = process_image(args.image_path)
    image = torch.unsqueeze(image, 0).type(torch.FloatTensor).to(device)
    
    model = load_model_from_path(args.checkpoint, device)
    
    top_p, top_c = predict(image, model, args.top_k)
    
    name = get_flower_name(args.category_names, top_c)
    
    print(f'name: {name}')
    print(f'prob: {[itm for itm in top_p.tolist()[0]]}')

    return name, top_p
    
    
    
    
if __name__=='__main__':
    main()