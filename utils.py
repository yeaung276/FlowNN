from typing import List
import torch
import json

def predict(tensor, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        log_y = model.forward(tensor)
        y_cap = torch.exp(log_y)        
        return y_cap.topk(topk, dim=1) 
    
def get_accuracy(prediction, labels):
    true_counts = prediction == labels.view(*prediction.shape)
    return torch.mean(true_counts.type(torch.FloatTensor)).item()

def checkpoint_model(model, arch: str, dir: str = '.'):
    checkpoint = {
        'input_size': (224,224,3),
        'output': 102,
        'pretrained_modal': arch,
        'classifier': [each for each in model.classifier ],
        'state': model.state_dict()
    }
    torch.save(checkpoint, '{dir}/checkpoint.pth')
    
def get_flower_name(dict_path: str, class_indexs: List[int]):
    with open(dict_path, 'r') as f:
        cat_to_name = json.load(f)
    name = []
    for class_idx in class_indexs.tolist()[0]:
        name.append(cat_to_name.get(str(class_idx + 1), 'unrecognized'))
    return name
    
