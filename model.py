import torch
from torch import nn
from torchvision import models

model_map = {
    'densenet121': lambda: models.densenet121(pretrained=True)
}

def load_model_from_path(state_dict_path: str, device: str):
    model_dict = torch.load(state_dict_path)

    classifier = load_classifier(model_dict['classifier'])
    pretrained_model = get_pretrained_model(model_dict['pretrained_modal'])

    pretrained_model.classifier = classifier

    pretrained_model.load_state_dict(model_dict['state'])
    pretrained_model.eval()
    return pretrained_model.to(device)

def get_pretrained_model(model_name: str):
    model = model_map[model_name]()
    for layer in model.parameters():
        layer.requires_grad = False   
    return model 

def load_classifier(classifier_dict):
    return nn.Sequential(*classifier_dict)

def get_untrained_model(model_name: str, hidden_layer: int, device: str):
    base_model = get_pretrained_model(model_name)
    classifier = nn.Sequential(
        nn.Linear(1024, hidden_layer),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_layer, 102),
        nn.LogSoftmax(dim=1)
    )
    base_model.classifier = classifier
    return base_model.to(device), classifier.to(device)
    
