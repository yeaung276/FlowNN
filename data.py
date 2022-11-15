from torchvision import transforms, datasets
import torch

def get_datasets(data_dir: str):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    return train_loader, valid_loader, test_loader