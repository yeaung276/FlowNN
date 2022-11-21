import torch
from torch import nn, optim
from argparse import ArgumentParser
from model import get_untrained_model
from data import get_datasets
from utils import get_accuracy, checkpoint_model

parser = ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', required=False, default='.')
parser.add_argument('--arch', required=False, default='densenet121')
parser.add_argument('--lr', required=False, default=0.01, type=float)
parser.add_argument('--epochs', required=False, default=10, type=int)
parser.add_argument('--hidden_units', required=False, default=512, type=int)
parser.add_argument('--gpu', required=False, default=False, action="store_true")

def main():
    args = parser.parse_args()
    device = 'cuda' if args.gpu else 'cpu'
    
    train_data, valid_data, _ = get_datasets(args.data_dir)
    
    model, classifier = get_untrained_model(args.arch, args.hidden_units, device)
    
    cretirion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    
    # training
    for epoch_idx in range(args.epochs):
        train_loss = 0
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_y = model.forward(images)
            loss = cretirion(log_y, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('.')
        else:
            model.eval()
            accuracy = 0
            vloss = 0
            with torch.no_grad():
                for images, labels in valid_data:
                    images, labels = images.to(device), labels.to(device)
                    logy = model.forward(images)
                    vloss += cretirion(logy, labels)
                    
                    y_cap = torch.exp(logy)
                    
                    top_p, top_c = y_cap.topk(1, dim=1) 
                    accuracy += get_accuracy(top_c, labels)
            model.train()
            print(f'Epoch: {epoch_idx + 1}, '
                f'train_loss: {train_loss/len(train_data):.3f}, '
                f'valid_loss: {vloss/len(valid_data):.3f}, '
                f'valid_acc: {accuracy/len(valid_data):.3f}')
    
    checkpoint_model(model, args.arch, args.save_dir)


if __name__=='__main__':
    main()