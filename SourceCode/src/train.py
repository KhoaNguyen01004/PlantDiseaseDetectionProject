import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import argparse
from src.preprocessing.preprocess import load_config, build_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, cfg, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        print(f'Epoch {epoch+1}: Train L={train_loss/len(train_loader):.3f} Acc={train_acc:.1f}% | '
              f'Val L={val_loss/len(val_loader):.3f} Acc={val_acc:.1f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'💾 Best model saved (Val Acc: {best_acc:.1f}%)')
    
    print(f'✅ Training complete. Best model saved to models/best_model.pth')

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Classifier Training')
    parser.add_argument('--cfg', default='configs/config.yaml', help='Config path')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    cfg = load_config(args.cfg)
    dataloaders, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)
    print(f'Loaded {num_classes} classes from data/raw')
    
    # Model - EfficientNet_B0 (mobile optimized)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    classifier = model.classifier
    in_features = classifier[1].in_features
    model.classifier = nn.Sequential(classifier[0], nn.Linear(in_features, num_classes))
    model = model.to(DEVICE)
    print(f'Model on {DEVICE}')
    
    # Data
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print('Sample classes:', list(label_map.items())[:5])
    
    # Train
    train_model(model, train_loader, val_loader, cfg, args.epochs)

if __name__ == '__main__':
    main()
