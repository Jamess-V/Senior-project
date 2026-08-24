"""Training and fine-tuning script.""" 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from tqdm import tqdm 
import numpy as np 
import os 
import json 

def train_epoch(model, dataloader, criterion, optimizer, device): 
    """Train for one epoch.""" 
    model.train() 
    total_loss = 0 
    correct = 0 
    total = 0 

    for images, labels in tqdm(dataloader, desc="Training"): 
        images, labels = images.to(device), labels.to(device) 

        optimizer.zero_grad() 
        outputs = model(images).squeeze() 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

        total_loss += loss.item() 
        predictions = torch.sigmoid(outputs) > 0.5 
        correct += (predictions == labels).sum().item() 
        total += labels.size(0) 

    return total_loss / len(dataloader), correct / total 

def validate(model, dataloader, criterion, device): 
    """Validation loop.""" 
    model.eval() 
    total_loss = 0 
    correct = 0 
    total = 0 

    with torch.no_grad(): 
        for images, labels in tqdm(dataloader, desc="Validation"): 
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images).squeeze() 
            loss = criterion(outputs, labels) 

            total_loss += loss.item() 
            predictions = torch.sigmoid(outputs) > 0.5 
            correct += (predictions == labels).sum().item() 
            total += labels.size(0) 

    return total_loss / len(dataloader), correct / total  

def train_model(model, train_loader, val_loader, config): 
    """Main training loop.""" 
    device = config.device 
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5) 
     
    best_val_acc = 0 
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} 

    print("Starting training...") 

    for epoch in range(config.num_epochs): 
        print(f"\nEpoch {epoch+1}/{config.num_epochs}") 

        # Train 
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device) 
        # Validate 
        val_loss, val_acc = validate(model, val_loader, criterion, device) 

        # Update learning rate 
        scheduler.step(val_loss) 

        # Save history 
        history['train_loss'].append(train_loss) 
        history['val_loss'].append(val_loss) 
        history['train_acc'].append(train_acc) 
        history['val_acc'].append(val_acc) 

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}") 
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}") 

        # Save best model 
        if val_acc > best_val_acc: 
            best_val_acc = val_acc 
            checkpoint_path = os.path.join(config.output_dir, "checkpoints", "best_model.pth") 
            torch.save({ 
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'val_acc': val_acc, 
                'config': config 
            }, checkpoint_path) 
            print(f"Saved best model with val_acc: {val_acc:.4f}") 

    # Save training history 
    history_path = os.path.join(config.output_dir, "results", "training_history.json") 
    with open(history_path, 'w') as f: 
        json.dump(history, f, indent=2) 

    return model, history 