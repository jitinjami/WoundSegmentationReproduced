'''
Has the model training and testing function
'''
import pandas as pd
from tqdm import tqdm
import torch

def train_epoch(model, dataloaders, device, criterion, optimizer, train_metrics, train_df, epoch):
    model.train()
    for images, masks in tqdm(dataloaders['train']):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Update metrics
        train_metrics['dice'].update(outputs, masks.int())
        train_metrics['precision'].update(outputs, masks)
        train_metrics['recall'].update(outputs, masks)
        train_metrics['iou'].update(outputs, masks)

    # Append training epoch results
    train_df.loc[len(train_df)] = {
        "Epoch": epoch,
        "Loss": loss.item(),
        "Dice": train_metrics['dice'].compute().item(),
        "Precision": train_metrics['precision'].compute().item(),
        "Recall": train_metrics['recall'].compute().item(),
        "IoU": train_metrics['iou'].compute().item()
        }
    return model, train_metrics, train_df

def val_epoch(model, dataloaders, device, criterion, val_metrics, val_df, epoch):
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloaders['val']):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            #Update metrics
            val_metrics['dice'].update(outputs, masks.int())
            val_metrics['precision'].update(outputs, masks)
            val_metrics['recall'].update(outputs, masks)
            val_metrics['iou'].update(outputs, masks)

        # Append val epoch results
        val_df.loc[len(val_df)] = {
            "Epoch": epoch,
            "Loss": loss.item(),
            "Dice": val_metrics['dice'].compute().item(),
            "Precision": val_metrics['precision'].compute().item(),
            "Recall": val_metrics['recall'].compute().item(),
            "IoU": val_metrics['iou'].compute().item(),
            }
    
    return model, val_metrics, val_df



def train(model, dataloaders, device, criterion, optimizer, num_epochs, 
          metrics, model_save_path, model_name, metric_save_path):
    '''
    Train the model
    Args:
        model: Torch model
        dataloaders: Dict of dataloders with 'train', 'val' and possible 'test' as keys
        device: For CPU/GPU
        criterion: loss function
        optimizer: PyTorch optimizer
        num_epochs: number of epochs
        metrics: List of metrics
    '''
    train_metrics = {'dice': metrics[0].to(device), 'precision': metrics[1].to(device), 
                     'recall':metrics[2].to(device), 'iou':metrics[3].to(device)}
    val_metrics = {'dice': metrics[0].to(device), 'precision': metrics[1].to(device), 
                   'recall':metrics[2].to(device), 'iou':metrics[3].to(device)}
    
    #Pandas dataframes for evaluation metrics
    train_df = pd.DataFrame(columns=["Epoch", "Loss", "Dice", "Precision", "Recall", "IoU"])
    val_df = pd.DataFrame(columns=["Epoch", "Loss", "Dice", "Precision", "Recall", "IoU"])

    # Training the model
    model = model.to(device)

    #Early stopping parameters
    wait_count = 0
    tol = 1e-4
    best_val_dice_coeff = -tol
    patience = 200

    for epoch in range(num_epochs):

        print(f"Epoch: {epoch}/{num_epochs}")

        # Train Epoch
        model, train_metrics, train_df = train_epoch(model, dataloaders, device, 
                                                     criterion, optimizer, train_metrics, 
                                                     train_df, epoch)
        
        
        #Saving train metrics every epoch
        train_df.to_csv(metric_save_path + 'train_results.csv')
        val_df.to_csv(metric_save_path + 'val_results.csv')

        # Validate and Save model
        if model_name == 'MobileNetv2':
            if epoch % 100 == 0:
                # Validate Model
                model, val_metrics, val_df = val_epoch(model, dataloaders, device, 
                                               criterion, val_metrics, 
                                               val_df, epoch)
                torch.save(model.state_dict(), model_save_path + f'{model_name}_model_{epoch}.pt')

        if model_name == 'WSNet':
            if epoch % 10 == 0:
                model, val_metrics, val_df = val_epoch(model, dataloaders, device, 
                                               criterion, val_metrics, 
                                               val_df, epoch)
                torch.save(model.state_dict(), model_save_path + f'{model_name}_model_{epoch}.pt')

        # Early Stopping
        if val_df["Dice"].iloc[-1] - best_val_dice_coeff > tol:
            best_val_dice_coeff = val_df["Dice"].iloc[-1]
            best_epoch = epoch
            best_model = model.state_dict()  # Save the best model weights
            wait_count = 0
            print('Dice increased')
        else:
            wait_count += 1
            print(f"Dice hasn't changed much in {wait_count} epochs.")

        if epoch >= 100 and wait_count >= patience:
            print(f"Early stopping triggered after {wait_count} epochs!")
            torch.save(best_model.state_dict(), 
                       model_save_path + f'{model_name}_best_model_{best_epoch}.pt')
            return model, train_df, val_df

    return model, train_df, val_df

def test(model, dataloaders, device, criterion, metrics, metric_save_path):
    
    test_metrics = {'dice': metrics[0], 'precision': metrics[1], 'recall':metrics[2], 
                    'iou':metrics[3].to(device)}

    test_df = pd.DataFrame(columns=["Loss", "Dice", "Precision", "Recall", "IoU"])
    
    # Test Model
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloaders['val']):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            #Update metrics
            test_metrics['dice'].update(outputs, masks.int())
            test_metrics['precision'].update(outputs, masks)
            test_metrics['recall'].update(outputs, masks)

        # Append val epoch results
        test_df.loc[len(test_df)] = {
            "Loss": loss.item(),
            "Dice": test_metrics['dice'].compute().item(),
            "Precision": test_metrics['precision'].compute().item(),
            "Recall": test_metrics['recall'].compute().item(),
        }
        #Saving metrics every epoch
        test_df.to_csv(metric_save_path + 'test_results.csv')

    return test_df