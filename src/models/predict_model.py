import torch
import pandas as pd
import tqdm

def test(model, dataloaders, device, criterion, metrics, model_name, metric_save_path):
    
    test_metrics = {'dice': metrics[0], 'precision': metrics[1], 'recall':metrics[2], 
                    'iou':metrics[3]}

    test_df = pd.DataFrame(columns=["Loss", "Dice", "Precision", "Recall", "IoU"])
    
    # Test Model
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloaders['test']):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            #Update metrics
            test_metrics['dice'].update(outputs, masks.int())
            test_metrics['precision'].update(outputs, masks)
            test_metrics['recall'].update(outputs, masks)
            test_metrics['iou'].update(outputs, masks)

        # Append val epoch results
        test_df.loc[len(test_df)] = {
            "Loss": loss.item(),
            "Dice": test_metrics['dice'].compute().item(),
            "Precision": test_metrics['precision'].compute().item(),
            "Recall": test_metrics['recall'].compute().item(),
            "IoU": test_metrics['iou'].compute().item()
        }
        #Saving metrics every epoch
        test_df.to_csv(metric_save_path + f'{model_name}_test_results.csv')

    return test_df