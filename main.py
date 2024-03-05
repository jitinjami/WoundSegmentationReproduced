'''
Main function, currently serving as test
'''
import random
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import Dice, Precision, Recall
import pandas as pd
import numpy as np
from src.data.dataset import ProcessedWoundDataset
from src.data.make_dataset import make_dataset1, make_dataset2
from src.utils import empty_directory
from src.models.mobilnetv2 import MobileNetV2withDecoder
from src.models.train_model import train, test
from config.defaults import get_cfg_defaults

def main():
    '''
    Main function, currently serving as test
    '''
    cfg = get_cfg_defaults()
    device = torch.device('mps')
    
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    #Empty the data directories except 'external' if indicated
    if cfg.CLEAR_DATA:
        for folder in Path(cfg.DATA_PATH).glob('*'):
            if folder.name != 'external' and folder.name != '.DS_Store':
                empty_directory(folder)
        cfg.MAKE_DATA = True

    if cfg.MAKE_DATA:
        #Make the dataset1 ready for training
        make_dataset1(cfg.DATA_PATH)

        #Make the dataset2 ready for training
        make_dataset2(cfg.DATA_PATH)
    
    datasets = {}

    datasets['train'] = ProcessedWoundDataset(cfg.DATA_PATH + 'processed/train/')
    datasets['val'] = ProcessedWoundDataset(cfg.DATA_PATH + 'processed/val/')
    datasets['test'] = ProcessedWoundDataset(cfg.DATA_PATH + 'processed/test/')

    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=cfg.BATCH_SIZE, shuffle=True,
                                 num_workers=cfg.NUM_WORKERS, drop_last=True) for x in ['train','val', 'test']}

    # Instantiate the model
    base_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')

    no_classifier_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))

    stripped_model = torch.nn.Sequential(*(list(no_classifier_model[0].children())[:-1]))

    model = MobileNetV2withDecoder(stripped_model, classes=1)
    if cfg.RESUME_TRAINING:
        model.load_state_dict(torch.load(cfg.MODELS_PATH + 'model_300.pt'))

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    device = torch.device('mps')

    #Evaluation Metrics
    dice_metric = Dice(threshold=0.5)
    precision_metric = Precision(task='binary')
    recall_metric = Recall(task='binary')

    metrics = [dice_metric, precision_metric, recall_metric]

    #Train model
    trained_model, train_results, val_results = train(model, dataloaders, device, 
                                                      criterion, optimizer, cfg.NUM_EPOCHS, metrics,
                                                      model_save_path=cfg.MODELS_PATH, metric_save_path=cfg.VIZ_PATH)
    
    #Test model
    test_results = test(trained_model, dataloaders, device, criterion, metrics)

    #Save metric dataframes
    train_results.to_csv('./train_results.csv')
    val_results.to_csv('./val_results.csv')
    test_results.to_csv('./test_results.csv')
    
if __name__ == '__main__':
    main()
