import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision

import data_utils
import train_eval_utils

def main(
    epochs, 
    lr, 
    weighted_loss,
    batch_size,
    num_workers
    ):

    trainloader, validloader, testloader, trainset, data_df, pathology_list = data_utils.get_nih_subset(
        batch_size = batch_size,
        num_workers = num_workers
    )

    freq_pos, freq_neg = train_eval_utils.compute_class_freqs(data_df.iloc[:,1:])
    df = pd.DataFrame({"Class": pathology_list, "Label": "Positive", "Value": freq_pos})
    df = df.append([{"Class": pathology_list[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
    if weighted_loss:
        pos_weights = freq_neg
        neg_weights = freq_pos
        pos_contribution = freq_pos * pos_weights 
        neg_contribution = freq_neg * neg_weights
        df = pd.DataFrame({"Class": pathology_list, "Label": "Positive", "Value": pos_contribution})
        df = df.append([{"Class": pathology_list[l], "Label": "Negative", "Value": v} for l,v in enumerate(neg_contribution)], ignore_index=True)
    else:
        pos_weights = 1.0
        neg_weights = 1.0

    model = torchvision.models.resnet18(
        weights="IMAGENET1K_V1"
        )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 14),
        torch.nn.Sigmoid()
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor = 0.1,
        patience = 4
        )

    train_eval_utils.train(
        model,
        epochs,
        trainloader,
        validloader,
        optimizer,
        scheduler,
        pos_weights,
        neg_weights,
        device
        )

    print("Train Dataset Accuracy Report")
    train_acc_list = train_eval_utils.class_accuracy(
        device, 
        trainloader, 
        model, 
        pathology_list
        )
    train_df = train_eval_utils.get_acc_data(pathology_list,train_acc_list)
    print(train_df)
    train_df.to_csv(
        f'./result_stats/train_df_epochs_{epochs}.csv'
    )

    print("Test Dataset Accuracy Report")
    test_acc_list = train_eval_utils.class_accuracy(
        device, 
        testloader, 
        model, 
        pathology_list
        )
    test_df = train_eval_utils.get_acc_data(pathology_list, test_acc_list)
    test_df.to_csv(
        f'./result_stats/test_df_epochs_{epochs}.csv'
    )
    print(test_df)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # HPS #
    epochs = 15
    lr = 1e-4
    weighted_loss=True
    batch_size = 32
    num_workers = 8

    main(
        epochs = epochs,
        lr = lr,
        weighted_loss=weighted_loss,
        batch_size = batch_size,
        num_workers = num_workers
    )

    

    

    




