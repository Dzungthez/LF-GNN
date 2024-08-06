import logging
import argparse
import json
import os
import torch
from transformers import LongformerForSequenceClassification
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import ColieeDataset, create_train_val_dataframe
from LF_GCN import LongformerWithNodeEmbeddings
from utils import (
    adj_to_edge_index,
    get_edge_attr,
    compute_metrics2,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    gcn_training
)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    return device, n_gpu

def download_and_extract_dataset():
    import kaggle
    import zipfile

    kaggle.api.dataset_download_files('nhddddz84/lf-gcn-data', path='/app/data', unzip=True)

def training(model, optimizer, scheduler, train_loader, val_loader, num_epochs, device, n_gpu, training, load_from_checkpoint, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if load_from_checkpoint and latest_checkpoint:
        if os.path.exists(latest_checkpoint):
            start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        else:
            print(f"Checkpoint file {latest_checkpoint} not found. Starting from scratch.")
    
    if training:
        if n_gpu > 1:
            model = nn.DataParallel(model)
        
        model.to(device)

        for epoch in range(start_epoch, num_epochs + start_epoch):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs + epoch}', leave=False)
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                node1_embeddings = batch['node1'].to(device)
                node2_embeddings = batch['node2'].to(device)
                
                logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, 
                                     node1_embeddings=node1_embeddings, node2_embeddings=node2_embeddings, labels=labels)
                
                # Ensure the loss is averaged over the batch
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss/len(train_loader))
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
            
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)
            
            val_metrics, preds = evaluate(model, val_loader, device)
            print(f'Validation metrics: {val_metrics}')
            
            os.makedirs('logs', exist_ok=True)
            with open(f'logs/metrics_gcn_lf_att_epoch{epoch+1}.json', 'w') as f:
                json.dump(val_metrics, f)
            model.train()
    else:
        print("Training is disabled. Set 'training' to True to enable training.")
        print('-----Evaluating------')
        val_metrics, preds = evaluate(model, val_loader, device)
        print(f'Validation metrics: {val_metrics}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Longformer with GCN embeddings')
    parser.add_argument('--input_path', type=str, default="data/longformer_data/input_longformer", help='Path to input data')
    parser.add_argument('--labels_path', type=str, default="data/longformer_data/full_data_labels_test.json", help='Path to labels')
    parser.add_argument('--file_weight_matrix', type=str, default="data/gcn_input/lawyers/weight_matrix_lids_1.csv", help='Path to weight matrix')
    parser.add_argument('--file_matrix_path', type=str, default="data/gcn_input/lawyers/adjacency_matrix_lids_1.csv", help='Path to adjacency matrix')
    parser.add_argument('--case_meta_path', type=str, default='data/gcn_input/case_meta', help='Path to case meta data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--load_from_checkpoint', action='store_true', help='Load from checkpoint')
    parser.add_argument('--training', action='store_true', help='Enable training')
    parser.add_argument('--download_dataset', action='store_true', help='Download dataset from Kaggle')
    args = parser.parse_args()

    device, n_gpu = setup_device()

    if args.download_dataset:
        download_and_extract_dataset()

    adj_matrix = pd.read_csv(args.file_matrix_path)
    weight_matrix = pd.read_csv(args.file_weight_matrix)
    adj_matrix = adj_matrix.iloc[:, 1:]
    weight_matrix = weight_matrix.iloc[:, 1:]

    map_case = {}
    for idx, col in enumerate(weight_matrix.columns):
        map_case[col.split('.')[0]] = idx

    case_lawyer_mapping = {}
    for case in os.listdir(args.case_meta_path):
        with open(os.path.join(args.case_meta_path, case)) as f:
            data = json.load(f)
            case_id = data['id']
            case_lawyer_mapping[case_id] = []
            for counsel in data['counsels']:
                if counsel['cid'] in map_case:
                    case_lawyer_mapping[case_id].append(counsel['cid'])

    adj_matrix = adj_matrix.to_numpy()
    weight_matrix = weight_matrix.to_numpy()

    edge_index = adj_to_edge_index(adj_matrix)
    edge_attr = get_edge_attr(weight_matrix, edge_index)
    num_nodes = adj_matrix.shape[0]

    gcn_embs = gcn_training(num_nodes, edge_index, edge_attr, device)

    train_df, val_df = create_train_val_dataframe(args.input_path, args.labels_path)
    train_dataset = ColieeDataset(train_df, max_len=4096, lawyer_idx_to_embs=gcn_embs, lawyer_to_idx=map_case,
                                  case_lawyer_mapping=case_lawyer_mapping, device=device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ColieeDataset(val_df, max_len=4096, lawyer_idx_to_embs=gcn_embs, lawyer_to_idx=map_case,
                                case_lawyer_mapping=case_lawyer_mapping, device=device)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    longformer_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096').to(device)

    node_embedding_dim = 128
    combined_dim = 1000
    model = LongformerWithNodeEmbeddings(longformer_model, node_embedding_dim, combined_dim, device)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    training(model, optimizer, scheduler, train_loader, val_loader, args.num_epochs, device, n_gpu, args.training, args.load_from_checkpoint, args.checkpoint_dir)
