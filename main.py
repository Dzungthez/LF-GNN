import logging
import argparse
import json
import os
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import json

import numpy as np
import pandas as pd
import csv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
from tqdm import tqdm
import sys
from dataset import ColieeDataset, create_train_val_dataframe
from sklearn.metrics import accuracy_score
from GCN_model import GCNAutoencoder
from LF_GCN import LongformerWithNodeEmbeddings

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_weight_matrix = "data/gcn_input/lawyers/weight_matrix_lids_1.csv"
file_matrix_path = "data/gcn_input/lawyers/adjacency_matrix_lids_1.csv"
case_meta_path = 'data/gcn_input/case_meta'

input_path = "data/longformer_data/input_longformer"
labels_path = "data/longformer_data/full_data_labels_test.json"

training = True
load_from_checkpoint = True

# Chuyển đổi ma trận kề và ma trận trọng số sang danh sách cạnh
def adj_to_edge_index(adj_matrix):
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def get_edge_attr(weight_matrix, edge_index):
    edge_attr = weight_matrix[edge_index[0], edge_index[1]]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    return edge_attr

def create_data(edge_index, edge_attr, num_nodes):
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.eye(num_nodes)  # Đặc trưng của các nút là ma trận đơn vị (identity matrix)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def compute_metrics2(pred):
    preds, labels = pred
    logits = preds
    preds = np.argmax(preds, axis=1)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    acc = accuracy_score(labels, preds)
    print(np.sum(preds))
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, dataloader, device):
    cnt = 0
    model.eval()
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            node1_embeddings = batch['node1'].to(device)
            node2_embeddings = batch['node2'].to(device)
            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, 
                    node1_embeddings=node1_embeddings, node2_embeddings=node2_embeddings, labels=labels)

            preds = logits.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics2((all_preds, all_labels)), all_preds

def save_checkpoint(model, optimizer, scheduler, epoch, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(state, file_path)
    print(f"Checkpoint saved at {file_path}")

def load_checkpoint(file_path, model, optimizer, scheduler=None):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}, starting from epoch {start_epoch}")
    return start_epoch

def gcn_training(num_nodes, edge_index, edge_attr):
    model = GCNAutoencoder(in_channels=num_nodes, hidden_channels=16, out_channels=out_channels)

    x = torch.eye(num_nodes)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    model.to(device)
    data.to(device)

    x = data.x
    edge_index = data.edge_index

    criterion = torch.nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        out, z = model(x, edge_index)
        adj_dense = to_dense_adj(edge_index)[0]
        loss = criterion(out, adj_dense)
        loss.backward()
        optimizer.step()
        
        # print(f'Epoch {epoch}, Loss: {loss.item()}')

        with torch.no_grad():
            output, embs = model(x, edge_index)
    return embs

def get_latest_checkpoint(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(directory, latest_file)

def training(model, optimizer, scheduler, train_loader, val_loader, num_epochs, device, training, load_from_checkpoint, checkpoint_dir):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if load_from_checkpoint and latest_checkpoint:
        if os.path.exists(latest_checkpoint):
            start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        else:
            print(f"Checkpoint file {latest_checkpoint} not found. Starting from scratch.")
    
    if training:
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
            
            # Ensure the logs directory exists
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
    adj_matrix = pd.read_csv(file_matrix_path)
    weight_matrix = pd.read_csv(file_weight_matrix)
    adj_matrix = adj_matrix.iloc[:, 1:]
    weight_matrix = weight_matrix.iloc[:, 1:]

    map_case = {}
    for (idx,col) in enumerate(weight_matrix.columns):
        map_case[col.split('.')[0]] = idx

    case_lawyer_mapping = {}
    for case in os.listdir(case_meta_path):
        with open(os.path.join(case_meta_path, case)) as f:
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

    out_channels = 128
    gcn_embs = gcn_training(num_nodes, edge_index, edge_attr)

    train_df, val_df = create_train_val_dataframe(input_path, labels_path)
    train_dataset = ColieeDataset(train_df, max_len=4096, lawyer_idx_to_embs=gcn_embs, lawyer_to_idx= map_case,
                                case_lawyer_mapping= case_lawyer_mapping,device = device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = ColieeDataset(val_df, max_len=4096, lawyer_idx_to_embs=gcn_embs, lawyer_to_idx= map_case,
                                case_lawyer_mapping= case_lawyer_mapping,device = device)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    longformer_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096').to(device)

    node_embedding_dim = 128
    combined_dim = 1000
    model = LongformerWithNodeEmbeddings(longformer_model, node_embedding_dim, combined_dim, device)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = None
    num_epochs = 10
    checkpoint_dir = 'checkpoints'
    # train_subset = [next(iter(train_loader)) for _ in range(200)]

    training(model, optimizer, scheduler, train_loader, val_loader, num_epochs, device, training, load_from_checkpoint, checkpoint_dir)
