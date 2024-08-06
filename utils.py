import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj  # Ensure this import is present

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
    x = torch.eye(num_nodes)
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
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            node1_embeddings = batch['node1'].to(device)
            node2_embeddings = batch['node2'].to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, 
                    node1_embeddings=node1_embeddings, node2_embeddings=node2_embeddings, labels=labels)

            preds = logits.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics2((all_preds, all_labels)), all_preds

def save_checkpoint(model, optimizer, scheduler, epoch, file_path):
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

def get_latest_checkpoint(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(directory, latest_file)

def gcn_training(num_nodes, edge_index, edge_attr, device):
    from GCN_model import GCNAutoencoder
    model = GCNAutoencoder(in_channels=num_nodes, hidden_channels=16, out_channels=128)

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

    with torch.no_grad():
        _, embs = model(x, edge_index)
    return embs
