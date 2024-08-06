import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import functional as F

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        
    def forward(self, z, edge_index):
        z = self.conv1(z, edge_index)
        return z
    
class GCNAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNAutoencoder, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GCNDecoder(out_channels, in_channels)
        
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_index)
        return out, z
