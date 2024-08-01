import torch.nn.functional as F
import torch
from transformers import LongformerModel
import torch.nn as nn

class LongformerWithNodeEmbeddings(nn.Module):
    def __init__(self, longformer_model, node_embedding_dim, combined_dim, device, num_heads=4):
        super(LongformerWithNodeEmbeddings, self).__init__()
        self.longformer = longformer_model.longformer
        self.num_heads = num_heads
        self.head_dim = 64

        self.W_q = nn.Linear(node_embedding_dim, self.num_heads * self.head_dim) # 128, 4 * 64
        self.W_k = nn.Linear(node_embedding_dim, self.num_heads * self.head_dim)
        self.W_v = nn.Linear(node_embedding_dim, self.num_heads * self.head_dim)
        self.W_o = nn.Linear(self.head_dim * self.num_heads, self.head_dim) # 256, 64

        self.layer_norm = nn.LayerNorm(longformer_model.config.hidden_size + self.head_dim)

        self.classifier = nn.Sequential(
            nn.Linear(longformer_model.config.hidden_size + self.head_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim, 200),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(200, longformer_model.config.num_labels)
        )
        self.device = device
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))

    def forward(self, input_ids, attention_mask, node1_embeddings, node2_embeddings, labels=None):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        # Transpose the last two dimensions
        node1_embeddings = node1_embeddings.permute(0, 2, 1)  # Shape [B, 128, N]
        node2_embeddings = node2_embeddings.permute(0, 2, 1)  # Shape [B, 128, M]

        extra_lawyer = pooled_output[:, :128].unsqueeze(1)  # Shape [B, 1, 128]

        node2_embeddings = torch.cat([node2_embeddings, extra_lawyer], dim=1)  # Shape [B, 128, M+1]

        batch_size = node1_embeddings.size(0)

        node1_q = self.W_q(node1_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, N, 64]
        node1_k = self.W_k(node1_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, N, 64]
        node1_v = self.W_v(node1_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, N, 64]

        node2_q = self.W_q(node2_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, M+1, 64]
        node2_k = self.W_k(node2_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, M+1, 64]
        node2_v = self.W_v(node2_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, 4, M+1, 64]

        attention_scores = torch.einsum('bhid,bhjd->bhij', node1_q, node2_k) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.einsum('bhij,bhjd->bhid', attention_probs, F.gelu(node2_v))

        transformed_node1_v = F.gelu(node1_v)
        combined_layer = context_layer + transformed_node1_v

        max_pooled_context, _ = torch.max(combined_layer, dim=2)
        concatenated_context = max_pooled_context.view(batch_size, -1)

        final_context = self.W_o(concatenated_context)

        combined_embedding = torch.cat((pooled_output, final_context), dim=1)
        combined_embedding = self.layer_norm(combined_embedding)
        logits = self.classifier(combined_embedding)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return logits, loss