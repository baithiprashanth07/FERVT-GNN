import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

class GraphConstructor:
    """
    Utility class for constructing graphs from facial features
    """

    def __init__(self, method='grid', grid_size=7):
        """
        Initialize graph constructor

        Args:
            method: 'grid' or 'landmark' based graph construction
            grid_size: size of the grid for grid-based construction
        """
        self.method = method
        self.grid_size = grid_size

    def build_grid_graph(self, feature_map):
        """
        Build a grid-based graph from feature map

        Args:
            feature_map: tensor of shape (batch_size, channels, height, width)

        Returns:
            node_features: tensor of shape (batch_size * num_nodes, channels)
            edge_index: tensor of shape (2, num_edges)
            batch: batch tensor for PyTorch Geometric
        """
        batch_size, channels, height, width = feature_map.shape
        device = feature_map.device # Get device from feature_map

        # Resize feature map to grid_size x grid_size
        resized_features = F.adaptive_avg_pool2d(feature_map, (self.grid_size, self.grid_size))

        # Reshape to nodes: (batch_size, grid_size*grid_size, channels)
        num_nodes_per_graph = self.grid_size * self.grid_size
        node_features = resized_features.view(batch_size, channels, num_nodes_per_graph).permute(0, 2, 1)

        # Create edge index for grid connectivity (4-connected)
        edge_index = self._create_grid_edges(self.grid_size)

        # Flatten node features for PyTorch Geometric
        node_features_flat = node_features.reshape(-1, channels)

        # Create batch tensor
        batch = torch.arange(batch_size).repeat_interleave(num_nodes_per_graph)

        # Adjust edge_index for batched graphs
        edge_index_batched = []
        for i in range(batch_size):
            offset = i * num_nodes_per_graph
            edge_index_batched.append(edge_index + offset)
        edge_index_batched = torch.cat(edge_index_batched, dim=1)

        # Move tensors to the same device as the feature_map
        node_features_flat = node_features_flat.to(device)
        edge_index_batched = edge_index_batched.to(device)
        batch = batch.to(device)


        return node_features_flat, edge_index_batched, batch

    def _create_grid_edges(self, grid_size):
        """
        Create edge index for grid connectivity

        Args:
            grid_size: size of the grid

        Returns:
            edge_index: tensor of shape (2, num_edges)
        """
        edges = []

        for r in range(grid_size):
            for c in range(grid_size):
                node_idx = r * grid_size + c

                # Connect to neighbors (4-connected)
                if r > 0:  # Up
                    neighbor = (r-1) * grid_size + c
                    edges.append([node_idx, neighbor])
                if r < grid_size - 1:  # Down
                    neighbor = (r+1) * grid_size + c
                    edges.append([node_idx, neighbor])
                if c > 0:  # Left
                    neighbor = r * grid_size + (c-1)
                    edges.append([node_idx, neighbor])
                if c < grid_size - 1:  # Right
                    neighbor = r * grid_size + (c+1)
                    edges.append([node_idx, neighbor])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index


class AttentionGNN(nn.Module):
    """
    Attention Graph Neural Network module for facial feature enhancement
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
        """
        Initialize Attention GNN

        Args:
            input_dim: input feature dimension
            hidden_dim: hidden feature dimension
            output_dim: output feature dimension
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super(AttentionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, concat=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through Attention GNN

        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge connectivity (2, num_edges)
            batch: batch tensor for multiple graphs

        Returns:
            enhanced_features: enhanced node features
        """
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = self.layer_norm2(x)
        x = F.relu(x)

        return x


class GraphPooling(nn.Module):
    """
    Graph pooling module to aggregate node features back to image-level features
    """

    def __init__(self, pooling_method='mean'):
        """
        Initialize graph pooling

        Args:
            pooling_method: 'mean', 'max', or 'attention'
        """
        super(GraphPooling, self).__init__()
        self.pooling_method = pooling_method

    def forward(self, x, batch, grid_size=7):
        """
        Pool node features back to image-level features

        Args:
            x: node features (num_nodes, feature_dim)
            batch: batch tensor
            grid_size: size of the grid for reshaping

        Returns:
            pooled_features: image-level features
        """
        batch_size = batch.max().item() + 1
        feature_dim = x.size(1)
        num_nodes_per_graph = grid_size * grid_size

        if self.pooling_method == 'mean':
            # Global mean pooling
            pooled = torch.zeros(batch_size, feature_dim, device=x.device)
            for i in range(batch_size):
                mask = (batch == i)
                pooled[i] = x[mask].mean(dim=0)
            return pooled

        elif self.pooling_method == 'reshape':
            # Reshape back to grid format
            x_reshaped = x.view(batch_size, grid_size, grid_size, feature_dim)
            x_reshaped = x_reshaped.permute(0, 3, 1, 2)  # (batch, feature_dim, grid_size, grid_size)
            return x_reshaped

        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")