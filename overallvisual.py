import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch # Import Batch
import warnings
warnings.filterwarnings('ignore')
import os # Import os module

# Your existing model code (GraphConstructor, AttentionGNN, etc.) would be imported here
# For demonstration, I'll include the key classes

class ModelVisualizer:
    """
    Comprehensive visualization tool for FERVT-GNN model
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.activations = {}
        self.feature_maps = {}
        self.attention_weights = {} # Dictionary to store attention weights

        # Register hooks to capture intermediate outputs
        self.register_hooks()

    def register_hooks(self):
        """Register forward hooks to capture intermediate activations"""

        def get_activation(name):
            def hook(model, input, output):
                # Store the output directly, whether it's a tensor or a tuple
                self.activations[name] = output
            return hook

        # Register hooks for key components
        if hasattr(self.model, 'gwa'):
            # Register hook for the entire GWA module's output
            self.model.gwa.register_forward_hook(get_activation('gwa_output'))

        if hasattr(self.model, 'gwa_f'):
            self.model.gwa_f.register_forward_hook(get_activation('gwa_fusion'))

        if hasattr(self.model, 'backbone'):
            # Register hook for the backbone module's output
            self.model.backbone.register_forward_hook(get_activation('backbone_output'))

        if hasattr(self.model, 'attention_gnn'):
            # Register hook for the GNN module's output
            self.model.attention_gnn.register_forward_hook(get_activation('gnn_output'))
            # # Also register hooks for GAT layers to get attention weights
            # # Note: _register_forward_hook might not be supported by torch_geometric GATConv
            # if hasattr(self.model.attention_gnn, 'gat1'):
            #      self.model.attention_gnn.gat1._register_forward_hook(self._get_attention_hook('gnn_gat1_attn'))
            # if hasattr(self.model.attention_gnn, 'gat2'):
            #      self.model.attention_gnn.gat2._register_forward_hook(self._get_attention_hook('gnn_gat2_attn'))


        if hasattr(self.model, 'vta'):
            self.model.vta.register_forward_hook(get_activation('vta_output'))
            # Register hooks for Transformer blocks to get attention weights
            if hasattr(self.model.vta, 'transformer') and hasattr(self.model.vta.transformer, 'blocks'):
                for i, block in enumerate(self.model.vta.transformer.blocks):
                    if hasattr(block, 'attn'):
                        # Assuming MultiHeadedSelfAttention has a 'scores' attribute or returns attention
                        block.attn.register_forward_hook(self._get_attention_hook(f'transformer_block_{i+1}_attn'))


    def _get_attention_hook(self, name):
        """Helper function to create attention hook"""
        def hook(module, input, output):
            # Attention weights are typically stored in a specific attribute (e.g., 'alpha')
            # or returned as part of a tuple output.
            # For PyG GATConv, attention weights are often returned as the second element of the output tuple
            # if `return_attention_weights=True` is set during initialization.
            # Since that's not the case here, we'll try to access a common attribute if available.
            # Note: This might need adjustment based on the exact GATConv implementation.
            if hasattr(module, 'alpha') and module.alpha is not None:
                 self.attention_weights[name] = module.alpha.detach().cpu()
            # For Transformer attention, scores are usually returned or stored
            elif hasattr(module, 'scores') and module.scores is not None:
                 self.attention_weights[name] = module.scores.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 1:
                 # Assuming attention weights are the second element if a tuple is returned
                 self.attention_weights[name] = output[1].detach().cpu()

        return hook


    def load_and_preprocess_image(self, image_path, target_size=(64, 64)):
        """Load and preprocess the input image"""

        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # Define preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        return input_tensor, original_image, np.array(image.resize(target_size))

    def visualize_complete_pipeline(self, image_path, save_path=None):
        """Visualize the complete model pipeline"""

        # Load and preprocess image
        input_tensor, original_image, resized_image = self.load_and_preprocess_image(image_path)

        # Forward pass through model to capture activations
        self.activations = {} # Clear previous activations
        self.attention_weights = {} # Clear previous attention weights
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 24))

        # 1. Original Input
        plt.subplot(6, 4, 1)
        plt.imshow(original_image)
        plt.title('Original Input Image', fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(6, 4, 2)
        plt.imshow(resized_image)
        plt.title('Resized Input (64x64)', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 2. GWA Processing
        if 'gwa_output' in self.activations:
            gwa_output = self.activations['gwa_output']
            if isinstance(gwa_output, tuple) and len(gwa_output) == 2 and isinstance(gwa_output[0], torch.Tensor) and isinstance(gwa_output[1], torch.Tensor):
                gwa_img, gwa_map = gwa_output

                # Original image from GWA
                plt.subplot(6, 4, 3)
                gwa_img_vis = self.denormalize_tensor(gwa_img[0])
                plt.imshow(gwa_img_vis)
                plt.title('GWA: Processed Image', fontsize=12, fontweight='bold')
                plt.axis('off')

                # Attention map from GWA
                plt.subplot(6, 4, 4)
                gwa_map_vis = self.denormalize_tensor(gwa_map[0])
                plt.imshow(gwa_map_vis)
                plt.title('GWA: Attention Map', fontsize=12, fontweight='bold')
                plt.axis('off')
            else:
                print("Warning: GWA output is not a tuple of 2 tensors. Skipping GWA visualization.")


        # 3. GWA Fusion
        if 'gwa_fusion' in self.activations and isinstance(self.activations['gwa_fusion'], torch.Tensor):
            fusion_output = self.activations['gwa_fusion'][0]
            plt.subplot(6, 4, 5)
            fusion_vis = self.denormalize_tensor(fusion_output)
            plt.imshow(fusion_vis)
            plt.title('GWA Fusion Output', fontsize=12, fontweight='bold')
            plt.axis('off')
        else:
             print("Warning: GWA Fusion output is not a tensor. Skipping GWA Fusion visualization.")


        # 4. GNN Processing (if available)
        if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
            self.visualize_gnn_processing(fig, input_tensor)

        # 5. Backbone Features
        if 'backbone_output' in self.activations and isinstance(self.activations['backbone_output'], torch.Tensor):
            backbone_features = self.activations['backbone_output'][0]  # Shape: (seq_len, feature_dim)

            plt.subplot(6, 4, 9)
            # Visualize backbone features as heatmap
            # Ensure we don't exceed the available tokens
            num_tokens = backbone_features.size(0)
            sns.heatmap(backbone_features[:min(num_tokens, 5)].cpu().numpy(), cmap='viridis', cbar=True) # Visualize first 5 tokens
            plt.title('Backbone Features (First few tokens)', fontsize=12, fontweight='bold')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Token Index')
            plt.yticks(np.arange(min(num_tokens, 5)) + 0.5, ['CLS', 'L1', 'L2', 'L3', 'GNN'][:min(num_tokens, 5)], rotation=0)
        else:
             print("Warning: Backbone output is not a tensor. Skipping Backbone visualization.")


        # 6. Transformer Attention Patterns
        self.visualize_attention_patterns(fig)

        # 7. Final Prediction
        plt.subplot(6, 4, 21)
        if isinstance(output, torch.Tensor):
            predictions = F.softmax(output, dim=1)[0].cpu().numpy()
            # Adjust emotion labels based on the expected output size (8 classes for FERVT_GNN)
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt'] # Assuming 8 classes

            # Ensure the number of labels matches the number of predictions
            if len(emotion_labels) != len(predictions):
                 print(f"Warning: Number of emotion labels ({len(emotion_labels)}) does not match prediction size ({len(predictions)}). Using generic labels.")
                 emotion_labels = [f'Class {i}' for i in range(len(predictions))]


            bars = plt.bar(range(len(predictions)), predictions, color='skyblue')
            plt.title('Emotion Predictions', fontsize=12, fontweight='bold')
            plt.xlabel('Emotions')
            plt.ylabel('Probability')
            plt.xticks(range(len(emotion_labels)), emotion_labels, rotation=45)

            # Highlight the predicted emotion
            max_idx = np.argmax(predictions)
            bars[max_idx].set_color('red')

            # Add prediction values on bars
            for i, v in enumerate(predictions):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            print("Warning: Model output is not a tensor. Skipping Final Prediction visualization.")


        # 8. Feature Statistics
        self.visualize_feature_statistics(fig)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return output, self.activations

    def visualize_gnn_processing(self, fig, input_tensor):
        """Visualize GNN processing steps"""

        # Get GWA fusion output for GNN input
        if 'gwa_fusion' in self.activations and isinstance(self.activations['gwa_fusion'], torch.Tensor):
            gnn_input = self.activations['gwa_fusion']

            # Build graph
            graph_constructor = self.model.graph_constructor
            node_features, edge_index, batch = graph_constructor.build_grid_graph(gnn_input)

            # Visualize graph structure
            plt.subplot(6, 4, 6)
            self.plot_graph_structure(edge_index.cpu(), graph_constructor.grid_size)
            plt.title('GNN Graph Structure (7x7 Grid)', fontsize=12, fontweight='bold')

            # Visualize input node features
            plt.subplot(6, 4, 7)
            # Ensure we take features from the first graph if batch size > 1
            num_nodes_per_graph = graph_constructor.grid_size * graph_constructor.grid_size
            node_features_vis = node_features[:num_nodes_per_graph].mean(dim=1).reshape(graph_constructor.grid_size, graph_constructor.grid_size).cpu().numpy()
            sns.heatmap(node_features_vis, cmap='plasma', cbar=True)
            plt.title('GNN Input Node Features (Avg)', fontsize=12, fontweight='bold')

            # GNN output visualization
            if 'gnn_output' in self.activations and isinstance(self.activations['gnn_output'], torch.Tensor):
                plt.subplot(6, 4, 8)
                gnn_out = self.activations['gnn_output'][:num_nodes_per_graph].mean(dim=1).reshape(graph_constructor.grid_size, graph_constructor.grid_size).cpu().numpy()
                sns.heatmap(gnn_out, cmap='plasma', cbar=True)
                plt.title('GNN Enhanced Features (Avg)', fontsize=12, fontweight='bold')
            else:
                 print("Warning: GNN output is not a tensor. Skipping GNN output visualization.")

        else:
             print("Warning: GWA Fusion output (GNN input) is not a tensor. Skipping GNN processing visualization.")


    def visualize_attention_patterns(self, fig):
        """Visualize transformer attention patterns"""

        # This requires capturing attention weights via hooks
        # Iterate through captured attention weights and plot
        attn_subplot_start = 10 # Start subplot index for attention maps
        attn_plotted_count = 0 # Initialize attn_plotted_count

        for name, attn_weights in self.attention_weights.items():
             if attn_plotted_count >= 4: # Limit number of attention maps shown
                 break

             # Attention weights shape for Transformer: (batch_size, num_heads, seq_len, seq_len)
             # Attention weights shape for GAT: (num_edges, num_heads) or (num_nodes, num_heads) depending on implementation
             if attn_weights.ndim == 4: # Assuming Transformer attention (batch, heads, seq, seq)
                plt.subplot(6, 4, attn_subplot_start + attn_plotted_count)
                # Average attention across heads and batches for simplicity
                avg_attn = attn_weights[0].mean(dim=0).cpu().numpy() # Take first batch, average heads

                # Define labels based on sequence length (assuming CLS + 3 pyramid + GNN)
                seq_len = avg_attn.shape[0]
                labels = ['CLS', 'L1', 'L2', 'L3', 'GNN'][:seq_len] # Adjust if sequence length varies

                sns.heatmap(avg_attn, cmap='Blues', cbar=True,
                           xticklabels=labels, yticklabels=labels)
                plt.title(f'{name} Attention', fontsize=12, fontweight='bold')
                plt.xlabel('Key Tokens')
                plt.ylabel('Query Tokens')
                attn_plotted_count += 1

             elif attn_weights.ndim == 2: # Assuming GAT attention (edges, heads) or (nodes, heads)
                 # GAT attention visualization is more complex, might skip for now or show stats
                 print(f"Skipping visualization for GAT attention weights: {name} (shape {attn_weights.shape})")
                 pass # Skip for now


        # If no attention maps were plotted, add a placeholder
        if attn_plotted_count == 0:
            plt.subplot(6, 4, attn_subplot_start)
            plt.text(0.5, 0.5, "No Attention Maps Captured", horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.title('Transformer Attention Pattern', fontsize=12, fontweight='bold')
            plt.axis('off')


    def visualize_feature_statistics(self, fig):
        """Visualize feature statistics across different layers"""

        plt.subplot(6, 4, 22)

        layer_names = []
        feature_norms = []

        # Filter for tensors and calculate norm
        for name, features in self.activations.items():
            if isinstance(features, torch.Tensor) and features.numel() > 0:
                layer_names.append(name.replace('_', '\n'))
                feature_norms.append(torch.norm(features.cpu()).item())

        if layer_names:
            plt.bar(range(len(layer_names)), feature_norms, color='lightcoral')
            plt.title('Feature Magnitudes by Layer', fontsize=12, fontweight='bold')
            plt.xlabel('Layers')
            plt.ylabel('L2 Norm')
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        else:
            plt.text(0.5, 0.5, "No Tensor Activations Captured", horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.title('Feature Magnitudes by Layer', fontsize=12, fontweight='bold')
            plt.axis('off')


    def plot_graph_structure(self, edge_index, grid_size):
        """Plot the graph structure"""

        # Create grid positions
        positions = {}
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                positions[node_id] = (j, grid_size - 1 - i)  # Flip y-axis for proper visualization

        # Plot nodes
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        plt.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7)

        # Plot edges
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            # Ensure node indices are within the valid range of positions
            if src < len(positions) and dst < len(positions):
                x1, y1 = positions[src]
                x2, y2 = positions[dst]
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)

        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)
        plt.axis('equal')

    def denormalize_tensor(self, tensor):
        """Denormalize tensor for visualization"""

        # Assuming ImageNet normalization was used
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        denormalized = tensor.cpu() * std + mean # Ensure tensor is on CPU before denormalization
        denormalized = torch.clamp(denormalized, 0, 1)

        return denormalized.permute(1, 2, 0).numpy()

    def create_detailed_analysis(self, image_path, save_dir=None):
        """Create detailed analysis with separate plots for each component"""

        input_tensor, original_image, resized_image = self.load_and_preprocess_image(image_path)

        self.activations = {} # Clear previous activations
        self.attention_weights = {} # Clear previous attention weights
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)

        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create individual detailed plots
        self.plot_gwa_analysis(save_dir)
        self.plot_gnn_analysis(save_dir)
        self.plot_backbone_analysis(save_dir)
        self.plot_transformer_analysis(save_dir)

        return output, self.activations

    def plot_gwa_analysis(self, save_dir=None):
        """Detailed GWA analysis"""

        if 'gwa_output' not in self.activations:
            print("GWA output not found in activations.")
            return

        gwa_output = self.activations['gwa_output']

        if not (isinstance(gwa_output, tuple) and len(gwa_output) == 2 and isinstance(gwa_output[0], torch.Tensor) and isinstance(gwa_output[1], torch.Tensor)):
            print("GWA output is not the expected tuple of two tensors. Skipping GWA analysis.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        gwa_img, gwa_map = gwa_output

        # Original processed image
        axes[0, 0].imshow(self.denormalize_tensor(gwa_img[0]))
        axes[0, 0].set_title('GWA Processed Image')
        axes[0, 0].axis('off')

        # Attention map
        axes[0, 1].imshow(self.denormalize_tensor(gwa_map[0]))
        axes[0, 1].set_title('GWA Attention Map')
        axes[0, 1].axis('off')

        # Channel-wise attention analysis
        if gwa_map.shape[1] > 0: # Check if there are channels to plot
            for i in range(gwa_map.shape[1]):
                axes[0, 2].plot(gwa_map[0, i].flatten().cpu().numpy(),
                              label=f'Channel {i}', alpha=0.7)
            axes[0, 2].set_title('Channel-wise Attention Distribution')
            axes[0, 2].legend()
        else:
            axes[0, 2].text(0.5, 0.5, "No Channels to Plot", horizontalalignment='center', verticalalignment='center')
            axes[0, 2].set_title('Channel-wise Attention Distribution')


        # Fusion result
        if 'gwa_fusion' in self.activations and isinstance(self.activations['gwa_fusion'], torch.Tensor):
            fusion_output = self.activations['gwa_fusion'][0]
            axes[1, 0].imshow(self.denormalize_tensor(fusion_output))
            axes[1, 0].set_title('GWA Fusion Result')
            axes[1, 0].axis('off')

            # Difference map
            if gwa_img[0].shape == fusion_output.shape:
                diff = torch.abs(gwa_img[0].cpu() - fusion_output.cpu())
                axes[1, 1].imshow(self.denormalize_tensor(diff))
                axes[1, 1].set_title('Difference Map (Original vs Fused)')
                axes[1, 1].axis('off')
            else:
                 axes[1, 1].text(0.5, 0.5, "Cannot Compute Difference Map", horizontalalignment='center', verticalalignment='center')
                 axes[1, 1].set_title('Difference Map (Original vs Fused)')
                 axes[1, 1].axis('off')


            # Statistics
            stats_text = f"""
            Original Mean: {gwa_img[0].mean().item():.4f}
            Fusion Mean: {fusion_output.mean().item():.4f}
            Difference Mean: {diff.mean().item():.4f} if 'diff' in locals() else 'N/A'
            Attention Strength: {gwa_map[0].std().item():.4f}
            """
            axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12,
                           verticalalignment='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('GWA Statistics')
            axes[1, 2].axis('off')
        else:
             axes[1, 0].text(0.5, 0.5, "GWA Fusion Output Not Found", horizontalalignment='center', verticalalignment='center')
             axes[1, 0].set_title('GWA Fusion Result')
             axes[1, 0].axis('off')
             axes[1, 1].axis('off') # Hide difference map if fusion output is missing
             axes[1, 2].text(0.5, 0.5, "GWA Fusion Output Not Found", horizontalalignment='center', verticalalignment='center')
             axes[1, 2].set_title('GWA Statistics')
             axes[1, 2].axis('off')


        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/gwa_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_gnn_analysis(self, save_dir=None):
        """Detailed GNN analysis"""

        if not hasattr(self.model, 'use_gnn') or not self.model.use_gnn:
            print("GNN is not enabled in the model.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        if 'gwa_fusion' in self.activations and isinstance(self.activations['gwa_fusion'], torch.Tensor):
            gnn_input = self.activations['gwa_fusion']

            # Node feature analysis
            graph_constructor = self.model.graph_constructor
            try:
                node_features, edge_index, batch = graph_constructor.build_grid_graph(gnn_input)
            except Exception as e:
                 print(f"Error building graph from GWA fusion output: {e}. Skipping GNN analysis.")
                 return

            # Input node features heatmap
            if node_features.shape[0] > 0: # Check if there are nodes
                num_nodes_per_graph = graph_constructor.grid_size * graph_constructor.grid_size
                if node_features.shape[0] >= num_nodes_per_graph: # Ensure there's at least one full graph
                    node_feat_grid = node_features[:num_nodes_per_graph].mean(dim=1).reshape(graph_constructor.grid_size, graph_constructor.grid_size).cpu().numpy()
                    sns.heatmap(node_feat_grid, cmap='plasma', ax=axes[0, 0], cbar=True)
                    axes[0, 0].set_title('Input Node Features')
                else:
                    axes[0, 0].text(0.5, 0.5, "Insufficient Nodes", horizontalalignment='center', verticalalignment='center')
                    axes[0, 0].set_title('Input Node Features')
            else:
                 axes[0, 0].text(0.5, 0.5, "No Nodes Generated", horizontalalignment='center', verticalalignment='center')
                 axes[0, 0].set_title('Input Node Features')


            # Graph connectivity
            if edge_index.shape[1] > 0: # Check if there are edges
                 self.plot_graph_structure_detailed(edge_index.cpu(), graph_constructor.grid_size, axes[0, 1])
                 axes[0, 1].set_title('Graph Connectivity (7x7 Grid)')
            else:
                 axes[0, 1].text(0.5, 0.5, "No Edges Generated", horizontalalignment='center', verticalalignment='center')
                 axes[0, 1].set_title('Graph Connectivity (7x7 Grid)')
                 axes[0, 1].axis('off') # Hide axis if no edges

            # Feature distribution
            if node_features.numel() > 0: # Check if there are features
                axes[0, 2].hist(node_features.flatten().cpu().numpy(), bins=50, alpha=0.7)
                axes[0, 2].set_title('Node Feature Distribution')
                axes[0, 2].set_xlabel('Feature Value')
                axes[0, 2].set_ylabel('Frequency')
            else:
                axes[0, 2].text(0.5, 0.5, "No Features to Plot", horizontalalignment='center', verticalalignment='center')
                axes[0, 2].set_title('Node Feature Distribution')


            if 'gnn_output' in self.activations and isinstance(self.activations['gnn_output'], torch.Tensor):
                gnn_output = self.activations['gnn_output']

                # Enhanced features
                if gnn_output.shape[0] >= num_nodes_per_graph: # Ensure there's at least one full graph
                    enhanced_grid = gnn_output[:num_nodes_per_graph].mean(dim=1).reshape(graph_constructor.grid_size, graph_constructor.grid_size).cpu().numpy()
                    sns.heatmap(enhanced_grid, cmap='plasma', ax=axes[1, 0], cbar=True)
                    axes[1, 0].set_title('GNN Enhanced Features')
                else:
                    axes[1, 0].text(0.5, 0.5, "Insufficient Enhanced Nodes", horizontalalignment='center', verticalalignment='center')
                    axes[1, 0].set_title('GNN Enhanced Features')


                # Feature enhancement comparison
                if node_feat_grid.shape == enhanced_grid.shape: # Ensure shapes match
                    improvement = enhanced_grid - node_feat_grid
                    sns.heatmap(improvement, cmap='RdBu_r', ax=axes[1, 1], cbar=True, center=0)
                    axes[1, 1].set_title('Feature Enhancement (GNN - Input)')
                else:
                     axes[1, 1].text(0.5, 0.5, "Cannot Compute Enhancement Map", horizontalalignment='center', verticalalignment='center')
                     axes[1, 1].set_title('Feature Enhancement (GNN - Input)')
                     axes[1, 1].axis('off')


                # Statistics
                stats_text = f"""
                Input Feature Mean: {node_features.mean().item():.4f}
                Enhanced Feature Mean: {gnn_output.mean().item():.4f}
                Attention Heads: {self.model.attention_gnn.gat1.heads}
                """
                axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12,
                               verticalalignment='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('GNN Statistics')
                axes[1, 2].axis('off')

            else:
                 axes[1, 0].text(0.5, 0.5, "GNN Output Not Found", horizontalalignment='center', verticalalignment='center')
                 axes[1, 0].set_title('GNN Enhanced Features')
                 axes[1, 0].axis('off')
                 axes[1, 1].axis('off') # Hide enhancement map if GNN output is missing
                 axes[1, 2].text(0.5, 0.5, "GNN Output Not Found", horizontalalignment='center', verticalalignment='center')
                 axes[1, 2].set_title('GNN Statistics')
                 axes[1, 2].axis('off')


        else:
             print("GWA Fusion output (GNN input) not found or not a tensor. Skipping GNN analysis.")
             for ax_row in axes:
                 for ax in ax_row:
                     ax.axis('off')


        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/gnn_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_backbone_analysis(self, save_dir=None):
        """Detailed Backbone analysis"""
        if 'backbone_output' not in self.activations or not isinstance(self.activations['backbone_output'], torch.Tensor):
             print("Backbone output not found or not a tensor. Skipping Backbone analysis.")
             return

        backbone_output = self.activations['backbone_output'][0] # Take first batch

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Feature heatmap
        num_tokens = backbone_output.size(0)
        sns.heatmap(backbone_output[:min(num_tokens, 10)].cpu().numpy(), cmap='viridis', cbar=True, ax=axes[0]) # Visualize first 10 tokens
        axes[0].set_title('Backbone Output Features')
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Token Index')
        axes[0].set_yticks(np.arange(min(num_tokens, 10)) + 0.5, ['CLS', 'L1', 'L2', 'L3', 'GNN'][:min(num_tokens, 10)], rotation=0)


        # Feature distribution
        if backbone_output.numel() > 0:
             axes[1].hist(backbone_output.flatten().cpu().numpy(), bins=50, alpha=0.7)
             axes[1].set_title('Backbone Feature Distribution')
             axes[1].set_xlabel('Feature Value')
             axes[1].set_ylabel('Frequency')
        else:
             axes[1].text(0.5, 0.5, "No Features to Plot", horizontalalignment='center', verticalalignment='center')
             axes[1].set_title('Backbone Feature Distribution')


        # Statistics
        stats_text = f"""
        Mean: {backbone_output.mean().item():.4f}
        Std: {backbone_output.std().item():.4f}
        Min: {backbone_output.min().item():.4f}
        Max: {backbone_output.max().item():.4f}
        Num Tokens: {num_tokens}
        Feature Dim: {backbone_output.size(1)}
        """
        axes[2].text(0.1, 0.5, stats_text, fontsize=12,
                       verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Backbone Statistics')
        axes[2].axis('off')

        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/backbone_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_transformer_analysis(self, save_dir=None):
        """Detailed Transformer analysis, including attention"""

        # Check if transformer attention weights were captured
        transformer_attn_keys = [key for key in self.attention_weights.keys() if 'transformer_block' in key]
        if not transformer_attn_keys:
            print("No Transformer attention weights captured. Skipping Transformer analysis.")
            return

        # Plot attention heatmaps for captured layers
        num_attn_plots = min(len(transformer_attn_keys), 4) # Plot up to 4 attention maps
        fig, axes = plt.subplots(1, num_attn_plots, figsize=(5 * num_attn_plots, 5))

        if num_attn_plots == 1: # Ensure axes is iterable
             axes = [axes]

        for i, key in enumerate(transformer_attn_keys[:num_attn_plots]):
             attn_weights = self.attention_weights[key]

             if attn_weights.ndim == 4: # Assuming Transformer attention (batch, heads, seq, seq)
                # Average attention across heads and batches for simplicity
                avg_attn = attn_weights[0].mean(dim=0).cpu().numpy() # Take first batch, average heads

                # Define labels based on sequence length (assuming CLS + 3 pyramid + GNN)
                seq_len = avg_attn.shape[0]
                labels = ['CLS', 'L1', 'L2', 'L3', 'GNN'][:seq_len] # Adjust if sequence length varies

                sns.heatmap(avg_attn, cmap='Blues', cbar=True,
                           xticklabels=labels, yticklabels=labels, ax=axes[i])
                axes[i].set_title(f'{key.replace("_", " ")}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Key Tokens')
                axes[i].set_ylabel('Query Tokens')
             else:
                 axes[i].text(0.5, 0.5, f"Cannot Plot {key}", horizontalalignment='center', verticalalignment='center')
                 axes[i].set_title(f'{key.replace("_", " ")}', fontsize=12, fontweight='bold')
                 axes[i].axis('off')


        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/transformer_attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_graph_structure_detailed(self, edge_index, grid_size, ax):
        """Detailed graph structure plot"""

        positions = {}
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                positions[node_id] = (j, grid_size - 1 - i)

        # Plot nodes with different colors based on position
        for node_id, (x, y) in positions.items():
            color = 'red' if (x == 0 or x == grid_size-1 or y == 0 or y == grid_size-1) else 'lightblue'
            ax.scatter(x, y, c=color, s=100, alpha=0.7)

        # Plot edges
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            # Ensure node indices are within the valid range of positions
            if src < len(positions) and dst < len(positions):
                x1, y1 = positions[src]
                x2, y2 = positions[dst]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')


# Example Usage:
# Assuming FERVT_GNN model and necessary components (GraphConstructor, AttentionGNN, etc.) are defined
# from model import FERVT_GNN # Replace with your actual import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FERVT_GNN(device=device, use_gnn=True)

# Create visualizer
visualizer = ModelVisualizer(model, device)

# Visualize complete pipeline
image_path = "/content/drive/MyDrive/FERVT/dataset/CK+/happy/S010_006_00000013.png"
output, activations = visualizer.visualize_complete_pipeline(image_path, save_path="model_pipeline.png")

# Create detailed analysis
output, activations = visualizer.create_detailed_analysis(image_path, save_dir="./analysis")