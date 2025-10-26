import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
from torch_geometric.nn import GATConv
from graph import GraphConstructor,GraphPooling,AttentionGNN

from transformer import Transformer

# backbone + token_embedding + position_embedding
class Backbone(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(Backbone, self).__init__()

        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Adaptive pooling layers to handle varying input sizes
        self.adaptive_pool_l1 = nn.AdaptiveAvgPool2d((21, 21)) # Target size for convtran1
        self.adaptive_pool_l2 = nn.AdaptiveAvgPool2d((7, 7))   # Target size for convtran2
        self.adaptive_pool_l3 = nn.AdaptiveAvgPool2d((1, 1))   # Target size for convtran3 to output 1x1

        #  feature resize networks
        # shape trans 128
        # Changed output channels from 3 to 192 to match layernorm input
        self.convtran1 = nn.Conv2d(128, 192, 21, 1)
        self.bntran1 = nn.BatchNorm2d(192)
        self.convtran2 = nn.Conv2d(256, 192, 7, 1)
        self.bntran2 = nn.BatchNorm2d(192)
        # Changed kernel_size to 1 to output 1x1 when input is 1x1 from adaptive_pool_l3
        self.convtran3 = nn.Conv2d(512, 192, 1, 1)
        self.bntran3 = nn.BatchNorm2d(192)
        # Visual Token Embedding.
        self.layernorm = nn.LayerNorm(192)
        self.dropout = nn.Dropout(0.2)
        self.line = nn.Linear(192, 192)
        # class token init
        self.class_token = nn.Parameter(torch.zeros(1, 192))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(5, 192))  # Changed from 4 to 5 to accommodate GNN token

        self.apply(self.weight_init)

    def forward(self, x, gnn_features=None):
        batchsize = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)
        # Apply adaptive pooling before convtran1
        l1_pooled = self.adaptive_pool_l1(x)
        # L1  feature transformation from the pyramid features
        l1 = F.leaky_relu(self.bntran1(self.convtran1(l1_pooled)))
        # Flatten to (batchsize, 192)
        l1 = l1.view(batchsize, 192)
        # L1 token_embedding to T1    L1(1xCHW)--->T1(1xD)
        # in this model D=192
        l1 = self.line(self.dropout(F.relu(self.layernorm(l1))))
        l1 = l1.unsqueeze(1) # Add sequence dimension for transformer

        x = self.layer3(x)
        # Apply adaptive pooling before convtran2
        l2_pooled = self.adaptive_pool_l2(x)
        l2 = F.leaky_relu(self.bntran2(self.convtran2(l2_pooled)))
        l2 = l2.view(batchsize, 192)
        l2 = self.line(self.dropout(F.relu(self.layernorm(l2))))
        l2 = l2.unsqueeze(1)

        x = self.layer4(x)
        # Apply adaptive pooling before convtran3
        l3_pooled = self.adaptive_pool_l3(x)
        l3 = F.leaky_relu(self.bntran3(self.convtran3(l3_pooled)))
        l3 = l3.view(batchsize, 192)
        l3 = self.line(self.dropout(F.relu(self.layernorm(l3))))
        l3 = l3.unsqueeze(1)

        x = torch.cat((l1, l2), dim=1)
        x = torch.cat((x, l3), dim=1)
        
        # Add GNN features if provided
        if gnn_features is not None:
            gnn_features = gnn_features.unsqueeze(1)  # (batch_size, 1, 192)
            x = torch.cat((x, gnn_features), dim=1)  # (batch_size, 4, 192)
        
        x = torch.cat((self.class_token.expand(batchsize, 1, 192), x), dim=1)
        x = x + self.pos_embedding.expand(batchsize, x.shape[1], 192)

        return x


#  refer to SubSection 3.3
# input: img(batchsize,c,h,w)--->output: img_feature_map(batchsize,c,h,w)
# in FER+ (b,3,48,48)
class GWA(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA, self).__init__()
        # low level feature extraction
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        # 图像分割，每块16x16，用一个卷积层实现
        # Adjust kernel_size and stride based on expected input image size
        # For 64x64 input, if we want 16 patches, each patch would be 16x16 (64/4 = 16)
        # So, kernel_size and stride should be 16
        self.patch_size = 16 # Assuming 64x64 input, 16x16 patches -> 4x4 grid = 16 patches
        self.num_patches_per_dim = 64 // self.patch_size # 4
        self.num_patches = self.num_patches_per_dim * self.num_patches_per_dim # 16
        self.patch_embeddings = nn.Conv2d(in_channels=3,
                                          out_channels=3 * self.patch_size * self.patch_size,
                                          kernel_size=(self.patch_size, self.patch_size),
                                          stride=(self.patch_size, self.patch_size))
        # 使用自适应pool压缩一维
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self.weight_init)

    def forward(self, x):
        img = x
        batchsize = x.shape[0]
        device = x.device
        
        x = self.patch_embeddings(x)
        
        x = x.flatten(2).transpose(-1, -2).view(batchsize, self.num_patches, 3, self.patch_size, self.patch_size)
        
        temp = []
        for i in range(x.shape[1]):
            temp.append(F.leaky_relu(self.bn2(self.conv2(
                F.leaky_relu(self.bn1(self.conv1(x[:, i, :, :, :])))))).unsqueeze(0).transpose(0, 1))

        x = torch.cat(tuple(temp), dim=1)
        query = x
        key = torch.transpose(query, 3, 4)
        # The division factor in softmax should be based on the feature dimension of the patches, not fixed 56
        # The feature dimension of each patch is patch_size
        attn = F.softmax(torch.matmul(query, key) / self.patch_size,dim=1)
        temp = []
        for i in range(attn.shape[1]):
            temp.append(self.aap(attn[:, i, :, :, :]).unsqueeze(0).transpose(0, 1))
        
        pooled_attn_values = torch.cat(tuple(temp), dim=1)
        
        resized_attn = pooled_attn_values.view(batchsize, 3, self.num_patches_per_dim, self.num_patches_per_dim)
        
        # Upsample to original image size
        pattn = F.interpolate(resized_attn, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
        
        map = pattn * img
        return img, map


class GWA_Fusion(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA_Fusion, self).__init__()
        # 原图特征转换网络
        self.convt1 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt1 = nn.BatchNorm2d(3)
        # map特征转换网络
        self.convt2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt2 = nn.BatchNorm2d(3)
        # RFN参与特征融合网络
        self.convrfn1 = nn.Conv2d(3,3,(3,3),1,1)
        self.bnrfn1 = nn.BatchNorm2d(3)
        self.prelu1 = nn.PReLU(3)
        self.convrfn2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnrfn2 = nn.BatchNorm2d(3)
        self.prelu2 = nn.PReLU(3)
        self.convrfn3 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.sigmod = nn.Sigmoid()

        self.apply(self.weight_init)

    def forward(self, img, map):
        img_trans = F.relu(self.bnt1(self.convt1(img)))
        map_trans = F.relu(self.bnt2(self.convt2(map)))
        result = self.prelu1(self.bnrfn1(self.convrfn1(img_trans + map_trans)))
        result = self.prelu2(self.bnrfn2(self.convrfn2(result)))
        result = self.sigmod(self.convrfn3(result+img_trans + map_trans))

        return result


class VTA(nn.Module):
    def __init__(self):
        super(VTA, self).__init__()

        self.transformer = Transformer(num_layers=12, dim=192, num_heads=8,
                                       ff_dim=768, dropout=0.1)
        self.layernorm = nn.LayerNorm(192)
        self.fc = nn.Linear(192, 8)

    def forward(self, x):
        x = self.transformer(x)
        x = self.layernorm(x)[:, 0, :]
        x = self.fc(x)
        return x


class FERVT_GNN(nn.Module):
    def __init__(self, device, use_gnn=True):
        super(FERVT_GNN, self).__init__()
        self.use_gnn = use_gnn
        self.device = device
        
        self.gwa = GWA()
        self.gwa.to(device)
        self.gwa_f = GWA_Fusion()
        self.gwa_f.to(device)
        self.backbone = Backbone()
        self.backbone.to(device)
        self.vta = VTA()
        self.vta.to(device)

        # GNN Integration
        if self.use_gnn:
            self.graph_constructor = GraphConstructor(method='grid', grid_size=7)
            self.attention_gnn = AttentionGNN(input_dim=3, hidden_dim=128, output_dim=192)  # input_dim=3 for RGB channels
            self.graph_pooling = GraphPooling(pooling_method='mean')
            self.gnn_projection = nn.Linear(192, 192)  # Project GNN features to match backbone dimension
            
            self.attention_gnn.to(device)
            self.gnn_projection.to(device)

        self.to(device)

    def forward(self, x):
        # Grid-wise attention processing
        img, map = self.gwa(x)
        fused_features = self.gwa_f(img, map)
        
        gnn_features = None
        if self.use_gnn:
            # Construct graph from fused features
            node_features, edge_index, batch = self.graph_constructor.build_grid_graph(fused_features)
            
            # Apply Attention GNN
            gnn_output = self.attention_gnn(node_features, edge_index, batch)
            
            # Pool GNN output back to image-level features
            pooled_gnn_features = self.graph_pooling(gnn_output, batch, self.graph_constructor.grid_size)
            
            # Project to match backbone feature dimension
            gnn_features = self.gnn_projection(pooled_gnn_features)
        
        # Process through backbone with optional GNN features
        backbone_output = self.backbone(fused_features, gnn_features)
        
        # Visual transformer attention and classification
        emotions = self.vta(backbone_output)
        
        return emotions


# CrossEntropyLoss with Label Smoothing is added in pytorch 1.7.0+,change it will be ok if your version >1.7
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            if len(true_dist.shape) == 1:
                true_dist.scatter_(1, target.data.unsqueeze(0), self.confidence)
            else:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


