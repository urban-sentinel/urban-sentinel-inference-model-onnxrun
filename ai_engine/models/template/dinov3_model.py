import os
import sys
import torch
import torch.nn as nn
import timm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.dinov3_cfg import ModelConfig as Config

class TemporalAttentionBlock(nn.Module):

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True 
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        linear_output = self.linear2(self.dropout(self.gelu(self.linear1(x))))
        x = x + self.dropout(linear_output)
        x = self.norm2(x)
        
        return x

class UrbanSentinelModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(
            Config.MODEL_NAME, 
            pretrained=Config.PRETRAINED, 
            num_classes=0 
        )

        with torch.no_grad():
            dummy_tensor = torch.zeros(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            spatial_features_dim = self.backbone(dummy_tensor).shape[1] 
            
        attention_dim = 384
        self.pre_attention = nn.Linear(spatial_features_dim, attention_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, Config.NUM_FRAMES, attention_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.temporal_attention = TemporalAttentionBlock(
            d_model=attention_dim, 
            nhead=6, 
            dropout=0.1
        )

        self.urban_head = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.GELU(),
            nn.Dropout(p=Config.DROPOUT_RATE / 2),
            nn.Linear(attention_dim // 2, Config.NUM_CLASSES)
        )

    def forward(self, x):
        
        B, C, T, H, W = x.shape
        
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        features_flat = self.backbone(x_flat) 
        
        features = features_flat.view(B, T, -1)
        features = self.pre_attention(features) 
        
        features = features + self.pos_embed
        
        attended_features = self.temporal_attention(features)
        
        video_feature = attended_features.mean(dim=1) 
        
        return self.urban_head(video_feature)

if __name__ == "__main__":
    modelo = UrbanSentinelModel()
    video_falso = torch.randn(Config.BATCH_SIZE, 3, Config.NUM_FRAMES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    salida = modelo(video_falso)
    print("Arquitectura Compilada: DINOv3 + ATENCIÓN TEMPORAL.")
    print(f"Entrada: {video_falso.shape} -> Salida: {salida.shape}")