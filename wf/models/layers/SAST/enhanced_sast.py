# models/layers/SAST/enhanced_sast.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .SAST import SAST_block
from .wavelet_freq import AdaptiveWaveletFreqModule, MultiScaleFreqAttention

class EnhancedSASTBlock(nn.Module):
    """增强的SAST块，融合频域特征"""
    def __init__(
        self,
        dim: int,
        attention_cfg,
        first_block: bool = False,
        use_wavelet: bool = True,
        use_freq_attention: bool = True
    ):
        super().__init__()
        
        # 原始SAST块
        self.sast_block = SAST_block(dim, attention_cfg, first_block)
        
        # 小波-频域增强
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.wavelet_module = AdaptiveWaveletFreqModule(
                dim=dim,
                num_heads=attention_cfg.get('num_heads', 8),
                adaptive=True
            )
        
        # 多尺度频域注意力
        self.use_freq_attention = use_freq_attention
        if use_freq_attention:
            self.freq_attention = MultiScaleFreqAttention(
                dim=dim,
                num_scales=3,
                reduction=4
            )
        
        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x, pos_emb, r, index_list):
        """
        x: (B, H, W, C) 输入特征
        pos_emb: 位置编码
        r: 稀疏率
        index_list: 索引列表
        """
        # 原始SAST处理
        x_sast, p_loss, index_list = self.sast_block(x, pos_emb, r, index_list)
        
        # 转换为channel-first格式用于频域处理
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        features = [x_sast]
        
        # 小波-频域增强
        if self.use_wavelet:
            x_wavelet = self.wavelet_module(x_cf)
            x_wavelet = x_wavelet.permute(0, 2, 3, 1).contiguous()  # 转回(B, H, W, C)
            features.append(x_wavelet)
        
        # 多尺度频域注意力
        if self.use_freq_attention:
            x_freq = self.freq_attention(x_cf)
            x_freq = x_freq.permute(0, 2, 3, 1).contiguous()  # 转回(B, H, W, C)
            features.append(x_freq)
        
        # 自适应特征融合
        if len(features) > 1:
            weights = torch.softmax(self.fusion_weights[:len(features)], dim=0)
            x_fused = sum(w * f for w, f in zip(weights, features))
        else:
            x_fused = features[0]
        
        return x_fused, p_loss, index_list