# models/layers/SAST/wavelet_freq.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import Tuple, Optional

class DWT2D(nn.Module):
    """2D离散小波变换模块"""
    def __init__(self, wave='db4', level=1):
        super().__init__()
        self.wave = wave
        self.level = level
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: (LL, (LH, HL, HH))
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 转到CPU进行小波变换
        x_cpu = x.detach().cpu().numpy()
        coeffs_list = []
        
        for b in range(B):
            for c in range(C):
                coeffs = pywt.dwt2(x_cpu[b, c], self.wave)
                coeffs_list.append(coeffs)
        
        # 重组并转回GPU
        LL = torch.stack([torch.from_numpy(c[0]).float() for c in coeffs_list])
        LL = LL.view(B, C, *LL.shape[-2:]).to(device)
        
        LH = torch.stack([torch.from_numpy(c[1][0]).float() for c in coeffs_list])
        LH = LH.view(B, C, *LH.shape[-2:]).to(device)
        
        HL = torch.stack([torch.from_numpy(c[1][1]).float() for c in coeffs_list])
        HL = HL.view(B, C, *HL.shape[-2:]).to(device)
        
        HH = torch.stack([torch.from_numpy(c[1][2]).float() for c in coeffs_list])
        HH = HH.view(B, C, *HH.shape[-2:]).to(device)
        
        return LL, (LH, HL, HH)

class IDWT2D(nn.Module):
    """2D逆离散小波变换模块"""
    def __init__(self, wave='db4'):
        super().__init__()
        self.wave = wave
        
    def forward(self, LL, high_freq):
        """重构信号"""
        LH, HL, HH = high_freq
        B, C = LL.shape[:2]
        device = LL.device
        
        # 转到CPU进行逆变换
        LL_cpu = LL.detach().cpu().numpy()
        LH_cpu = LH.detach().cpu().numpy()
        HL_cpu = HL.detach().cpu().numpy()
        HH_cpu = HH.detach().cpu().numpy()
        
        recon_list = []
        for b in range(B):
            for c in range(C):
                coeffs = (LL_cpu[b, c], (LH_cpu[b, c], HL_cpu[b, c], HH_cpu[b, c]))
                recon = pywt.idwt2(coeffs, self.wave)
                recon_list.append(torch.from_numpy(recon).float())
        
        x = torch.stack(recon_list)
        x = x.view(B, C, *x.shape[-2:]).to(device)
        return x

class AdaptiveWaveletFreqModule(nn.Module):
    """自适应小波-频域增强模块"""
    def __init__(self, dim, num_heads=8, wave='db4', adaptive=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        
        # 小波变换
        self.dwt = DWT2D(wave=wave)
        self.idwt = IDWT2D(wave=wave)
        
        # 频域处理网络
        self.freq_processor = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1),
            nn.GroupNorm(num_heads, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 4, 1)
        )
        
        # 自适应门控
        if adaptive:
            self.density_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 4, 4, 1),
                nn.Sigmoid()
            )
        
        # 特征融合
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        
    def compute_event_density(self, x):
        """计算事件密度"""
        # 使用非零元素比例作为密度指标
        mask = (x != 0).float()
        density = F.adaptive_avg_pool2d(mask, 1)
        return density
    
    def forward(self, x, return_freq_features=False):
        """
        x: (B, C, H, W) 输入特征
        """
        B, C, H, W = x.shape
        
        # 小波分解
        LL, (LH, HL, HH) = self.dwt(x)
        
        # 组合频域特征
        freq_features = torch.cat([LL, LH, HL, HH], dim=1)
        
        # 频域处理
        freq_enhanced = self.freq_processor(freq_features)
        
        # 自适应门控
        if self.adaptive:
            density = self.compute_event_density(x)
            gates = self.density_gate(density)  # (B, 4, 1, 1)
            
            # 应用门控
            LL_g, LH_g, HL_g, HH_g = freq_enhanced.chunk(4, dim=1)
            LL_g = LL_g * gates[:, 0:1]
            LH_g = LH_g * gates[:, 1:2]
            HL_g = HL_g * gates[:, 2:3]
            HH_g = HH_g * gates[:, 3:4]
        else:
            LL_g, LH_g, HL_g, HH_g = freq_enhanced.chunk(4, dim=1)
        
        # 小波重构
        recon = self.idwt(LL_g, (LH_g, HL_g, HH_g))
        
        # 确保维度匹配
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # 特征融合
        fused = self.fusion(torch.cat([x, recon], dim=1))
        
        if return_freq_features:
            return fused, freq_features
        return fused

class MultiScaleFreqAttention(nn.Module):
    """多尺度频域注意力模块"""
    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 多尺度FFT
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim // reduction, 1),
                nn.ReLU(),
                nn.Conv2d(dim // reduction, dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Conv2d(dim * num_scales, dim, 1)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        
        multi_scale_features = []
        
        for scale_idx in range(self.num_scales):
            # 下采样到不同尺度
            scale_factor = 2 ** scale_idx
            if scale_factor > 1:
                x_scaled = F.avg_pool2d(x, scale_factor)
            else:
                x_scaled = x
            
            # FFT
            x_fft = torch.fft.rfft2(x_scaled, norm='ortho')
            x_fft_real = x_fft.real
            x_fft_imag = x_fft.imag
            
            # 处理频域特征
            x_freq = torch.cat([x_fft_real, x_fft_imag], dim=1)
            attention = self.scale_processors[scale_idx](x_freq)
            
            # 应用注意力
            x_fft_weighted = x_fft * attention.unsqueeze(-1)
            
            # IFFT
            x_spatial = torch.fft.irfft2(x_fft_weighted, s=(x_scaled.shape[-2], x_scaled.shape[-1]), norm='ortho')
            
            # 上采样回原始尺寸
            if scale_factor > 1:
                x_spatial = F.interpolate(x_spatial, size=(H, W), mode='bilinear', align_corners=False)
            
            multi_scale_features.append(x_spatial)
        
        # 融合多尺度特征
        fused = self.scale_fusion(torch.cat(multi_scale_features, dim=1))
        return fused

class EfficientWaveletTransform(nn.Module):
    """高效的可学习小波变换"""
    def __init__(self, dim, num_levels=2):
        super().__init__()
        self.num_levels = num_levels
        
        # 可学习的小波滤波器
        self.low_pass = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.high_pass = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        # 初始化为Haar小波
        with torch.no_grad():
            # Low-pass filter
            self.low_pass.weight.fill_(1/9)
            # High-pass filter
            self.high_pass.weight.fill_(0)
            self.high_pass.weight[:, :, 1, 1] = 1
            self.high_pass.weight[:, :, 0, 0] = -0.25
            self.high_pass.weight[:, :, 0, 2] = -0.25
            self.high_pass.weight[:, :, 2, 0] = -0.25
            self.high_pass.weight[:, :, 2, 2] = -0.25
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: 多尺度小波系数
        """
        coeffs = []
        
        for level in range(self.num_levels):
            # 低频和高频分解
            low = F.avg_pool2d(self.low_pass(x), 2)
            high = F.avg_pool2d(self.high_pass(x), 2)
            
            coeffs.append((low, high))
            x = low  # 下一层级使用低频分量
        
        return coeffs