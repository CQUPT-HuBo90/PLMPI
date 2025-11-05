from functools import partial
import torch
import numpy as np
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from module_mamba import VisionMamba, PatchEmbed, Block, Mamba 
import v3 
from MiDaS_master.midas.model_loader import default_models, load_model
from SSLSOD_main.prediction_rgbd import RGBD_sal as SaliencyModel
import os
from torchvision.utils import save_image
from timm.models.layers import DropPath

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Attention_Cross(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, q):
        B, N, C = x.shape
        kv = self.to_kv(x).reshape(B, N, 2, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = q.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, heads=num_heads, dim_head=dim//num_heads, dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATCrossLayer(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention_Cross(dim, heads=num_heads, dim_head=dim//num_heads, dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, q_extra):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, q_extra)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, q_extra)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class SequenceAligner(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length
        self.align = nn.AdaptiveAvgPool1d(target_length)
        
    def forward(self, x):
        x = x.transpose(1, 2) 
        x_aligned = self.align(x)  
        return x_aligned.transpose(1, 2) 


class NATBlock(nn.Module):
    def __init__(self, dim, depth, depth_cross, num_heads, 
                 mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     layer_scale=layer_scale)
            for i in range(depth)])

        self.cross_blocks = nn.ModuleList([
            NATCrossLayer(dim=dim,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          drop=drop, attn_drop=attn_drop,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer=norm_layer,
                          layer_scale=layer_scale)
            for i in range(depth_cross)])

        self.channel_compress = nn.Linear(2 * dim, dim)

    def forward(self, x, x_multi):
        x_cross = x.clone()
        for blk in self.blocks:
            x = blk(x)

        for cross_blk in self.cross_blocks:
            x_cross = cross_blk(x_multi, x_cross)

        x = torch.cat([x, x_cross], dim=-1)
        x = self.channel_compress(x)
        
        return x

class NATInjectModule(nn.Module):
    def __init__(self, dim, hid_dim, target_len=197):
        super().__init__()
        self.target_len = target_len
        
        self.aligner = SequenceAligner(target_length=target_len)
        
        self.proj_mamba = nn.Linear(dim, hid_dim)
        
        self.nat_block = NATBlock(
            dim=hid_dim,
            depth=4,          
            depth_cross=2,    
            num_heads=8,       
            mlp_ratio=4.,
            drop=0.1,
            attn_drop=0.1,
            layer_scale=1e-4
        )
        
        self.proj_out = nn.Linear(hid_dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, cnn_feat):
        
        cnn_feat_aligned = self.aligner(cnn_feat)
        
        x_proj = self.proj_mamba(x)
        
        fused = self.nat_block(x_proj, cnn_feat_aligned)
        
        x_h = self.proj_out(fused)
        return x + x_h * self.scale


class DepthGenerator(nn.Module):
    def __init__(self, model_type='dpt_levit_224'):
        super().__init__()
        self.model, self.transform, _, _ = load_model(
            device=device,
            model_path=None,
            model_type=model_type,
            optimize=True
        )
        self.model = self.model.to(memory_format=torch.channels_last).to(device)
        
    def forward(self, x):
        if not hasattr(self, '_device'):
            self._device = x.device
            self.model = self.model.to(self._device)
            if self._device.type == 'cuda':
                self.model = self.model.half()
        
        with torch.no_grad():
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
            
            x_norm = (x / 255.0).cpu().numpy()
            
            input_tensor = torch.stack([
                torch.from_numpy(self.transform({"image": img})["image"]) 
                for img in x_norm
            ]).to(self._device)
            
            if self._device.type == 'cuda':
                input_tensor = input_tensor.half()
            prediction = self.model(input_tensor)
            return F.interpolate(
                prediction.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).float()


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # backbone
        self.mamba = VisionMamba(img_size=224, patch_size=16, embed_dim=192, depth=3, rms_norm=True,
                                 residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean',
                                 if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
                                 if_cls_token=True, if_divide_out=True, use_middle_cls_token=True).to(device)
        self.blocks = nn.ModuleList([
                                 Block(dim=192, mixer_cls=partial(Mamba, d_state=28, layer_idx=None, bimamba_type="v2",
                                 if_divide_out=False, init_layer_scale=None), norm_cls=nn.LayerNorm) for _ in range(3)]).to(device)
        self.load_pretrained_with_fourier_interpolation("/home/cty/science/my_loda/vim_t_midclstok_ft_78p3acc.pth")
        
        # patch embeding
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=512).to(device)
        
        # resnet
        self.a2_net = v3.A_2_net(code_length=12, num_classes=1000, att_size=3, feat_size=2048, device=device, pretrained=True)
        self.a2_net.eval()
        # resnet layer4
        self.refine_local1 = v3.A_2_net_refine(is_local=True, inplanes=256, planes=64,pretrained=False).to(device)
        self.refine_local2 = v3.A_2_net_refine(is_local=True, inplanes=512, planes=64,pretrained=False).to(device)
        self.refine_local3 = v3.A_2_net_refine(is_local=True, inplanes=1024, planes=64,pretrained=False).to(device)
        
        # 显著性处理
        self.depth_generator = DepthGenerator(model_type='dpt_levit_224').to(device).eval()
        self.saliency_model = SaliencyModel().to(device).eval().half()
        
        self.inject_module = NATInjectModule(dim=192, hid_dim=512, target_len=197).to(device)
        
        # adjust
        self.local_f_adjust = nn.Conv2d(256, 512, 1).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.proj = nn.Linear(512, 192).to(device)
        # MLP
        self.head = nn.Linear(192, 1)
        self.fz()

    def fz(self):
        for param in self.mamba.parameters():
            param.requires_grad = False
        for param in self.a2_net.parameters():
            param.requires_grad = False
        for param in self.depth_generator.parameters():
            param.requires_grad = False
        for param in self.saliency_model.parameters():
            param.requires_grad = False
        for param in self.refine_local1.parameters():
            param.requires_grad = False
        for param in self.refine_local2.parameters():
            param.requires_grad = False

    def load_pretrained_with_fourier_interpolation(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        original_embed = checkpoint["model"]["pos_embed"]
        
        new_embed = self.fourier_interpolate_pos_embed(
            original_embed, 
            new_num_patches=self.mamba.pos_embed.shape[1]
        )
        
        checkpoint["model"]["pos_embed"] = new_embed
        self.mamba.load_state_dict(checkpoint["model"], strict=False)

    def fourier_interpolate_pos_embed(self, pos_embed, new_num_patches):
        pos_embed = pos_embed.squeeze(0)
        old_num_patches, dim = pos_embed.shape
        
        pos_embed_f = torch.fft.rfft(pos_embed, dim=0)
        pos_embed_f_real = pos_embed_f.real.unsqueeze(0).unsqueeze(0)
        
        new_length = new_num_patches // 2 + 1
        new_pos_embed_f_real = F.interpolate(
            pos_embed_f_real,
            size=(new_length, dim),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        new_pos_embed_f = torch.complex(
            new_pos_embed_f_real,
            torch.zeros_like(new_pos_embed_f_real)
        )
        
        new_pos_embed = torch.fft.irfft(new_pos_embed_f, n=new_num_patches, dim=0)
        return new_pos_embed.unsqueeze(0)

    def generate_enhanced_input(self, x):
        with torch.no_grad():
            B, C, H, W = x.shape
            
            device = x.device
            dtype = torch.half if device.type == 'cuda' else torch.float32
            
            depth_maps = self.depth_generator(x)
            depth_maps = depth_maps.repeat(1, 3, 1, 1).to(dtype=dtype)
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                saliency = self.saliency_model(x.to(dtype), depth_maps)
                saliency = saliency[0] if isinstance(saliency, tuple) else saliency
            
            mask = F.interpolate(
                saliency.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).sigmoid()
            
            if C == 3:
                mask = mask.expand(-1, 3, -1, -1)
            
            return x * mask.to(device=x.device, dtype=x.dtype)

    def resnet_forward(self, x):
        x = self.a2_net.backbone.conv1(x)
        x = self.a2_net.backbone.bn1(x)
        x = self.a2_net.backbone.relu(x)
        x = self.a2_net.backbone.maxpool(x)

        # layer1
        x = self.a2_net.backbone.layer1(x)
        att1 = self.a2_net.attention1(x)
        batch_size, channels, h, w = x.shape
        out_local1 = att1.unsqueeze(2) * x.unsqueeze(1)
        out_local1 = out_local1.view(-1, 256, 56, 56)
        local_f1, _ = self.refine_local1(out_local1)
        local_f1 = self.local_f_adjust(local_f1)
        local_f1 = local_f1.view(batch_size, 3, 512, 28*28).permute(0,1,3,2).reshape(batch_size, -1, 512)

        # layer2
        x = self.a2_net.backbone.layer2(x)
        att2 = self.a2_net.attention2(x)
        out_local2 = att2.unsqueeze(2) * x.unsqueeze(1)
        out_local2 = out_local2.view(-1, 512, 28, 28)
        local_f2, _ = self.refine_local2(out_local2)
        local_f2 = self.local_f_adjust(local_f2)
        local_f2 = local_f2.view(batch_size, 3, 512, 14*14).permute(0,1,3,2).reshape(batch_size, -1, 512)

        # layer3
        x = self.a2_net.backbone.layer3(x)
        att_map = self.a2_net.attention3(x)
        batch_size, channels, h, w = x.shape
        out_local = (att_map.unsqueeze(2) * x.unsqueeze(1)).view(batch_size * 3, channels, h, w)
        local_f3, _ = self.refine_local3(out_local)
        local_f3 = self.local_f_adjust(local_f3)
        local_f3 = local_f3.view(batch_size, 3, 512, 7*7).permute(0,1,3,2).reshape(batch_size, -1, 512)
            
        return  local_f1, local_f2, local_f3

    def inject_forward(self, x, local_f1, local_f2, local_f3):
        x = self.patch_embed(x)
        x = self.proj(x)        
        cls_token = self.mamba.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Block 1: Inject local_f1
        x = self.inject_module(x, local_f1) 
        x, _ = self.blocks[0](x)  # Mamba Block 1

        # Block 2: Inject local_f2
        x = self.inject_module(x, local_f2)
        x, _ = self.blocks[1](x)  # Mamba Block 2

        # Block 3: Inject local_f3
        x = self.inject_module(x, local_f3)
        x, _ = self.blocks[2](x)  # Mamba Block 3

        # Final normalization
        x = self.mamba.norm_f(x)
        return x[:, 0, :]  #(CLS token)

    def forward(self, x):
        enhanced_x = self.generate_enhanced_input(x)
        y1, y2, y3 = self.resnet_forward(enhanced_x)
        x = self.inject_forward(x, y1, y2, y3)
        x = self.dropout(x)
        x = self.head(x)
        return x