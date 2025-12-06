

import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")




import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import _cfg, Attention, DropPath, Mlp, partial, LayerScale, _cfg, Block
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.registry import register_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from einops import rearrange





import timm
print(timm.__version__)





import einops
print(einops.__version__)





_logger = logging.getLogger(__name__)




import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Attention
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torchvision.transforms as transforms


_logger = logging.getLogger(__name__)


import torch
import numpy as np

# Reshape pos_embed to match the new input size
def resize_pos_embed(pos_embed_checkpoint, new_num_patches, num_extra_tokens):
    """
    Reshapes the positional embeddings to match the new input size.
    Args:
        pos_embed_checkpoint: Original positional embeddings from the checkpoint.
        new_num_patches: Number of patches in the new input size.
        num_extra_tokens: Number of extra tokens (e.g., [CLS] tokens).
    Returns:
        Resized positional embeddings.
    """
    # Separate class token and spatial tokens
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    spatial_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    
    # Calculate the old and new grid sizes
    old_grid_size = int(np.sqrt(spatial_tokens.shape[1]))
    new_grid_size = int(np.sqrt(new_num_patches))
    
    # Reshape spatial tokens to 2D grid
    spatial_tokens = spatial_tokens.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    
    # Resize using interpolation
    spatial_tokens = torch.nn.functional.interpolate(
        spatial_tokens, size=(new_grid_size, new_grid_size), mode='bilinear', align_corners=False
    )
    
    # Flatten back to 1D
    spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)
    
    # Concatenate extra tokens and resized spatial tokens
    new_pos_embed = torch.cat((extra_tokens, spatial_tokens), dim=1)
    return new_pos_embed







def token2feature(tokens):
    B, L, D = tokens.shape
    H = W = int(L ** 0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B, C, W, H = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

class Fovea(nn.Module):
    def __init__(self, smooth=False):
        super(Fovea, self).__init__()
        self.smooth = smooth
        if smooth:
            self.smooth_param = nn.Parameter(torch.tensor(10.0))  # Default value is 10

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        if self.smooth:
            mask = F.softmax(x * self.smooth_param, dim=-1)
        else:
            mask = F.softmax(x, dim=-1)
        output = mask * x
        output = output.view(b, c, h, w)
        return output

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True,
                 attention=True, num_heads=4, qkv_bias=False, attn_drop=0., drop=0.):
        super(Adapter, self).__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
        self.attn = Attention(D_hidden_features, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop) if attention else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.attn(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class TMAdapter(nn.Module):
    def __init__(self, D_features, num_frames, ratio=0.25):
        super(TMAdapter, self).__init__()
        self.num_frames = num_frames
        self.T_Adapter = Adapter(D_features, mlp_ratio=ratio, skip_connect=False, attention=True)
        self.norm = nn.LayerNorm(D_features)
        self.S_Adapter = Adapter(D_features, mlp_ratio=ratio, skip_connect=False, attention=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bt, n, d = x.shape
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = self.T_Adapter(xt)
        x = rearrange(xt, '(b n) t d -> (b t) n d', n=n)
        x = self.S_Adapter(self.norm(x))
        return x



# In[ ]:


class Prompt_block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False, num_frames=1, ratio=0.25):
        super(Prompt_block, self).__init__()

        self.num_frames = num_frames

        self.conv0_0 = nn.Conv2d(
            in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(
            in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(
            in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        self.TMA = TMAdapter(inplanes, num_frames, ratio=ratio)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()

        x2 = x0.view(B, C // 2, -1).transpose(1, 2).contiguous()
        x2 = self.TMA(x2)

        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        x0 = self.conv1x1(x0)
        return x0, x2


# In[11]:


class Prompt_PatchEmbed(nn.Module):
    """ Convert 2D image to Patch Embedding """

    def __init__(self, img_size=14, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BxCxWxH -> BxNxC
        x = self.norm(x)
        return x


# In[12]:


import torch
import torch.nn as nn
from einops import rearrange




class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=112,
            patch_size=16,
            in_chans=3,
            in_chans_l=128,
            num_frames=6,
            num_classes=5,  # تعداد خروجی‌های رگرسیون
            prompt_type='shallow',
            global_pool='token',
            hidden_dim=8,
            embed_dim=768,
            depth=12,
            adapter_scale=0.25,
            head_dropout_ratio=0.5,
            num_tadapter=1,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.5,
            attn_drop_rate=0.05,
            drop_path_rate=0.05,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )

        num_patches = self.patch_embed.num_patches

        '''patch_embed_prompt'''
        self.patch_embed_prompt = Prompt_PatchEmbed(
            img_size=14, patch_size=patch_size, in_chans=in_chans_l, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * .02, requires_grad=False)
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.prompt_type = prompt_type
        if self.prompt_type in ['shallow', 'deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(
                    inplanes=embed_dim, hide_channel=hidden_dim, smooth=True, num_frames=num_frames, ratio=adapter_scale))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.ln_post = norm_layer(
            embed_dim) if not use_fc_norm else nn.Identity()
        
        # تغییر Head برای رگرسیون
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temporal_embedding, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def no_weight_decay(self):
        return {'pos_embed', 'temporal_embedding', 'cls_token', 'dist_token'}  

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)
        

    def forward_features(self, x, a):
        print(f"x shape before patch_embed: {x.shape}")
        print(f"a shape before resizing: {a.shape}")
        x = self.patch_embed(x)
        print(f"x shape after patch_embed: {x.shape}")

        resize_transform = transforms.Resize((14, 14))
        a = resize_transform(a) 
        print(f"a shape after resizing: {a.shape}")
        a = self.patch_embed_prompt(a)

        T = 6
        a = a.unsqueeze(1)  # اضافه کردن بعد زمان (تعداد فریم‌ها)
        a = a.repeat(1, T, 1, 1, 1)  # تکرار لندمارک‌ها برای تعداد فریم‌ها
        a = rearrange(a, 'b t c h w -> (b t) c h w') 
        
        print(f"a shape after pormptembd: {a.shape}")
        '''input prompt: by adding to rgb tokens'''
        if self.prompt_type in ['shallow', 'deep']:
            print("Shape before rearrenge:", a.shape) 
            print("Shape before squeeze:", a.shape) 
            a = a.squeeze(1)
            print("Shape after squeeze:", a.shape) 
            x_feat = token2feature(self.prompt_norms[0](x))
            a_feat = token2feature(self.prompt_norms[0](a))
            print(f"x_feat shape: {x_feat.shape}")
            print(f"a_feat shape: {a_feat.shape}")

            if x_feat.shape[2:] != a_feat.shape[2:]:
                print(f"Resizing a_feat from {a_feat.shape[2:]} to {x_feat.shape[2:]}")
                a_feat = torch.nn.functional.interpolate(a_feat, size=(x_feat.shape[2], x_feat.shape[3]), mode='bilinear', align_corners=False)
            print(f"x_feat after resizing shape: {x_feat.shape}")
            print(f"a_feat after resizing shape: {a_feat.shape}")
            x_feat = torch.cat([x_feat, a_feat], dim=1)
            print(f"x_feat after cat shape: {x_feat.shape}")
            del a_feat
            torch.cuda.empty_cache()
            x_feat, x1 = self.prompt_blocks[0](x_feat)
            x_feat = feature2token(x_feat)
            a = x_feat
            x = x + x1 + x_feat
        else:
            x += a

        x = self._pos_embed(x)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        print(f'x before norm_pre {x.shape}')
        x = self.norm_pre(x)
        print(f'x after norm_pre{x.shape}')
        
        print(f"x shape before blocks: {x.shape}")
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            print(f"Iteration {i}, x shape: {x.shape}")

            if i >= 1:
                if self.prompt_type in ['deep']:
                    x_ori = x
                    print(f"Before prompt_norms[{i - 1}], x shape: {x.shape}")
                    x = self.prompt_norms[i - 1](x)  # اعمال نرمال‌سازی
                    print(f"After prompt_norms[{i - 1}], x shape: {x.shape}")
                
                    x_feat = token2feature(x[:, 1:])
                    print(f"x_feat shape after token2feature: {x_feat.shape}")
                
                    a_feat = token2feature(self.prompt_norms[0](a))
                    print(f"a_feat shape after token2feature: {a_feat.shape}")
                
                    x_feat = torch.cat([x_feat, a_feat], dim=1)
                    print(f"x_feat shape after cat: {x_feat.shape}")
                
                    x_feat, x1 = self.prompt_blocks[i](x_feat)
                    print(f"x_feat shape after prompt_blocks[{i}]: {x_feat.shape}, x1 shape: {x1.shape}")
                
                    x_feat = feature2token(x_feat)
                    print(f"x_feat shape after feature2token: {x_feat.shape}")
                
                    x = torch.cat([x_ori[:, 0:1], x_ori[:, 1:] + x1 + x_feat], dim=1)
                    print(f"x shape after concatenation: {x.shape}")

            x = self.blocks[i](x)
            print(f"After block {i}, x shape: {x.shape}")
            # if i == 9:
            #     x_middle = x
        print(f"x shape after all blocks: {x.shape}")
        
        x = self.ln_post(x)
        print(f"x shape after ln_post: {x.shape}")
        return x


    # def forward_head(self, x, pre_logits: bool = False):
    #     if self.global_pool:
    #         x = x[:, self.num_prefix_tokens:].mean(
    #             dim=1) if self.global_pool == 'avg' else x[:, 0]
    #     x = self.fc_norm(x)
    #     return x
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'token':  # استفاده فقط از CLS token
            x = x[:, 0]  # انتخاب CLS token (اولین توکن)
        elif self.global_pool == 'avg':  # استفاده از میانگین سایر توکن‌ها
            x = x[:, self.num_prefix_tokens:].mean(dim=1)  # میانگین سایر توکن‌ها
        else:
            raise ValueError(f"Unsupported global_pool type: {self.global_pool}")  # مدیریت خطا
        x = self.fc_norm(x)  # نرمال‌سازی (اختیاری)
        return x


    def forward(self, x, a):
        B, C, H, W = x.shape  # حذف T از shape
        T = 6  # مقداردهی استاتیک به تعداد فریم‌ها
        # assert T == self.num_frames, f'Input video must have {self.num_frames} frames, but got {T} frames'
        
        # تغییر شکل ورودی برای پردازش
        x = x.unsqueeze(2)  # اضافه کردن بعد t
        x = x.repeat(1, 1, T, 1, 1)  
        x = rearrange(x, 'b c t h w -> (b t) c h w', t=T)
        x = self.forward_features(x, a)
        print(f'before forward_head {x.shape}')
        x = self.forward_head(x)
        print(f'after forward_head {x.shape}')
        print(f'before rearrange {x.shape}')
        x = rearrange(x, '(b t) c -> b c t', b=B, t=T)
        # x = x.unsqueeze(-1).unsqueeze(-1) 
        x = x.mean(dim=-1) 
        print(f'after rearrange {x.shape}')
        print(f'before self.head {x.shape}')
        score = self.head(x)
        print(f'after self.head {score.shape}')
        return score, None


def _create_vision_transformer(pretrained=False, num_classes=5, **kwargs):
    model = VisionTransformer(**kwargs)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "model_state_dict" in checkpoint.keys():
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint["model"]

        if 'pos_embed' in state_dict:
            num_extra_tokens = model.num_prefix_tokens  
            
            new_num_patches = model.patch_embed.num_patches
            
            print(f"Reshaping pos_embed from {state_dict['pos_embed'].shape} to match {new_num_patches + num_extra_tokens} tokens.")
            state_dict['pos_embed'] = resize_pos_embed(
                state_dict['pos_embed'], new_num_patches, num_extra_tokens
            )

        if 'patch_embed_prompt.proj.weight' in state_dict:
            pretrained_weight = state_dict['patch_embed_prompt.proj.weight']
            if pretrained_weight.shape[1] != 3:  
                print(f"Reshaping patch_embed_prompt.proj.weight from {pretrained_weight.shape} to match input channels.")
                state_dict['patch_embed_prompt.proj.weight'] = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] != num_classes:
            print(f"Reshaping head.weight from {state_dict['head.weight'].shape} to {[num_classes, state_dict['head.weight'].shape[1]]}")
            state_dict['head.weight'] = state_dict['head.weight'][:num_classes, :]
        if 'head.bias' in state_dict and state_dict['head.bias'].shape[0] != num_classes:
            print(f"Reshaping head.bias from {state_dict['head.bias'].shape} to {[num_classes]}")
            state_dict['head.bias'] = state_dict['head.bias'][:num_classes]

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False  
        )
        print('Load pretrained model from: ' + pretrained)
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    model.head = nn.Linear(in_features=kwargs.get('embed_dim', 768), out_features=num_classes)

    return model





@register_model
def s2d_base_patch16_224(pretrained=False, pretrained_cfg=None, input_size=112, patch_size=16, num_classes=5, **kwargs):
    """ ViT-Base model (ViT-B/16) with custom input size and patch size. """
    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes, 
        **kwargs
    )
    
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
  
    model.default_cfg = _cfg(
        input_size=(3, input_size, input_size),
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
    )
    
    return model




