import math
import logging
from functools import partial
from collections import OrderedDict

import jittor as jt
import jittor.nn as nn
from copy import deepcopy

jt.flags.use_cuda = 1  # 启用CUDA，如果可用

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.bias = nn.Parameter(jt.zeros(normalized_shape))

    def execute(self, x):
        return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化"""

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with jt.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(l, u)
        tensor = tensor * std + mean
        return tensor


def lecun_normal_(tensor):
    """LeCun正态分布初始化"""
    fan_in = tensor.size(1) * tensor[0][0].numel() if len(tensor.shape) > 2 else tensor.size(1)
    std = math.sqrt(1. / fan_in)
    with jt.no_grad():
        tensor.normal_(0, std)
        return tensor


# ===============================
# 基础模块：DropPath, Mlp, PatchEmbed
# ===============================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(0, 2, 1)
        return x



class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = jt.zeros((dim, dim))
        self.n_matrix = 0
        self.cur_matrix = jt.zeros((dim, dim))
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.constant_(self.lora_B_k[t].weight, 0)
            nn.init.constant_(self.lora_B_v[t].weight, 0)

    def init_param_ada(self, t, r):
        device = self.qkv.weight.device if hasattr(self.qkv.weight, 'device') else None
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.constant_(self.lora_B_k[t].weight, 0)
        nn.init.constant_(self.lora_B_v[t].weight, 0)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def execute(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if get_feat:
            x_detach = x.detach()
            matrix_update = jt.bmm(x_detach.permute(0, 2, 1), x_detach).sum(dim=0)
            self.matrix = (self.matrix * self.n_matrix + matrix_update) / (self.n_matrix + x.shape[0] * x.shape[1])
            self.n_matrix += x.shape[0] * x.shape[1]

        if get_cur_feat:
            x_detach = x.detach()
            cur_matrix_update = jt.bmm(x_detach.permute(0, 2, 1), x_detach).sum(dim=0)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + cur_matrix_update) / (
                        self.n_cur_matrix + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0] * x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            # Jittor版本的矩阵乘法
            weight_k = jt.stack([jt.matmul(self.lora_B_k[t].weight, self.lora_A_k[t].weight)
                                 for t in range(int(task) + 1)], dim=0).sum(dim=0)
            weight_v = jt.stack([jt.matmul(self.lora_B_v[t].weight, self.lora_A_v[t].weight)
                                 for t in range(int(task) + 1)], dim=0).sum(dim=0)

            lora_k = nn.matmul_transpose(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2,
                                                                                                                 1, 3)
            lora_v = nn.matmul_transpose(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2,
                                                                                                                 1, 3)

            k = k + lora_k
            v = v + lora_v

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            # Jittor目前不支持直接的register_hook，可以使用其他方式记录梯度

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = jt.matmul(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = jt.matmul(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v

    def get_pre_matrix(self, task):
        with jt.no_grad():
            weight_k = jt.stack([jt.matmul(self.lora_B_k[t].weight, self.lora_A_k[t].weight)
                               for t in range(task)], dim=0).sum(dim=0)
            weight_v = jt.stack([jt.matmul(self.lora_B_v[t].weight, self.lora_A_v[t].weight)
                               for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * jt.ones(dim))

    def execute(self, x):
        return x * self.gamma  # Jittor通常不鼓励inplace操作


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   n_tasks=n_tasks, r=r)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        attn_out = self.attn(self.norm1(x), task, register_hook=register_hook,
                            get_feat=get_feat, get_cur_feat=get_cur_feat)
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x




class VisionTransformer(nn.Module):

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(jt.zeros((1, 1, embed_dim)))
        self.cls_token_grow = nn.Parameter(jt.zeros((1, 5000, embed_dim)))
        self.pos_embed = nn.Parameter(jt.zeros((1, num_patches + self.num_tokens, embed_dim)))
        self.pos_embed_grow = nn.Parameter(jt.zeros((1, num_patches + 1000, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, init_values=init_values, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                     act_layer=act_layer, n_tasks=n_tasks, r=rank)
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = embed_dim

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.

        # 位置编码初始化
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_grow, std=.02)

        # CLS token初始化
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_token_grow, std=1e-6)

        # 初始化所有线性层
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'head' in name:
                    nn.init.zeros_(m.weight)
                    nn.init.constant_(m.bias, head_bias)
                elif 'qkv' in name:
                    # 分开初始化Q, K, V
                    fan_in = m.weight.shape[1]
                    val = math.sqrt(6. / fan_in)
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def execute_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = jt.concat([cls_token, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # 逐个处理block，传递task参数
        for block in self.blocks:
            x = block(x, task=-1)  # 默认不使用LoRA

        x = self.norm(x)
        return x

    def execute_features_grow(self, x, class_num):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = jt.concat([cls_token, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # 添加额外的class tokens
        cls_token_grow = self.cls_token_grow[:, :class_num * 2, :].expand(x.shape[0], -1, -1)
        x = jt.concat([cls_token_grow, x], dim=1)

        for block in self.blocks:
            x = block(x, task=-1)

        x = self.norm(x)
        return x

    def execute_head(self, x, pre_logits=False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def execute(self, x, task_id=-1, grow_flag=False, numcls=0, get_cur_feat=False):
        if not grow_flag:
            x = self.execute_features(x)
        else:
            x = self.execute_features_grow(x, numcls)

        # 如果需要获取当前特征
        if get_cur_feat:
            return {
                'fmaps': [x],
                'features': x
            }

        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        return {
            'fmaps': [x],
            'features': x
        }

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


# ===============================
# 辅助函数和模型工厂
# ===============================
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) 模型"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16) 模型"""
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """ViT-Large (ViT-L/16) 模型"""
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


# ===============================
# 模型测试示例
# ===============================
if __name__ == "__main__":
    # 测试ViT-Base模型
    model = vit_base_patch16_224(num_classes=1000, n_tasks=10, rank=64)

    # 创建随机输入
    x = jt.randn((2, 3, 224, 224))

    # 前向传播
    output = model(x, task_id=0)
    print(f"Output shape: {output['features'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


