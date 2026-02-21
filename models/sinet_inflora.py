import logging
import math
import jittor as jt
import jittor.nn as nn
from jittor.attention import MultiheadAttention
from copy import deepcopy

# ===============================
#   Attention with LoRA (InfLoRA核心)
# ===============================
class Attention_LoRA(nn.Module):
    """
    InfLoRA的核心注意力模块
    包含原始qkv投影（冻结）+ 每任务的LoRA分支
    """
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 rank=64, n_tasks=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rank = rank
        self.n_tasks = n_tasks

        # ----- 原始预训练权重（冻结）-----
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ----- LoRA参数（每个任务一套）-----
        # InfLoRA核心：A是组合系数（可训练），B是基础子空间（第一个任务后冻结）
        self.lora_A_q = nn.ModuleList()  # 组合系数
        self.lora_B_q = nn.ModuleList()  # 基础子空间
        self.lora_A_k = nn.ModuleList()
        self.lora_B_k = nn.ModuleList()
        self.lora_A_v = nn.ModuleList()
        self.lora_B_v = nn.ModuleList()
        self.lora_A_o = nn.ModuleList()
        self.lora_B_o = nn.ModuleList()

        for _ in range(n_tasks):
            # 初始化LoRA参数：A随机初始化（组合系数），B初始化为0（基础子空间）
            # 符合论文：微调A等价于在B定义的子空间内微调
            A_q = nn.Linear(embed_dim, rank, bias=False)
            B_q = nn.Linear(rank, embed_dim, bias=False)
            jt.init.gauss_(A_q.weight, 0.0, 0.02)  # 随机初始化
            jt.init.constant_(B_q.weight, 0)       # 初始化为0
            self.lora_A_q.append(A_q)
            self.lora_B_q.append(B_q)

            A_k = nn.Linear(embed_dim, rank, bias=False)
            B_k = nn.Linear(rank, embed_dim, bias=False)
            jt.init.gauss_(A_k.weight, 0.0, 0.02)
            jt.init.constant_(B_k.weight, 0)
            self.lora_A_k.append(A_k)
            self.lora_B_k.append(B_k)

            A_v = nn.Linear(embed_dim, rank, bias=False)
            B_v = nn.Linear(rank, embed_dim, bias=False)
            jt.init.gauss_(A_v.weight, 0.0, 0.02)
            jt.init.constant_(B_v.weight, 0)
            self.lora_A_v.append(A_v)
            self.lora_B_v.append(B_v)

            A_o = nn.Linear(embed_dim, rank, bias=False)
            B_o = nn.Linear(rank, embed_dim, bias=False)
            jt.init.gauss_(A_o.weight, 0.0, 0.02)
            jt.init.constant_(B_o.weight, 0)
            self.lora_A_o.append(A_o)
            self.lora_B_o.append(B_o)

        # ----- 用于DualGPM的正交基存储 -----
        self.subspace_basis = None  # 存储当前层的基础子空间（由B构成）
        self.cur_matrix = None      # 存储激活，用于DualGPM计算

    def execute(self, x, task_id):
        """
        前向传播：原始qkv + LoRA增量
        """
        B, N, C = x.shape

        # 1. 原始qkv计算（冻结）
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # 2. LoRA增量计算
        # 注意：lora_A是组合系数，lora_B是基础子空间
        lora_q = self.lora_B_q[task_id](self.lora_A_q[task_id](x))
        lora_k = self.lora_B_k[task_id](self.lora_A_k[task_id](x))
        lora_v = self.lora_B_v[task_id](self.lora_A_v[task_id](x))
        lora_o = self.lora_B_o[task_id](self.lora_A_o[task_id](x))

        # 重塑LoRA增量以匹配注意力维度
        lora_q = lora_q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        lora_k = lora_k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        lora_v = lora_v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. 合并（原始 + LoRA）
        q = q + lora_q
        k = k + lora_k
        v = v + lora_v

        # 4. 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 5. 输出投影 + LoRA输出增量
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) + lora_o
        x = self.proj_drop(x)

        # 6. 收集激活用于DualGPM（取平均池化作为该层的特征表示）
        self.cur_matrix = x.mean(dim=1)  # [B, C]

        return x

    def set_subspace_basis(self, basis):
        """设置正交基（由DualGPM计算得到）"""
        self.subspace_basis = basis
# ===============================
#   Transformer Block 类 (Jittor 版本)
# ===============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0., attn_drop_rate=0., 
                 rank=64, n_tasks=10, block_idx=0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.rank = rank
        self.n_tasks = n_tasks
        self.block_idx = block_idx
        
        # 自注意力层
        self.attn = Attention_LoRA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            rank=rank,
            n_tasks=n_tasks
        )
        # self.attn_drop = nn.Dropout(attn_drop_rate)
        
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate)
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(drop_rate)
        
        # 初始化 LoRA 参数
        # self._init_lora_params()
    
    # def _init_lora_params(self):
    #     """初始化 LoRA 参数"""
    #     # LoRA 参数列表
    #     self.lora_A_k = nn.ModuleList()
    #     self.lora_B_k = nn.ModuleList()
    #     self.lora_A_v = nn.ModuleList()
    #     self.lora_B_v = nn.ModuleList()
        
    #     # 初始化 LoRA 参数
    #     for _ in range(self.n_tasks):
    #         self.lora_A_k.append(nn.Linear(self.embed_dim, self.rank, bias=False))
    #         self.lora_B_k.append(nn.Linear(self.rank, self.embed_dim, bias=False))
    #         self.lora_A_v.append(nn.Linear(self.embed_dim, self.rank, bias=False))
    #         self.lora_B_v.append(nn.Linear(self.rank, self.embed_dim, bias=False))
            
    #         # 初始化 LoRA 参数为 0
    #         jt.init.constant_(self.lora_A_k[-1].weight, 0)
    #         jt.init.constant_(self.lora_B_k[-1].weight, 0)
    #         jt.init.constant_(self.lora_A_v[-1].weight, 0)
    #         jt.init.constant_(self.lora_B_v[-1].weight, 0)
    
    def execute(self, x, task_id=0):
        """前向传播"""
        # 第一层归一化
        x_norm = self.norm1(x)
        
        # 自注意力
        # 注意：Jittor 的 MultiheadAttention 期望输入形状为 (L, N, E)
        # 其中 L 是序列长度，N 是 batch size，E 是 embedding 维度
        # x_norm_transposed = x_norm.transpose(0, 1)  # (N, L, E) -> (L, N, E)
        # attn_output, _ = self.attn(x_norm_transposed, x_norm_transposed, x_norm_transposed)
        # attn_output = attn_output.transpose(0, 1)  # (L, N, E) -> (N, L, E)
        attn_output = self.attn(x_norm, task_id)
                
        # 残差连接
        x = x + self.dropout(attn_output)
        
        # 第二层归一化
        x_norm = self.norm2(x)
        
        # MLP
        mlp_output = self.mlp(x_norm)
        
        # 残差连接
        x = x + self.dropout(mlp_output)
        
        return x
    
    def forward(self, x, task_id=0):
        return self.execute(x, task_id)


# ===============================
#   ViT_lora_co 模型 (Jittor 版本)
# ===============================
class ViT_lora_co(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
        embed_layer=None, norm_layer=None, act_layer=None, block_fn=None, n_tasks=10, rank=64
    ):
        super().__init__()
        
        # 存储参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.n_tasks = n_tasks
        self.rank = rank
        
        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token
        self.cls_token = jt.randn(1, 1, embed_dim)
        
        # Position embedding
        self.pos_embed = jt.randn(1, num_patches + 1, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                rank=rank,
                n_tasks=n_tasks,
                block_idx=i
            )
            for i in range(depth)
        ])
        
        # Norm layer
        if norm_layer is None:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
    
    def execute(self, x, task_id=0, register_blk=-1, get_feat=False, get_cur_feat=False):
        # 检查输入形状
        # print(f"ViT_lora_co input shape: {x.shape}")
        B, C, H, W = x.shape
        
        # 自动检测和修复维度顺序问题
        # 如果通道数很大（如224），但高度/宽度很小（如3），可能维度顺序错了
        if C != self.in_chans:
            # print(f"Warning: Input channels {C} do not match expected {self.in_chans}")
            # print(f"Trying to auto-correct dimension order...")
            
            # 尝试判断维度顺序：如果是 BHWC 格式，转换为 BCHW
            if C == self.img_size and (H == self.img_size or H == 3) and (W == 3 or W == self.img_size):
                # print(f"Detected possible BHWC format (B, H, W, C) = ({B}, {C}, {H}, {W})")
                
                # 如果 H 或 W 是 3，可能是通道维度
                if H == 3:
                    # 可能是 (B, C, 3, W) -> 转换为 (B, 3, C, W) 或 (B, 3, W, C)?
                    # 更可能是 (B, H, W, C) = (B, 3, W, C)，其中 W 是224
                    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    # print(f"Permuted to shape: {x.shape}")
                elif W == 3:
                    # 可能是 (B, C, H, 3) -> 转换为 (B, 3, C, H)?
                    x = x.permute(0, 3, 1, 2)  # (B, C, H, 3) -> (B, 3, C, H)
                    # print(f"Permuted to shape: {x.shape}")
                else:
                    # 如果 C=224 且 H,W=224,3 或 3,224，尝试转置
                    # 检查最后一个维度是否为3
                    if x.shape[-1] == 3:
                        x = x.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
                        # print(f"Permuted (last dim=3) to shape: {x.shape}")
                
                # 更新形状
                B, C, H, W = x.shape
                
            # 如果转换后还是不匹配，尝试其他转换
            if C != self.in_chans:
                # 尝试将通道维度移到第2维
                if x.shape[-1] == self.in_chans:
                    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    # print(f"Permuted (last dim={x.shape[-1]}) to shape: {x.shape}")
                elif x.shape[1] == self.in_chans and len(x.shape) == 4:
                    # 已经正确，不需要转换
                    pass
                else:
                    raise ValueError(f"Cannot auto-correct input shape {x.shape}. Expected channels: {self.in_chans}")
                
                # 再次更新形状
                B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        # print(f"After patch_embed shape: {x.shape}")
        
        # 计算patch的数量
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        num_patches = H_patch * W_patch
        
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        # print(f"After flatten and transpose shape: {x.shape}")
        
        # 添加 class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)
        # print(f"After adding cls_token shape: {x.shape}")
        
        # 添加位置编码
        # 确保位置编码的形状匹配
        if x.shape[1] != self.pos_embed.shape[1]:
            print(f"Warning: Sequence length {x.shape[1]} doesn't match position embedding {self.pos_embed.shape[1]}")
            # 截断或填充位置编码
            if x.shape[1] < self.pos_embed.shape[1]:
                pos_embed = self.pos_embed[:, :x.shape[1], :]
            else:
                # 需要填充位置编码
                pad_len = x.shape[1] - self.pos_embed.shape[1]
                pos_embed = jt.concat([self.pos_embed, jt.zeros(1, pad_len, self.embed_dim)], dim=1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # 通过 Transformer blocks
        prompt_loss = jt.zeros((1,))
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id=task_id)
        
        x = self.norm(x)
        # print(f"Final output shape: {x.shape}")
        
        return x, prompt_loss
    
    def forward(self, x, task_id=0, register_blk=-1, get_feat=False, get_cur_feat=False):
        return self.execute(x, task_id, register_blk, get_feat, get_cur_feat)


# ===============================
#   创建 Vision Transformer 的函数
# ===============================
def _create_vision_transformer(variant, pretrained=False, **kwargs):
    """创建 Vision Transformer 模型"""
    # 从 variant 中解析参数
    if 'base' in variant:
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif 'large' in variant:
        embed_dim = 1024
        depth = 24
        num_heads = 16
    else:
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        num_heads = kwargs.get('num_heads', 12)
    
    # 合并参数
    model_kwargs = {
        'img_size': kwargs.get('img_size', 224),
        'patch_size': kwargs.get('patch_size', 16),
        'in_chans': kwargs.get('in_chans', 3),
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'n_tasks': kwargs.get('n_tasks', 10),
        'rank': kwargs.get('rank', 64),
        'mlp_ratio': kwargs.get('mlp_ratio', 4.0),
        'qkv_bias': kwargs.get('qkv_bias', True),
        'drop_rate': kwargs.get('drop_rate', 0.0),
        'attn_drop_rate': kwargs.get('attn_drop_rate', 0.0),
    }
    
    # 创建模型
    model = ViT_lora_co(**model_kwargs)
    
    # 加载预训练权重 (简化)
    # if pretrained:
    #     print(f"Note: Loading pretrained weights for {variant} is not implemented in this simplified version.")
    
    return model


# ===============================
#   SiNet for InfLoRA (Jittor 版本)
# ===============================
class SiNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 从 args 中提取参数
        total_sessions = args.get("total_sessions", 10)
        rank = args.get("rank", 64)
        embd_dim = args.get("embd_dim", 768)
        init_cls = args.get("init_cls", 10)
        
        # 创建 image_encoder
        model_kwargs = dict(
            patch_size=16, 
            embed_dim=embd_dim, 
            depth=12, 
            num_heads=12, 
            n_tasks=total_sessions, 
            rank=rank
        )
        
        # 创建 Vision Transformer
        self.image_encoder = _create_vision_transformer(
            'vit_base_patch16_224_in21k', 
            pretrained=args.get("use_pretrained", True), 
            **model_kwargs
        )

        # 分类器参数
        self.class_num = init_cls
        self.embd_dim = embd_dim
        self.total_sessions = total_sessions
        
        # 创建分类器池 (每个任务一个分类器)
        # self.classifier_pool = nn.ModuleList([
        #     self._create_classifier(embd_dim, self.class_num)
        #     for i in range(total_sessions)
        # ])

        # # 创建备份分类器池
        # self.classifier_pool_backup = nn.ModuleList([
        #     self._create_classifier(embd_dim, self.class_num)
        #     for i in range(total_sessions)
        # ])
        self.classifier_pool = nn.ModuleList()
        self.classifier_pool_backup = nn.ModuleList()
        # 当前任务索引
        self.numtask = 0
        # 在 __init__ 方法最后添加
        if args.get('use_pretrained', False):
            pretrained_path = args.get('pretrained_path', 'vit_base_patch16_224_in21k.pth')
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pth_path):
        """
        从 google/vit-base-patch16-224-in21k 权重加载到 Jittor 模型
        """
        try:
            import torch
            import numpy as np
        except ImportError:
            raise ImportError("Please install PyTorch (CPU version) to load pretrained weights")

        print(f"[Load Pretrain] Loading from {pth_path} ...")
        state = torch.load(pth_path, map_location='cpu')

        # 处理可能的嵌套
        if 'model' in state:
            state = state['model']
        elif 'state_dict' in state:
            state = state['state_dict']

        # 去除 'module.' 前缀
        state = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in state.items()}

        print("First 20 keys in state_dict:")
        for i, k in enumerate(list(state.keys())[:20]):
            print(f"  {i}: {k}")

        # 打印包含 'norm' 的键，用于定位 final norm
        norm_keys = [k for k in state.keys() if 'norm' in k.lower()]
        print("Keys containing 'norm':", norm_keys)

        vit = self.image_encoder
        with jt.no_grad():
            # ---- 1. patch_embed (Conv2d) ----
            patch_weight_key = 'embeddings.patch_embeddings.projection.weight'
            patch_bias_key   = 'embeddings.patch_embeddings.projection.bias'
            if patch_weight_key not in state:
                raise KeyError(f"Key {patch_weight_key} not found")
            vit.patch_embed.weight = jt.array(state[patch_weight_key].numpy())
            vit.patch_embed.bias   = jt.array(state[patch_bias_key].numpy())
            print("  [✓] patch_embed")

            # ---- 2. cls_token ----
            cls_key = 'embeddings.cls_token'
            vit.cls_token = jt.array(state[cls_key].numpy())
            print("  [✓] cls_token")

            # ---- 3. pos_embed ----
            pos_key = 'embeddings.position_embeddings'
            pos = state[pos_key].numpy()
            # 可能形状为 (1, num_patches+1, dim)
            if pos.shape[1] != vit.pos_embed.shape[1]:
                # 若长度不同，简单裁剪（HuggingFace 一般也是197）
                pos = pos[:, :vit.pos_embed.shape[1], :]
            vit.pos_embed = jt.array(pos)
            print("  [✓] pos_embed")

            # ---- 4. final norm (layernorm) ----
            # HuggingFace 中 final norm 通常是 'embeddings.LayerNorm.weight'
            final_norm_weight_key = None
            final_norm_bias_key = None
            for k in state.keys():
                if k == 'layernorm.weight':
                    final_norm_weight_key = k
                elif k == 'layernorm.bias':
                    final_norm_bias_key = k

            if final_norm_weight_key and final_norm_bias_key:
                vit.norm.weight = jt.array(state[final_norm_weight_key].numpy())
                vit.norm.bias   = jt.array(state[final_norm_bias_key].numpy())
                print("  [✓] final norm loaded")
            else:
                print("  [⚠] final norm not found, will remain random (may affect accuracy)")

            # ---- 5. Transformer blocks ----
            depth = len(vit.blocks)
            for i in range(depth):
                blk = vit.blocks[i]
                prefix = f'encoder.layer.{i}'

                # norm1 = layernorm_before
                norm1_w_key = f'{prefix}.layernorm_before.weight'
                norm1_b_key = f'{prefix}.layernorm_before.bias'
                blk.norm1.weight = jt.array(state[norm1_w_key].numpy())
                blk.norm1.bias   = jt.array(state[norm1_b_key].numpy())

                # norm2 = layernorm_after
                norm2_w_key = f'{prefix}.layernorm_after.weight'
                norm2_b_key = f'{prefix}.layernorm_after.bias'
                blk.norm2.weight = jt.array(state[norm2_w_key].numpy())
                blk.norm2.bias   = jt.array(state[norm2_b_key].numpy())

                # mlp.fc1 = intermediate.dense
                mlp_fc1_w_key = f'{prefix}.intermediate.dense.weight'
                mlp_fc1_b_key = f'{prefix}.intermediate.dense.bias'
                blk.mlp[0].weight = jt.array(state[mlp_fc1_w_key].numpy())
                blk.mlp[0].bias   = jt.array(state[mlp_fc1_b_key].numpy())

                # mlp.fc2 = output.dense
                mlp_fc2_w_key = f'{prefix}.output.dense.weight'
                mlp_fc2_b_key = f'{prefix}.output.dense.bias'
                blk.mlp[3].weight = jt.array(state[mlp_fc2_w_key].numpy())
                blk.mlp[3].bias   = jt.array(state[mlp_fc2_b_key].numpy())

                # 注意力层：q, k, v 分别独立
                attn = blk.attn

                # 拼接 q, k, v 权重
                q_w = state[f'{prefix}.attention.attention.query.weight'].numpy()
                k_w = state[f'{prefix}.attention.attention.key.weight'].numpy()
                v_w = state[f'{prefix}.attention.attention.value.weight'].numpy()
                qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
                attn.qkv.weight = jt.array(qkv_w)

                if attn.qkv.bias is not None:
                    q_b = state[f'{prefix}.attention.attention.query.bias'].numpy()
                    k_b = state[f'{prefix}.attention.attention.key.bias'].numpy()
                    v_b = state[f'{prefix}.attention.attention.value.bias'].numpy()
                    qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)
                    attn.qkv.bias = jt.array(qkv_b)

                # 加载输出投影
                out_w = state[f'{prefix}.attention.output.dense.weight'].numpy()
                out_b = state[f'{prefix}.attention.output.dense.bias'].numpy()
                attn.proj.weight = jt.array(out_w)
                attn.proj.bias = jt.array(out_b)

                print(f"  [✓] block {i} attention (original) loaded")

        print("[Load Pretrain] Finished.")
        return True

    def _create_classifier(self, in_features, out_features):
        """创建并初始化分类器"""
        head = nn.Linear(in_features, out_features, bias=True)
        
        # 初始化权重
        # 使用 Jittor 的 kaiming_uniform_ 初始化
        # jt.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        jt.init.gauss_(head.weight, mean=0.0, std=0.01)

        if head.bias is not None:
            # 偏置初始化为0，而不是均匀分布
            jt.init.constant_(head.bias, 0)
        return head

    @property
    def feature_dim(self):
        return self.image_encoder.embed_dim

    def extract_vector(self, image, task=None):
        if task is None:
            task_id = max(0, self.numtask - 1)
        else:
            task_id = task
        
        image_features, _ = self.image_encoder(image, task_id)
        image_features = image_features[:, 0, :]  # 取 cls token 的特征
        return image_features

    def execute(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        # print(f"SiNet execute input shape: {image.shape}")
        
        # 在进入 image_encoder 之前，先确保图像维度正确
        if len(image.shape) == 4 and image.shape[-1] == 3:
            # BHWC 转换为 BCHW
            image = image.permute(0, 3, 1, 2)
            # print(f"Permuted image shape: {image.shape}")
        
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return jt.concat(fc_outs, dim=1)

        # 确保有分类器可用
        if self.numtask <= 0 or len(self.classifier_pool) == 0:
            print(f"Warning: numtask={self.numtask}, no classifier available")
            # 返回一个默认的输出
            batch_size = image.shape[0]
            return {
                'logits': jt.zeros((batch_size, self.class_num)),
                'features': jt.zeros((batch_size, self.embd_dim)),
                'prompt_loss': jt.zeros((1,))
            }
        
        # 确保索引有效
        task_idx = min(self.numtask - 1, len(self.classifier_pool) - 1)
        
        image_features, prompt_loss = self.image_encoder(
            image, 
            task_id=task_idx, 
            get_feat=get_feat, 
            get_cur_feat=get_cur_feat
        )
        image_features = image_features[:, 0, :]  # 取 cls token 的特征
        image_features = image_features.view(image_features.shape[0], -1)
        
        if task_idx < 0 or task_idx >= len(self.classifier_pool):
            # 尚无分类器，返回零张量（一般不会发生，仅防御）
            batch_size = image.shape[0]
            return {
                'logits': jt.zeros((batch_size, self.class_num)),
                'features': jt.zeros((batch_size, self.embd_dim)),
                'prompt_loss': jt.zeros((1,))
            }
        
        classifier = self.classifier_pool[task_idx]
        out = classifier(image_features)
        if isinstance(out, tuple):
            out = out[0]
        
        return {
            'logits': out,
            'features': image_features,
            'prompt_loss': prompt_loss
        }
    
    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        return self.execute(image, get_feat, get_cur_feat, fc_only)

    def interface(self, image, task_id=None):
        if task_id is None:
            task_id = self.numtask - 1
        
        # 确保 task_id 有效
        if task_id < 0:
            print(f"Warning: task_id={task_id} is negative, using 0")
            task_id = 0
        
        # 确保图像维度正确
        if len(image.shape) == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
            # print(f"interface: permuted image shape: {image.shape}")
        
        image_features, _ = self.image_encoder(image, task_id=task_id)
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.shape[0], -1)

        # print(f"interface: image_features shape: {image_features.shape}")
        # print(f"interface: numtask={self.numtask}, classifier_pool length={len(self.classifier_pool)}")
        
        # 收集所有已学任务的 logits
        logits_list = []
        for i in range(min(self.numtask, len(self.classifier_pool))):
            out = self.classifier_pool[i](image_features)
            # 防止 Jittor 返回元组
            if isinstance(out, tuple):
                out = out[0]
            logits_list.append(out)
        
        if len(logits_list) == 0:
            batch_size = image.shape[0]
            return jt.zeros((batch_size, self.class_num))
        
        all_logits = jt.concat(logits_list, dim=1)
        return all_logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            task_id = int(task_ids[index].item())
            image_features, _ = self.image_encoder(
                image[index:index+1], 
                task_id=task_id
            )
            image_features = image_features[:, 0, :]
            image_features = image_features.view(image_features.shape[0], -1)
            logits.append(self.classifier_pool_backup[task_id](image_features))

        return jt.concat(logits, dim=0)

    def interface2(self, image_features):
        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        return jt.concat(logits, dim=1)

    def update_fc(self, nb_classes):
        """
        nb_classes: 当前任务的类别数（不是总类别数）
        """
        logging.info(f"[DEBUG update_fc] Entering: numtask={self.numtask}, nb_classes={nb_classes}, current pool len={len(self.classifier_pool)}")
        self.numtask += 1
        print(f"[SiNet] update_fc: task={self.numtask}, nb_classes={nb_classes}, pool_len={len(self.classifier_pool)}")
        
        # 确保分类器池长度至少等于当前任务索引+1
        while len(self.classifier_pool) < self.numtask:
            new_head = self._create_classifier(self.embd_dim, nb_classes)
            self.classifier_pool.append(new_head)
            # 同步备份池
            new_head_backup = self._create_classifier(self.embd_dim, nb_classes)
            self.classifier_pool_backup.append(new_head_backup)
            print(f"Added classifier for task {len(self.classifier_pool)-1}, pool length: {len(self.classifier_pool)}, out_dim={nb_classes}")
        
        # 备份当前任务的分类器
        logging.info(f"[DEBUG update_fc] After adding: numtask={self.numtask}, pool len={len(self.classifier_pool)}")
        self.classifier_backup(self.numtask - 1)

    def classifier_backup(self, task_id):
        """备份指定任务的分类器"""
        if task_id < len(self.classifier_pool) and task_id < len(self.classifier_pool_backup):
            # 复制权重
            self.classifier_pool_backup[task_id].load_state_dict(
                self.classifier_pool[task_id].state_dict()
            )

    def classifier_recall(self):
        """恢复分类器 (简化版本)"""
        print("Note: classifier_recall is not fully implemented in Jittor version")

    def copy(self):
        return deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.stop_grad()
        self.eval()
        return self