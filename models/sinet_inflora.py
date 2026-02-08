import math
import jittor as jt
import jittor.nn as nn
from jittor.attention import MultiheadAttention
from copy import deepcopy


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
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        
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
        self._init_lora_params()
    
    def _init_lora_params(self):
        """初始化 LoRA 参数"""
        # LoRA 参数列表
        self.lora_A_k = nn.ModuleList()
        self.lora_B_k = nn.ModuleList()
        self.lora_A_v = nn.ModuleList()
        self.lora_B_v = nn.ModuleList()
        
        # 初始化 LoRA 参数
        for _ in range(self.n_tasks):
            self.lora_A_k.append(nn.Linear(self.embed_dim, self.rank, bias=False))
            self.lora_B_k.append(nn.Linear(self.rank, self.embed_dim, bias=False))
            self.lora_A_v.append(nn.Linear(self.embed_dim, self.rank, bias=False))
            self.lora_B_v.append(nn.Linear(self.rank, self.embed_dim, bias=False))
            
            # 初始化 LoRA 参数为 0
            jt.init.constant_(self.lora_A_k[-1].weight, 0)
            jt.init.constant_(self.lora_B_k[-1].weight, 0)
            jt.init.constant_(self.lora_A_v[-1].weight, 0)
            jt.init.constant_(self.lora_B_v[-1].weight, 0)
    
    def execute(self, x, task_id=0):
        """前向传播"""
        # 第一层归一化
        x_norm = self.norm1(x)
        
        # 自注意力
        # 注意：Jittor 的 MultiheadAttention 期望输入形状为 (L, N, E)
        # 其中 L 是序列长度，N 是 batch size，E 是 embedding 维度
        x_norm_transposed = x_norm.transpose(0, 1)  # (N, L, E) -> (L, N, E)
        attn_output, _ = self.attn(x_norm_transposed, x_norm_transposed, x_norm_transposed)
        attn_output = attn_output.transpose(0, 1)  # (L, N, E) -> (N, L, E)
        attn_output = self.attn_drop(attn_output)
        
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
        print(f"ViT_lora_co input shape: {x.shape}")
        B, C, H, W = x.shape
        
        # 自动检测和修复维度顺序问题
        # 如果通道数很大（如224），但高度/宽度很小（如3），可能维度顺序错了
        if C != self.in_chans:
            print(f"Warning: Input channels {C} do not match expected {self.in_chans}")
            print(f"Trying to auto-correct dimension order...")
            
            # 尝试判断维度顺序：如果是 BHWC 格式，转换为 BCHW
            if C == self.img_size and (H == self.img_size or H == 3) and (W == 3 or W == self.img_size):
                print(f"Detected possible BHWC format (B, H, W, C) = ({B}, {C}, {H}, {W})")
                
                # 如果 H 或 W 是 3，可能是通道维度
                if H == 3:
                    # 可能是 (B, C, 3, W) -> 转换为 (B, 3, C, W) 或 (B, 3, W, C)?
                    # 更可能是 (B, H, W, C) = (B, 3, W, C)，其中 W 是224
                    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    print(f"Permuted to shape: {x.shape}")
                elif W == 3:
                    # 可能是 (B, C, H, 3) -> 转换为 (B, 3, C, H)?
                    x = x.permute(0, 3, 1, 2)  # (B, C, H, 3) -> (B, 3, C, H)
                    print(f"Permuted to shape: {x.shape}")
                else:
                    # 如果 C=224 且 H,W=224,3 或 3,224，尝试转置
                    # 检查最后一个维度是否为3
                    if x.shape[-1] == 3:
                        x = x.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
                        print(f"Permuted (last dim=3) to shape: {x.shape}")
                
                # 更新形状
                B, C, H, W = x.shape
                
            # 如果转换后还是不匹配，尝试其他转换
            if C != self.in_chans:
                # 尝试将通道维度移到第2维
                if x.shape[-1] == self.in_chans:
                    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    print(f"Permuted (last dim={x.shape[-1]}) to shape: {x.shape}")
                elif x.shape[1] == self.in_chans and len(x.shape) == 4:
                    # 已经正确，不需要转换
                    pass
                else:
                    raise ValueError(f"Cannot auto-correct input shape {x.shape}. Expected channels: {self.in_chans}")
                
                # 再次更新形状
                B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        print(f"After patch_embed shape: {x.shape}")
        
        # 计算patch的数量
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        num_patches = H_patch * W_patch
        
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        print(f"After flatten and transpose shape: {x.shape}")
        
        # 添加 class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)
        print(f"After adding cls_token shape: {x.shape}")
        
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
        print(f"Final output shape: {x.shape}")
        
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
    if pretrained:
        print(f"Note: Loading pretrained weights for {variant} is not implemented in this simplified version.")
    
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
        self.classifier_pool = nn.ModuleList([
            self._create_classifier(embd_dim, self.class_num)
            for i in range(total_sessions)
        ])

        # 创建备份分类器池
        self.classifier_pool_backup = nn.ModuleList([
            self._create_classifier(embd_dim, self.class_num)
            for i in range(total_sessions)
        ])

        # 当前任务索引
        self.numtask = 0
    
    def _create_classifier(self, in_features, out_features):
        """创建并初始化分类器"""
        head = nn.Linear(in_features, out_features, bias=True)
        
        # 初始化权重
        # 使用 Jittor 的 kaiming_uniform_ 初始化
        jt.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        
        if head.bias is not None:
            # 手动计算 fan_in
            if hasattr(head.weight, 'shape') and len(head.weight.shape) >= 2:
                fan_in = head.weight.shape[1]  # 输入维度
            else:
                fan_in = in_features
            
            # 使用均匀分布初始化 bias
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            jt.init.uniform_(head.bias, -bound, bound)
            
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
        print(f"SiNet execute input shape: {image.shape}")
        
        # 在进入 image_encoder 之前，先确保图像维度正确
        if len(image.shape) == 4:
            B, C, H, W = image.shape
            # 如果通道数不是3，尝试自动修正
            if C != 3:
                print(f"Warning: Input image has {C} channels, expected 3")
                
                # 尝试判断维度顺序
                if C == 224 and (H == 224 or H == 3) and (W == 3 or W == 224):
                    # 可能是 BHWC 格式
                    if H == 3:
                        image = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    elif W == 3:
                        image = image.permute(0, 3, 1, 2)  # (B, C, H, 3) -> (B, 3, C, H)
                    else:
                        # 尝试将最后一个维度移到第二维
                        image = image.permute(0, 3, 1, 2)
                    print(f"Auto-corrected image shape to: {image.shape}")
        
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return jt.concat(fc_outs, dim=1)

        image_features, prompt_loss = self.image_encoder(
            image, 
            task_id=self.numtask-1, 
            get_feat=get_feat, 
            get_cur_feat=get_cur_feat
        )
        image_features = image_features[:, 0, :]  # 取 cls token 的特征
        image_features = image_features.view(image_features.shape[0], -1)
        
        # 使用当前任务的分类器
        logits = []
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': jt.concat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }
    
    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        return self.execute(image, get_feat, get_cur_feat, fc_only)

    def interface(self, image, task_id=None):
        if task_id is None:
            task_id = self.numtask - 1
            
        image_features, _ = self.image_encoder(image, task_id=task_id)
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.shape[0], -1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        return jt.concat(logits, dim=1)
    
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
        self.numtask += 1

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