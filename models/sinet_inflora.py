import math
import jittor as jt
import jittor.nn as nn
from copy import deepcopy

##等待修改！！！
# ===============================
#   ViT wrapper (unchanged)
# ===============================
class ViT_lora_co(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, task_id=0, get_cur_feat=False):
        return self.backbone(x, task_id=task_id, get_cur_feat=get_cur_feat)
    
    def execute(self, x, task_id=None, get_cur_feat=False):
        return self.forward(x, task_id=task_id, get_cur_feat=get_cur_feat)
    
# ===============================
#   SiNet for InfLoRA
# ===============================
class SiNet(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()

        self.args = args
        self.backbone = ViT_lora_co(backbone)

        self.embd_dim = args['embd_dim']
        self.class_num = args['init_cls']

        # current task index
        self.numtask = 0

        # classifier pool (one head per task)
        self.classifier_pool = nn.ModuleList()
        self.classifier_pool_backup = nn.ModuleList()

    # --------------------------------------------------
    #  Feature extractor
    # --------------------------------------------------
    def extract_vector(self, x):
        task_id = max(0, self.numtask - 1)   # [MODIFIED] safety
        feat, _ = self.backbone(x, task_id=task_id)
        return feat

    # --------------------------------------------------
    #  Forward
    # --------------------------------------------------
    def execute(self, x, get_cur_feat=False):
        task_id = max(0, self.numtask - 1)   # safety
        feats, prompt_loss = self.backbone(x, task_id=task_id, get_cur_feat=get_cur_feat)

        idx = self._safe_task_index(task_id)
        logits = self.classifier_pool[idx](feats)
        return logits, prompt_loss

    # --------------------------------------------------
    #  Safe task index helper
    # --------------------------------------------------
    def _safe_task_index(self, idx):
        if idx < 0:
            return 0
        if idx >= len(self.classifier_pool):
            return len(self.classifier_pool) - 1
        return idx

    # --------------------------------------------------
    #  Update classifier + LoRA slots
    # --------------------------------------------------
    def update_fc(self, nb_classes):
        """
        Increase task count and:
        1) expand classifier_pool
        2) ensure LoRA A/B exist for new task
        """

        self.numtask += 1
        device = self.parameters()[0].device if self.parameters() else None

        # ========== classifier head ==========
        while len(self.classifier_pool) < self.numtask:
            if len(self.classifier_pool) == 0:
                head = nn.Linear(self.embd_dim, nb_classes)
            else:
                ref = self.classifier_pool[0]
                head = nn.Linear(ref.in_features, ref.out_features, bias=(ref.bias is not None))
            if hasattr(head, 'weight'):
                nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)

            self.classifier_pool.append(head)

        # backup old heads (used by evaluation)
        self.classifier_pool_backup = deepcopy(self.classifier_pool)

        # ========== ensure LoRA slots ==========
        cur_task = self.numtask - 1
        for m in self.backbone.modules():
            if not hasattr(m, "lora_A_k"):
                continue

            # [MODIFIED] ensure enough LoRA slots
            self._ensure_lora_slots(m, cur_task)

    # --------------------------------------------------
    #  Ensure LoRA A/B exists for task
    # --------------------------------------------------
    def _ensure_lora_slots(self, module, task_id):
        device = self.parameters()[0].device if self.parameters() else None
        rank = module.rank
        embed_dim = module.embed_dim

        def _append_linear(mlist, in_dim, out_dim):
            layer = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.constant_(layer.weight, 0)
            mlist.append(layer)

        while len(module.lora_A_k) <= task_id:
            _append_linear(module.lora_A_k, embed_dim, rank)
            _append_linear(module.lora_B_k, rank, embed_dim)
            _append_linear(module.lora_A_v, embed_dim, rank)
            _append_linear(module.lora_B_v, rank, embed_dim)

    # --------------------------------------------------
    #  Interfaces used by trainer
    # --------------------------------------------------
    def interface(self, x):
        task_id = max(0, self.numtask - 1)
        feat, _ = self.backbone(x, task_id=task_id)
        out = []
        for i in range(len(self.classifier_pool_backup)):
            out.append(self.classifier_pool_backup[i](feat))
        return jt.concat(out, dim=1)

    def interface1(self, x):
        task_id = max(0, self.numtask - 1)
        feat, _ = self.backbone(x, task_id=task_id)
        idx = self._safe_task_index(task_id)
        return self.classifier_pool_backup[idx](feat)

    def interface2(self, x):
        return self.interface1(x)
