# ============================================================
# methods/inflora_jt.py
# Jittor re-implementation of InfLoRA
# ============================================================
# # python main.py --device 0 --config configs/cifar100_inflora_debug.json
import math
from copy import deepcopy
import logging
import numpy as np

import jittor as jt
import jittor.nn as nn
from jittor import optim

from sklearn.cluster import KMeans

from .base import BaseLearner
from models.sinet_inflora import SiNet,Attention_LoRA
from utils.toolkit import accuracy


# ============================================================
# Helper functions (JT specific)
# ============================================================

def freeze(param):
    """Freeze parameter (stop gradient)"""
    # [JT-CHANGE] torch.requires_grad = False → stop_grad()
    param.stop_grad()
    

def unfreeze(param):
    """Unfreeze parameter (enable gradient)"""
    # [JT-CHANGE] torch.requires_grad = True → start_grad()
    param.start_grad()

def to_var(x):
    """Convert numpy / list to jt.Var"""
    if isinstance(x, jt.Var):
        return x
    return jt.array(x)


# ============================================================
# InfLoRA
# ============================================================

class InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------
        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError("Unknown net type")

        self.args = args

        # ----------------------------------------------------
        # Hyper-parameters
        # ----------------------------------------------------
        self.init_epoch = args["init_epoch"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.batch_size = args["batch_size"]
        self.data_manager=None
        # ----------------------------------------------------
        # InfLoRA / DualGPM
        # ----------------------------------------------------
        self.lamb = args.get("lamb", 0.98)
        self.lame = args.get("lame", 1.0)
        self.total_sessions = args["total_sessions"]

        # Gradient subspace memory
        self.feature_list = []     # orthonormal bases (GPM)
        self.project_type = []     # remove / retain
        self.all_keys = []         # clustering centers

        # Task bookkeeping
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

    # ========================================================
    # Incremental training interface
    # ========================================================

    def incremental_train(self, data_manager):
        """
        Train one incremental task
        """
        self.data_manager=data_manager
        # 手动管理任务索引
        if not hasattr(self, '_cur_task') or self._cur_task < 0:
            self._cur_task = 0
        else:
            self._cur_task += 1

        # 获取当前任务的类别数
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size

        logging.info(f"[DEBUG] Task {self._cur_task}: known_classes={self._known_classes}, task_size={task_size}, total_classes={self._total_classes}")
        logging.info(f"[DEBUG] Before update_fc: numtask={self._network.numtask}, classifier_pool length={len(self._network.classifier_pool)}")
        # update classifier head
        # current_task_size = data_manager.get_task_size(self._cur_task)
        self._network.update_fc(task_size)

        logging.info(
            f"Task {self._cur_task}: classes "
            f"{self._known_classes} → {self._total_classes}"
        )
        logging.info(f"[DEBUG] After update_fc: numtask={self._network.numtask}, classifier_pool length={len(self._network.classifier_pool)}")
        train_indices = list(range(self._known_classes, self._total_classes))
        train_dataset = self.data_manager.get_dataset(
            indices=train_indices,
            source="train", mode="train"
        )
        # test_dataset = data_manager.get_dataset(
        #     np.arange(0, self._total_classes),
        #     source="test", mode="test"
        # )
        from jittor.dataset import DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=0)
        # self.test_loader = test_dataset

        self._train()
        self.clustering(self.train_loader)

        self._known_classes = self._total_classes

    # ========================================================
    # Core training logic
    # ========================================================

    def _train(self):
        """
        Key InfLoRA logic:
        1. Freeze all parameters
        2. Build / freeze B (basis)
        3. Enable only A (coefficients) for current task
        4. Train classifier head + A
        """

        # ----------------------------------------------------
        # 1. Freeze everything
        # ----------------------------------------------------
        for p in self._network.parameters():
            # print("freeze p!")
            freeze(p)

        # ----------------------------------------------------
        # 2. Enable current-task LoRA A
        # ----------------------------------------------------
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                t = self._cur_task
                print(f"Unfreezing LoRA A for task {t} in module {module}")
                # [InfLoRA CORE]
                # Only A is trainable, B is frozen
                while len(module.lora_A_q) <= t:
                    # 动态添加新的 LoRA 参数（根据当前 rank 和 embed_dim）
                    new_A = nn.Linear(module.embed_dim, module.rank, bias=False)
                    new_B = nn.Linear(module.rank, module.embed_dim, bias=False)
                    jt.init.gauss_(new_A.weight, 0.0, 0.02)
                    jt.init.constant_(new_B.weight, 0)
                    module.lora_A_q.append(new_A)
                    module.lora_B_q.append(new_B)
                while len(module.lora_A_k) <= t:
                    new_A = nn.Linear(module.embed_dim, module.rank, bias=False)
                    new_B = nn.Linear(module.rank, module.embed_dim, bias=False)
                    jt.init.gauss_(new_A.weight, 0.0, 0.02)
                    jt.init.constant_(new_B.weight, 0)
                    module.lora_A_k.append(new_A)
                    module.lora_B_k.append(new_B)
                while len(module.lora_A_v) <= t:
                    new_A = nn.Linear(module.embed_dim, module.rank, bias=False)
                    new_B = nn.Linear(module.rank, module.embed_dim, bias=False)
                    jt.init.gauss_(new_A.weight, 0.0, 0.02)
                    jt.init.constant_(new_B.weight, 0)
                    module.lora_A_v.append(new_A)
                    module.lora_B_v.append(new_B)
                while len(module.lora_A_o) <= t:
                    new_A = nn.Linear(module.embed_dim, module.rank, bias=False)
                    new_B = nn.Linear(module.rank, module.embed_dim, bias=False)
                    jt.init.gauss_(new_A.weight, 0.0, 0.02)
                    jt.init.constant_(new_B.weight, 0)
                    module.lora_A_o.append(new_A)
                    module.lora_B_o.append(new_B)


                if t == 0:
                    unfreeze(module.lora_A_q[t].weight)
                    unfreeze(module.lora_B_q[t].weight)
                    unfreeze(module.lora_A_k[t].weight)
                    unfreeze(module.lora_B_k[t].weight)
                    unfreeze(module.lora_A_v[t].weight)
                    unfreeze(module.lora_B_v[t].weight)
                    unfreeze(module.lora_A_o[t].weight)
                    unfreeze(module.lora_B_o[t].weight)
                else:
                    # 后续任务：仅解冻A（组合系数），B保持冻结
                    unfreeze(module.lora_A_q[t].weight)
                    unfreeze(module.lora_A_k[t].weight)
                    unfreeze(module.lora_A_v[t].weight)
                    unfreeze(module.lora_A_o[t].weight)
                    # B不解冻，保持为基础子空间

        # ----------------------------------------------------
        # 3. Enable classifier head
        # ----------------------------------------------------
        for name, p in self._network.named_parameters():
            if "classifier_pool" in name:
                print(f"Unfreezing classifier parameter: {name}")
                unfreeze(p)

        # if self._cur_task > 0:
        #     self._apply_orthogonal_constraints()
        # ----------------------------------------------------
        # 4. Optimizer
        # ----------------------------------------------------
        # [JT-CHANGE]
        # Jittor optimizer sees all params,
        # but gradient flow is controlled by stop_grad()
        optimizer = optim.AdamW(
            self._network.parameters(),
            lr=self.lrate,
            weight_decay=0.01
            # momentum=0.9,
            # weight_decay=5e-4
        )
        print("Trainable parameters after unfreeze:")
        for name, p in self._network.named_parameters():
            if not p.is_stop_grad():
                print(f"  {name} is trainable")
        # ----------------------------------------------------
        # 5. Training loop
        # ----------------------------------------------------
        if self._network.numtask <= 0:
            print("Warning: No tasks initialized yet. Updating classifier for first task...")
            # 更新分类器以初始化第一个任务
            self._network.update_fc(self._total_classes)

        print(f"Before training: numtask={self._network.numtask}, classifier_pool length={len(self._network.classifier_pool)}")

        current_task_idx = self._network.numtask - 1
        if current_task_idx >= 0 and current_task_idx < len(self._network.classifier_pool):
            classifier_out_dim = self._network.classifier_pool[current_task_idx].out_features
            # logging.info(f"[DEBUG] Current task classifier output dim: {classifier_out_dim}")
            # logging.info(f"[DEBUG] Expected local label range: 0~{classifier_out_dim-1}")  
        task_index=self._known_classes // 10
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct=0
            total=0
            
            for batch_idx, (_, imgs, targets) in enumerate(self.train_loader):
                # 确保输入是 Jittor Var
                if not isinstance(imgs, jt.Var):
                    imgs = jt.array(imgs)
                if not isinstance(targets, jt.Var):
                    targets = jt.array(targets, dtype=jt.int64)
                
                # 确保图像是 BCHW 格式
                if len(imgs.shape) == 4 and imgs.shape[-1] <= 3:
                    # BHWC 格式，转换为 BCHW
                    imgs = imgs.permute(0, 3, 1, 2)
                if epoch == 0:
                    with jt.no_grad():
                        sample_imgs = imgs[:4]  # 取 4 张图
                        feat, _ = self._network.image_encoder(sample_imgs, task_id=0)
                        # print(f"[Backbone] feature shape: {feat.shape}")
                        # print(f"[Backbone] feature mean: {feat.mean().item():.4f}, std: {feat.std().item():.4f}")
                        # print(f"[Backbone] feature min: {feat.min().item():.4f}, max: {feat.max().item():.4f}")
                # print(f"Training batch - imgs shape: {imgs.shape}, numtask: {self._network.numtask}")
                
                outputs = self._network(imgs)
                
                # 处理输出
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                elif isinstance(outputs, tuple):
                    logits = outputs[0] if len(outputs) > 0 else None
                else:
                    logits = outputs

                if logits is None:
                    continue
                if batch_idx == 0 and epoch == 0:
                    targets_np = targets.numpy().flatten()
                    unique_targets, counts = np.unique(targets_np, return_counts=True)
                    logging.info(f"[Data] Batch targets: {dict(zip(unique_targets, counts))}")
                mask = (targets >= self._known_classes) & (targets < self._total_classes)
                if mask.sum() == 0:
                    # 如果 batch 中没有当前任务样本，则跳过
                    continue
                
                current_logits = logits[mask]
                current_targets = targets[mask]
                # 将全局标签转换为局部标签（0 到 task_size-1）
                local_targets = current_targets - self._known_classes
                # local_targets=local_targets.int()  # 确保是整数类型
                if batch_idx == 0 and epoch == 0:
                    logging.info(f"[Mask] mask sum: {mask.sum().item()}, known_classes={self._known_classes}, total_classes={self._total_classes}")
                    logging.info(f"[Targets] current_targets range: {current_targets.min().item()}~{current_targets.max().item()}")
                    logging.info(f"[Local] local_targets range: {local_targets.min().item()}~{local_targets.max().item()}")
                    logging.info(f"[Logits] current_logits shape: {current_logits.shape}, mean={current_logits.mean().item():.2f}, std={current_logits.std().item():.2f}")
                    # ---------------------------------------------------------
                    logging.info(f"  [DEBUG] current_logits mean: {current_logits.mean().item():.4f}, std: {current_logits.std().item():.4f}")
                    logging.info(f"  [DEBUG] current_logits max: {current_logits.max().item():.4f}, min: {current_logits.min().item():.4f}")
                preds_train = current_logits.argmax(dim=1)
                if isinstance(preds_train, tuple):
                    preds_train = preds_train[0]
                if batch_idx == 0 and epoch == 0:
                    logging.info(f"  [DEBUG] preds distribution: {np.unique(preds_train.numpy(), return_counts=True)}")
                # 计算 loss
                loss = nn.cross_entropy_loss(current_logits, local_targets)
                optimizer.step(loss)
                if batch_idx == 0 and epoch == 0:
                    logging.info(f"  [DEBUG] current_logits mean: {current_logits.mean().item():.4f}, std: {current_logits.std().item():.4f}")
                    logging.info(f"  [DEBUG] current_logits max: {current_logits.max().item():.4f}, min: {current_logits.min().item():.4f}")
                # total_norm = 0.0
                # 放在 optimizer.step(loss) 之后
                # if batch_idx % 10 == 0:
                #     total_norm = 0.0
                    # logging.info(f"\n[Grad Debug] Batch {batch_idx}:")
                    # for name, p in self._network.named_parameters():
                    #     grad = p.opt_grad(optimizer)
                    #     if grad is not None:
                    #         # 计算梯度的全局范数（先 norm 再 sum 确保标量）
                    #         grad_norm = grad.norm().sum().item()
                    #         total_norm += grad_norm ** 2
                    #         if grad_norm > 1e-6:
                                # logging.info(f"  grad {name}: {grad_norm:.6f}")
                        # else:
                            # logging.info(f"  {name} has no grad")
                    
                    # total_norm = total_norm ** 0.5
                    # logging.info(f"  total grad norm: {total_norm:.6f}")
                    # 打印分类器权重范数
                    # w = self._network.classifier_pool[self._cur_task].weight
                    # logging.info(f"  classifier weight norm: {w.norm().sum().item():.6f}")
                total_loss += loss.item()
                log_softmax = current_logits - current_logits.logsumexp(dim=1, keepdims=True)
                nll = -log_softmax[jt.arange(len(local_targets)), local_targets]
                manual_loss = nll.mean()
                
                preds = current_logits.argmax(dim=1)
                if isinstance(preds, tuple):
                    preds = preds[0]
                # print(preds)
                preds = preds.int()
                # preds*=10
                # print(preds)
                correct += (preds == local_targets).sum().item()
                total += len(local_targets)

            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            logging.info(
                f"[Task {task_index}] "
                f"Epoch {epoch+1}/{self.epochs}, "
                # f"Loss={total_loss:.3f}, "
                f"[Loss] loss: {manual_loss.item():.4f} "
                f"Accuracy={epoch_acc:.4f}%"
            )

        # ----------------------------------------------------
        # 6. Collect activations & update DualGPM
        # ----------------------------------------------------
        mat_list = []
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))

        self.update_DualGPM(mat_list)

    # ========================================================
    # Clustering (same as original)
    # ========================================================

    def clustering(self, dataloader):
        features = []

        for _, imgs, targets in dataloader:
            with jt.no_grad():
                feat = self._network.extract_vector(imgs)
                feat = feat / jt.norm(feat, dim=1, keepdims=True)
                features.append(feat)

        if len(features) == 0:
            return

        features = jt.concat(features, dim=0).numpy()
        km = KMeans(n_clusters=5, random_state=0).fit(features)

        # store task keys
        self.all_keys.append(jt.array(km.cluster_centers_))


    def _apply_orthogonal_constraints(self):
        """
        为后续任务添加梯度钩子，确保新任务的梯度与旧任务的基础子空间正交
        这是InfLoRA消除干扰的核心机制
        """
        def make_grad_hook(basis):
            def hook(grad):
                if grad is None or basis is None:
                    return grad
                # 将梯度投影到与基础子空间正交的方向
                # proj = basis @ basis^T @ grad
                proj = grad @ basis.T @ basis
                return grad - proj  # 保留正交分量
            return hook

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA) and module.subspace_basis is not None:
                basis = module.subspace_basis
                # 为当前任务的A参数注册梯度钩子
                t = self._cur_task
                module.lora_A_q[t].weight.register_hook(make_grad_hook(basis))
                module.lora_A_k[t].weight.register_hook(make_grad_hook(basis))
                module.lora_A_v[t].weight.register_hook(make_grad_hook(basis))
                module.lora_A_o[t].weight.register_hook(make_grad_hook(basis))
    # ========================================================
    # DualGPM (core continual-learning constraint)
    # ========================================================

    def update_DualGPM(self, mat_list):
        """
        Dual Gradient Projection Memory
        (Structure strictly follows original implementation)
        """

        threshold = (
            (self.lame - self.lamb)
            * self._cur_task / max(1, self.total_sessions)
            + self.lamb
        )
        print("DualGPM threshold:", threshold)
        lora_layers = []
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                lora_layers.append(module)
        # ----------------------------------------------------
        # First task
        # ----------------------------------------------------
        if len(self.feature_list) == 0:
            for i,act in enumerate(mat_list):
                A = to_var(act)

                # [JT-CHANGE] torch.linalg.svd → jt.linalg.svd
                U, S, V = jt.linalg.svd(A)

                sval = (S ** 2).data
                ratio = sval / (sval.sum() + 1e-12)

                r = int(np.sum(np.cumsum(ratio) < threshold))
                r = max(r, 1)

                basis = U[:, :r]
                self.feature_list.append(basis)
                self.project_type.append("remove")
                
                # 将正交基存入对应的Attention层
                if i < len(lora_layers):
                    lora_layers[i].set_subspace_basis(basis)
            return

        # ----------------------------------------------------
        # Subsequent tasks
        # ----------------------------------------------------
        for i, act in enumerate(mat_list):
            if i >= len(lora_layers):
                continue
            A = to_var(act)
            F = self.feature_list[i]

            proj = F @ F.transpose() @ A
            A_hat = A - proj

            U, S, V = jt.linalg.svd(A_hat)

            sval = (S ** 2).data
            ratio = sval / (sval.sum() + 1e-12)
            r = int(np.sum(np.cumsum(ratio) < threshold))

            if r > 0:
                # 扩展正交基
                new_basis = jt.concat([F, U[:, :r]], dim=1)
                self.feature_list[i] = new_basis
                lora_layers[i].set_subspace_basis(new_basis)


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------
        print("-" * 40)
        print("DualGPM Summary")
        for i, f in enumerate(self.feature_list):
            print(
                f"Layer {i+1}: "
                f"{f.shape[1]}/{f.shape[0]} basis"
            )
        print("-" * 40)

    # 在 inflora.py 的 InfLoRA 类中添加

    def eval_task(self):
        """评估方法 - 返回包含所有必需键的字典"""
        print("Running eval_task...")
        
        # 计算准确率
        accuracy = self._compute_accuracy()
        
        # 创建符合期望的数据结构
        # 根据错误信息，至少需要 'grouped' 和 'top1' 键
        cnn_accy = {
            'grouped': accuracy,
            'top1': accuracy,
            'total': accuracy,
            'per_task': [accuracy] * self._cur_task if hasattr(self, '_cur_task') else [accuracy],
            'incremental': accuracy,
            'old': accuracy,
            'new': accuracy
        }
        
        # 其他三个字典也保持类似结构
        cnn_accy_with_task = {
            'grouped': accuracy,
            'top1': accuracy,
            'total': accuracy
        }
        
        nme_accy = {
            'grouped': accuracy,
            'top1': accuracy,
            'total': accuracy
        }
        
        cnn_accy_task = {
            'grouped': accuracy,
            'top1': accuracy,
            'total': accuracy
        }
        
        print(f"Returning accuracy: {accuracy:.2f}%")
        
        # 返回4个字典
        return cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task

    def _compute_accuracy(self):
        self._network.eval()
        test_indices = list(range(self._total_classes))
        test_dataset = self.data_manager.get_dataset(indices=test_indices, source='test', mode='test')
        from jittor.dataset import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=0)

        all_preds, all_targets = [], []
            
        with jt.no_grad():
            for i, data in enumerate(test_loader):
                # if i >= 10:  # 限制批次以加快调试
                #     break
                if len(data) == 3:
                    _, inputs, labels = data
                elif len(data) == 2:
                    inputs, labels = data
                else:
                    continue
                
                # 类型与维度处理
                if not isinstance(inputs, jt.Var):
                    inputs = jt.array(inputs)
                if not isinstance(labels, jt.Var):
                    labels = jt.array(labels, dtype=jt.int64)
                if len(inputs.shape) == 4 and inputs.shape[-1] == 3:
                    inputs = inputs.permute(0, 3, 1, 2)
                
                logits = self._network.interface(inputs)
                
                # 确保 logits 不是元组
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # logging.info(f"  [Test] logits shape: {logits.shape}, labels shape: {labels.shape}")
                # logging.info(f"  [Test] labels min: {labels.min().item()}, max: {labels.max().item()}")
                
                if logits is not None and hasattr(logits, 'argmax'):
                    argmax_result = logits.argmax(dim=1)
                    if isinstance(argmax_result, tuple):
                        preds = argmax_result[0]   # 索引在第二个位置
                    else:
                        preds = argmax_result
                    
                    # logging.info(f"  [Test] preds unique values: {np.unique(preds.numpy())}")
                    # 转换为一维列表
                    preds_np = preds.numpy().flatten().tolist()
                    labels_np = labels.numpy().flatten().tolist()
                    
                    # 检查长度一致
                    assert len(preds_np) == len(labels_np), f"Batch {i}: preds {len(preds_np)} != labels {len(labels_np)}"
                    
                    all_preds.extend(preds_np)
                    all_targets.extend(labels_np)
        
        self._network.train()
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        # 转换为整型后再比较
        all_preds = all_preds.astype(int)
        all_targets = all_targets.astype(int)
        if len(all_preds) != len(all_targets):
            print(f"ERROR: Total preds {len(all_preds)} != targets {len(all_targets)}")
            return 0.0
        correct = (all_preds == all_targets).sum()
        total = len(all_targets)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        logging.info(f"Overall test accuracy: {correct}/{total} = {accuracy:.2f}%")
        return accuracy