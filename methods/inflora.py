# ============================================================
# methods/inflora_jt.py
# Jittor re-implementation of InfLoRA
# ============================================================

import math
from copy import deepcopy
import logging
import numpy as np

import jittor as jt
import jittor.nn as nn
from jittor import optim

from sklearn.cluster import KMeans

from .base import BaseLearner
from models.sinet_inflora import SiNet
from models.vit_inflora import Attention_LoRA
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
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size

        # update classifier head
        self._network.update_fc(self._total_classes)

        logging.info(
            f"Task {self._cur_task}: classes "
            f"{self._known_classes} → {self._total_classes}"
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train"
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test", mode="test"
        )

        self.train_loader = train_dataset
        self.test_loader = test_dataset

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
            freeze(p)

        # ----------------------------------------------------
        # 2. Enable current-task LoRA A
        # ----------------------------------------------------
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                t = self._cur_task

                # [InfLoRA CORE]
                # Only A is trainable, B is frozen
                unfreeze(module.lora_A_k[t].weight)
                unfreeze(module.lora_A_v[t].weight)

        # ----------------------------------------------------
        # 3. Enable classifier head
        # ----------------------------------------------------
        for name, p in self._network.named_parameters():
            if "classifier_pool" in name:
                unfreeze(p)

        # ----------------------------------------------------
        # 4. Optimizer
        # ----------------------------------------------------
        # [JT-CHANGE]
        # Jittor optimizer sees all params,
        # but gradient flow is controlled by stop_grad()
        optimizer = optim.Adam(
            self._network.parameters(),
            lr=self.lrate
        )

        # ----------------------------------------------------
        # 5. Training loop
        # ----------------------------------------------------
        for epoch in range(self.epochs):
            total_loss = 0.0

            for _, imgs, targets in self.train_loader:
                outputs = self._network(imgs)
                logits = outputs["logits"]

                loss = nn.cross_entropy_loss(logits, targets)
                optimizer.step(loss)

                total_loss += loss.item()

            print(
                f"[Task {self._cur_task}] "
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Loss={total_loss:.3f}"
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

        # ----------------------------------------------------
        # First task
        # ----------------------------------------------------
        if len(self.feature_list) == 0:
            for act in mat_list:
                A = to_var(act)

                # [JT-CHANGE] torch.linalg.svd → jt.linalg.svd
                U, S, V = jt.linalg.svd(A, full_matrices=False)

                sval = (S ** 2).numpy()
                ratio = sval / (sval.sum() + 1e-12)

                r = int(np.sum(np.cumsum(ratio) < threshold))
                r = max(r, 1)

                self.feature_list.append(U[:, :r])
                self.project_type.append("remove")
            return

        # ----------------------------------------------------
        # Subsequent tasks
        # ----------------------------------------------------
        for i, act in enumerate(mat_list):
            A = to_var(act)
            F = self.feature_list[i]

            proj = F @ F.transpose() @ A
            A_hat = A - proj

            U, S, V = jt.linalg.svd(A_hat, full_matrices=False)

            sval = (S ** 2).numpy()
            ratio = sval / (sval.sum() + 1e-12)

            r = int(np.sum(np.cumsum(ratio) < threshold))
            if r == 0:
                continue

            self.feature_list[i] = jt.concat(
                [F, U[:, :r]], dim=1
            )

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
