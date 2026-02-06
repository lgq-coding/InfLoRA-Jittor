import os
import numpy as np
import jittor as jt
from jittor import nn


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    if isinstance(x, jt.Var):
        # Jittor不需要.is_cuda判断，直接使用.numpy()
        return x.numpy()
    elif isinstance(x, (list, tuple)):
        return [tensor2numpy(item) for item in x]
    elif isinstance(x, dict):
        return {key: tensor2numpy(value) for key, value in x.items()}
    else:
        return x


def target2onehot(targets, n_classes):
    onehot = jt.zeros((targets.shape[0], n_classes), dtype=jt.float32)
    indices = targets.long().unsqueeze(1)  # [batch_size, 1]
    values = jt.ones_like(targets, dtype=jt.float32).unsqueeze(1)  # [batch_size, 1]

    # Jittor的scatter操作
    onehot.scatter_(1, indices, values)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)




def accuracy_domain(y_pred, y_true, nb_old, increment=2, class_num=1):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred%class_num == y_true%class_num).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes),decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes), decimals=2)

    return all_acc



def accuracy_binary(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred%2 == y_true%2).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    # all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),decimals=2)
    all_acc['old'] = 0 if len(idxes) == 0 else np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes),decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)

    return all_acc


# Jittor特有的辅助函数
def jittor_one_hot(tensor, num_classes):
    """Jittor版的one-hot编码（更简洁的实现）"""
    return jt.nn.one_hot(tensor, num_classes)


def sync_cuda():
    """同步CUDA操作（Jittor通常不需要显式同步）"""
    if jt.flags.use_cuda:
        jt.sync_all()


def save_model(model, path):
    """保存Jittor模型"""
    model.save(path)


def load_model(model, path):
    """加载Jittor模型"""
    model.load(path)


def set_seed(seed):
    """设置随机种子"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    jt.set_seed(seed)


# 深度学习常用指标计算
def compute_metrics(predictions, targets, num_classes=None):
    """
    计算多种分类指标

    Args:
        predictions: 预测结果，可以是logits或类别索引
        targets: 真实标签
        num_classes: 类别数量

    Returns:
        dict: 包含各种指标的字典
    """
    if isinstance(predictions, jt.Var):
        predictions = tensor2numpy(predictions)
    if isinstance(targets, jt.Var):
        targets = tensor2numpy(targets)

    # 如果predictions是logits（二维），转换为类别索引
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    metrics = {}

    # 准确率
    accuracy = (predictions == targets).mean()
    metrics['accuracy'] = np.around(accuracy * 100, decimals=2)

    if num_classes is not None:
        # 各类别的准确率
        class_accuracies = []
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                class_acc = (predictions[mask] == c).mean()
                class_accuracies.append(class_acc)
                metrics[f'class_{c}_acc'] = np.around(class_acc * 100, decimals=2)

        # 平均类别准确率（当类别不平衡时更有意义）
        if class_accuracies:
            metrics['mean_class_acc'] = np.around(np.mean(class_accuracies) * 100, decimals=2)

    return metrics


def create_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0):
    """
    创建优化器

    Args:
        model: 模型
        optimizer_name: 优化器名称 ('adam', 'sgd', 'adamw')
        lr: 学习率
        weight_decay: 权重衰减

    Returns:
        优化器实例
    """
    if optimizer_name.lower() == 'adam':
        return jt.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return jt.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name.lower() == 'adamw':
        return jt.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def create_lr_scheduler(optimizer, scheduler_name='step', **kwargs):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('step', 'cosine', 'multi_step')
        **kwargs: 调度器参数

    Returns:
        学习率调度器
    """
    if scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return jt.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name.lower() == 'multi_step':
        milestones = kwargs.get('milestones', [30, 60, 90])
        gamma = kwargs.get('gamma', 0.1)
        return jt.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")

