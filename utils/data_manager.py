import logging
import numpy as np
from PIL import Image

# Jittor替换：将torch.utils.data和torchvision替换为jittor对应模块
from jittor.dataset import Dataset  # 替换torch.utils.data.Dataset
import jittor.transform as transforms  # 替换torchvision.transforms
from jittor.dataset import CIFAR100, CIFAR10  # 如果有需要可以导入Jittor的数据集

# 这里假设您的utils.data中的iCIFAR100等已经转换为Jittor版本
from utils.data import iCIFAR100, iIMAGENET_R, iIMAGENET_A, iCUB, iDomainNet, iCIFAR10



class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)


    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            # Jittor替换：RandomHorizontalFlip替换
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(1.0), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            # Jittor替换：RandomHorizontalFlip替换
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(1.0), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

# def pil_loader(path):
#     """加载图像文件的辅助函数"""
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

class DummyDataset(Dataset):  # 基类已替换为jittor.dataset.Dataset
    def __init__(self, images, labels, trsf, use_path=False):
        super().__init__()
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.set_attrs(total_len=len(self.images))  # Jittor数据集需要设置总长度

    def __getitem__(self, idx):
        index=idx
        if self.use_path:
            # 从路径加载图像
            image = pil_loader(self.images[idx])
        else:
            # 从数组创建图像
            # 确保图像数组是正确的数据类型和形状
            img_data = self.images[idx]
            if not isinstance(img_data, np.ndarray):
                img_data = np.array(img_data)
            if img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)
            image = Image.fromarray(img_data)
        
        # 应用转换
        if self.trsf is not None:
            image = self.trsf(image)
        if hasattr(image, 'numpy'):
            # 如果是 PyTorch Tensor
            image = image.numpy()
        elif hasattr(image, 'data'):
            # 如果是 Jittor Var
            image = image.data
        elif isinstance(image, Image.Image):
            # 如果是 PIL Image，转换为 numpy 数组
            image = np.array(image)
        
        # 确保图像形状是 HWC 格式（如果是 CHW 格式，需要转置）
        if len(image.shape) == 3 and image.shape[0] <= 3:
            # 假设是 CHW 格式，转换为 HWC
            image = np.transpose(image, (1, 2, 0))
            
        label = self.labels[idx]
    
        return index,image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == 'cifar100':
        return iCIFAR100(args)
    elif name == 'cifar10':
        return iCIFAR10(args)
    elif name == 'imagenet_r':
        return iIMAGENET_R(args)
    elif name == 'domainnet':
        return iDomainNet(args)
    elif name == 'imagenet_a':
        return iIMAGENET_A(args)
    elif name == 'cub':
        return iCUB(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # Jittor替换：这里需要调整，因为Jittor没有get_image_backend
    # 可以直接使用pil_loader
    return pil_loader(path)