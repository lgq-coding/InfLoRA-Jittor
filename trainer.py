import os
import os.path
import sys
import logging
import copy
import time
import jittor as jt  
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    # Jittor替换：移除torch.backends.cudnn设置，使用jt.flags
    # jt.flags.deterministic = 1  # Jittor的确定性模式（如果需要）
    jt.set_seed(myseed)  # Jittor设置随机种子


def _train(args):
    if args['model_name'] in ['InfLoRA', 'InfLoRA_domain', 'InfLoRAb5_domain', 'InfLoRAb5', 'InfLoRA_CA', 'InfLoRA_CA1']:
        logdir = 'logs/{}/{}_{}_{}/{}/{}/{}/{}_{}-{}'.format(args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim'], args['rank'], args['lamb'], args['lame'], args['lrate'])
    else:
        logdir = 'logs/{}/{}_{}_{}/{}/{}'.format(args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}'.format(args['seed']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    if not os.path.exists(logfilename):
        os.makedirs(logfilename)
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)

    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        # raise Exception
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
        cnn_curve_task['top1'].append(cnn_accy_task)
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
        logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))

        # if task >= 3: break

        # Jittor替换：使用jt.save替代torch.save
        jt.save(model._network.state_dict(), os.path.join(logfilename, "task_{}.pth".format(int(task))))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            # Jittor替换：CPU模式
            device = 'cpu'
        else:
            # Jittor替换：GPU设备表示
            device = 'cuda:{}'.format(device)

        gpus.append(device)

    args['device'] = gpus
    # Jittor附加：设置使用GPU还是CPU
    if device_type != -1 and len(device_type) > 0:
        jt.flags.use_cuda = 1
        # 设置可见的GPU设备
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_type])


def _set_random(args):
    # Jittor替换：移除torch的随机种子设置，使用jt.set_seed
    # jt.set_seed(args['seed'])
    # 为了兼容性，保留torch风格调用，但使用jt的函数
    pass  # Jittor的随机种子已经在train函数中设置了


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))