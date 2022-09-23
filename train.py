import json
"""Train script"""
import argparse
import datetime
import os
import warnings
from pathlib import Path
from time import time

from mindspore import context
from mindspore import dataset as de
from mindspore import nn
from mindspore import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback import _InternalCallbackParam

from src.pointpillars import PointPillarsWithLossCell
from src.pointpillars import TrainingWrapper
from src.utils import get_config
from src.utils import get_model_dataset

warnings.filterwarnings('ignore')

import numpy as np
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def set_default(args):
    """set default"""
    set_seed(0)

    cfg_path = Path(args.cfg_path)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    cfg = get_config(cfg_path)

    is_distributed = int(args.is_distributed)
    device_target = args.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    if is_distributed:
        # init distributed
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        rank = 0
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    return cfg, rank, device_num


def run_train(args):
    """run train"""
    cfg, rank, device_num = set_default(args)
    save_ckpt_log_flag = rank == 0

    train_cfg = cfg['train_config']

    pointpillarsnet, dataset = get_model_dataset(cfg, True)
    if save_ckpt_log_flag:
        print('PointPillarsNet created', flush=True)

    input_cfg = cfg['train_input_reader']
    n_epochs = input_cfg['max_num_epochs']
    batch_size = input_cfg['batch_size']

    steps_per_epoch = int(len(dataset) / batch_size / device_num)
    lr_cfg = train_cfg['learning_rate']
    lr = nn.exponential_decay_lr(
        learning_rate=lr_cfg['initial_learning_rate'],
        decay_rate=lr_cfg['decay_rate'],
        total_step=n_epochs * steps_per_epoch,
        step_per_epoch=steps_per_epoch,
        decay_epoch=lr_cfg['decay_epoch'],
        is_stair=lr_cfg['is_stair']
    )
    optimizer = Adam(
        pointpillarsnet.trainable_params(),
        learning_rate=lr,
        weight_decay=train_cfg['weight_decay']
    )

    pointpillarsnet_wloss = PointPillarsWithLossCell(pointpillarsnet, cfg['model'])
    
    network = TrainingWrapper(pointpillarsnet_wloss, optimizer)

    train_column_names = dataset.data_keys
    sampler = de.DistributedSampler(device_num, rank)
    ds = de.GeneratorDataset(
        dataset,
        column_names=train_column_names,
        python_multiprocessing=True,
        num_parallel_workers=1,
        max_rowsize=100,
        sampler=sampler
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(n_epochs)
    data_loader = ds.create_dict_iterator(num_epochs=n_epochs)
    network.set_train()

    if save_ckpt_log_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=steps_per_epoch,
            keep_checkpoint_max=train_cfg['keep_checkpoint_max']
        )
        ckpt_cb = ModelCheckpoint(
            config=ckpt_config,
            directory=args.save_path,
            prefix='pointpillars'
        )
        cb_params = _InternalCallbackParam()
        cb_params.train_network = pointpillarsnet
        cb_params.epoch_num = n_epochs
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    log_freq = train_cfg['log_frequency_step']
    old_progress = -1
    start = time()
    for i, data in enumerate(data_loader):
        voxels = data["voxels"]
        num_points = data["num_points"]
        coors = data["coordinates"]
        labels = data['labels']
        reg_targets = data['reg_targets']
        batch_anchors = data["anchors"]
        bev_map = data.get('bev_map', False)  # value not used if use_bev = False

        '''# 保存这一轮数据
        for value in data:
            trans = data[value].asnumpy()
            data[value] = trans

        with open('/home/hongru/4_hong/MINDSPORE/vision_ori/mindvision/ms_pointpiliar/test_data.json', 'w') as f:
            json.dump(data, f, cls=NumpyArrayEncoder)'''

        loss = network(voxels, num_points, coors, bev_map, labels, reg_targets, batch_anchors)
        if save_ckpt_log_flag:
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

            if i % log_freq == 0:
                time_used = time() - start
                epoch = i // steps_per_epoch
                fps = (i - old_progress) * batch_size * device_num / time_used
                date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'{date_time} epoch:{epoch}, iter:{i}, '
                      f'loss:{loss}, fps:{round(fps, 2)} imgs/sec',
                      flush=True)
                start = time()
                old_progress = i

            if (i + 1) % steps_per_epoch == 0:
                cb_params.cur_epoch_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='/home/hongru/4_hong/MINDSPORE/0_pointpillars/configs/ped_cycle_xyres16.yaml', help='Path to config file.')
    parser.add_argument('--save_path', default='/home/hongru/4_hong/MINDSPORE/0_pointpillars/experiment_test', help='Path to save checkpoints.')
    parser.add_argument('--device_target', default='GPU', help='device target')
    parser.add_argument('--is_distributed', default=0, help='distributed train')
    parse_args = parser.parse_args()
    run_train(parse_args)
