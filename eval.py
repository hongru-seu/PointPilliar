"""Evaluation script"""

import argparse
import os
import warnings
from time import time

import sys 
sys.path.append("..")

from mindspore import context
from mindspore import dataset as de
from mindspore import load_checkpoint
from mindspore import load_param_into_net

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])+"/models")
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.core.eval_utils import get_official_eval_result
from src.predict import predict
from src.predict import predict_kitti_to_anno
from src.utils import get_config
from src.utils import get_model_dataset
from src.utils import get_params_for_net

warnings.filterwarnings('ignore')


def run_evaluate(args):
    """run evaluate"""
    print('run evaluate')
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path 

    cfg = get_config(cfg_path)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_target = args.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

    model_cfg = cfg['model']

    center_limit_range = model_cfg['post_center_limit_range']

    pointpillarsnet, eval_dataset, box_coder = get_model_dataset(cfg, False)

    params = load_checkpoint(ckpt_path)
    new_params = get_params_for_net(params)
    load_param_into_net(pointpillarsnet, new_params)

    eval_input_cfg = cfg['eval_input_reader']

    eval_column_names = eval_dataset.data_keys

    ds = de.GeneratorDataset(
        eval_dataset,
        column_names=eval_column_names,
        python_multiprocessing=True,
        num_parallel_workers=3,
        max_rowsize=100,
        shuffle=False
    )
    batch_size = eval_input_cfg['batch_size']
    ds = ds.batch(batch_size, drop_remainder=False)
    data_loader = ds.create_dict_iterator(num_epochs=1)

    class_names = list(eval_input_cfg['class_names'])

    dt_annos = []
    gt_annos = [info["annos"] for info in eval_dataset.kitti_infos]

    log_freq = 100
    len_dataset = len(eval_dataset)
    print('len_dataset',len_dataset)
    start = time()
    for i, data in enumerate(data_loader):
        voxels = data["voxels"]
        num_points = data["num_points"]
        coors = data["coordinates"]
        bev_map = data.get('bev_map', False)

        preds = pointpillarsnet(voxels, num_points, coors, bev_map)
        if len(preds) == 2:
            preds = {
                'box_preds': preds[0],
                'cls_preds': preds[1],
            }
        else:
            preds = {
                'box_preds': preds[0],
                'cls_preds': preds[1],
                'dir_cls_preds': preds[2]
            }
        preds = predict(data, preds, model_cfg, box_coder)

        dt_annos += predict_kitti_to_anno(preds,
                                          data,
                                          class_names,
                                          center_limit_range)

        if i % log_freq == 0 and i > 0:
            time_used = time() - start
            print(f'processed: {i * batch_size}/{len_dataset} imgs, time elapsed: {time_used} s',
                  flush=True)

    result = get_official_eval_result(
        gt_annos,
        dt_annos,
        class_names,
    )
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='/data0/HR_dataset/4_hong/MINDSPORE/ms3d-GAI/example/point_pillars/experiments/cars/car_xyres16.yaml', help='Path to config file.')
    parser.add_argument('--ckpt_path', default='/data0/HR_dataset/4_hong/MINDSPORE/ms3d-GAI/example/point_pillars/experiments/cars/pointpillars-160_296960.ckpt', help='Path to checkpoint.')
    parser.add_argument('--device_target', default='GPU', help='device target')

    parse_args = parser.parse_args()

    run_evaluate(parse_args)
