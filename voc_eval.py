#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from argparse import ArgumentParser

import mmcv
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

import datasets
from mean_ap import eval_map


def obj_from_tfrecord(data_test):
    return


def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4, conf_thresh=0.5, show=False):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    mean_ap, eval_results = eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc,
        conf_thresh=conf_thresh,
        show=show)
    return mean_ap, eval_results

def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--iou-thr', type=float, default=0.35, help='IoU threshold for evaluation')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='IoU threshold for evaluation')
    parser.add_argument('--nproc', type=int, default=4, help='Processes to be used for computing mAP')
    parser.add_argument('--show', action='store_true', default=False, help='')
    parser.add_argument('--backend', type=str, default='mmdetection', help='chonse a backend in [tensorflow, mmdetection]')
    args = parser.parse_args()

    if args.backend == 'tensorflow':
        # TODO
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(args.config) as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)
        cfg = pipeline_config
        data_test = cfg.eval_input_reader[0].tf_record_input_reader  # only load first dataset
        label_map_path = cfg.eval_input_reader[0].label_map_path
        pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'MultiScaleFlipAug', 'img_scale': (1333, 1000), 'flip': False, 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'ImageToTensor', 'keys': ['img']}, {'type': 'Collect', 'keys': ['img']}]}]
        data_test = dict(type='TFRecordDataset', ann_file=cfg.eval_input_reader[0].tf_record_input_reader.input_path[0], pipeline=pipeline, label_map_path=label_map_path)
        test_dataset = obj_from_tfrecord(data_test)

    elif args.backend == 'mmdetection':
        cfg = mmcv.Config.fromfile(args.config)
        data_test = cfg.data.test

        print(data_test['pipeline'])

    else:
        print(f'Can not find this backend: {args.backend}')
        raise

    test_dataset = mmcv.runner.obj_from_dict(data_test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr, args.nproc, args.conf_thresh, show=args.show)


if __name__ == '__main__':
    main()

