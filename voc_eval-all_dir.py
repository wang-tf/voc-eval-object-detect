from argparse import ArgumentParser
import os
import mmcv
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import datasets
from mean_ap import eval_map
from class_names import get_classes


def draw_figure_3d(step, class_y_map, class_z_map, ylabel=''):
    fig = plt.figure()
    ax = Axes3D(fig)

    # mean y for all category
    mean_y = []
    for label in class_y_map.keys():
        # TODO
        x = []
        y = []
        z = []
        for step_x, step_y, step_z in zip(step, class_y_map[label], class_z_map[label]):
            filted_y = []
            filted_z = []
            for index in range(step_y.shape[0]):
                if step_z[index] not in filted_z:
                    filted_z.append(step_z[index])
                    filted_y.append(step_y[index])
                else:
                    if step_y[index] > filted_y[-1]:
                        filted_y[-1] = step_y[index]

            x += [step_x] * len(filted_y)
            y += filted_y
            z += filted_z

        #shape_list = [class_y_map[label][i].shape[0] for i in range(len(class_y_map[label]))]
        #x = np.array(step).repeat(shape_list)
        #y = np.hstack(class_y_map[label])
        #z = np.hstack(class_z_map[label])

        # print top 10
        sort_index = np.argsort(-np.array(y))
        sort_x = np.array(x)[sort_index]
        sort_y = np.array(y)[sort_index]
        sort_z = np.array(z)[sort_index]

        # ax.scatter(x, y, z, label=label)

    # ax.legend(loc='best')
    # ax.set_xlabel('step')
    # ax.set_ylabel(ylabel)
    # ax.set_zlabel('conf')
    # plt.show()


def print_top_n(step, class_y_map, class_z_map, ylabel='', topn=10):
    print('------------ {} ----------'.format(ylabel))
    split_num = 20
    filted_norm_z = np.arange(0, 1, 1/split_num)
    mean_y_xyz = np.zeros((len(step) * len(filted_norm_z), 3))
    mean_y_xyz[:, 2] = 1

    for label in class_y_map.keys():
        # TODO
        xyz_step = []
        for index, (y_step, z_step) in enumerate(zip(class_y_map[label], class_z_map[label])):
            x_step = np.array([step[index] for _ in range(split_num)]).reshape((split_num, 1))
            assert y_step.shape == z_step.shape
            yz_step = np.stack((y_step, z_step), axis=-1)  # original yz data
            # sorted by z and splited
            # yz_step = np.argsort(-yz_step[:, 1])
            filted_yz_step = np.zeros((split_num, 2))
            filted_yz_step[:, 1] = 1

            for yz in yz_step:
                for index in range(split_num):
                    if yz[1] >= filted_norm_z[index] and yz[1] < filted_norm_z[index] + 1/split_num:
                        if yz[1] < filted_yz_step[index, 1]:
                            filted_yz_step[index] = yz

            sorted_y_step_index = np.argsort(-filted_yz_step[:, 0])
            # print('\n'.join(['step {}: top {}: {} {}, score {}'.format(step[index], top_index, ylabel, filted_yz_step[sorted_y_step_index[top_index], 0], filted_yz_step[sorted_y_step_index[top_index], 1]) for top_index in range(topn)]))

            xyz_step += [np.concatenate((x_step, filted_yz_step), axis=1)]

        xyz = np.vstack(xyz_step)
        sorted_y_index = np.argsort(-xyz[:, 1])
        for index in range(topn):
            print('{} top {}: {} {}, step {}, score {}'.format(label, index, ylabel, xyz[sorted_y_index[index], 1], xyz[sorted_y_index[index], 0], xyz[sorted_y_index[index], 2]))

        mean_y_xyz[:, 0] = xyz[:, 0]
        mean_y_xyz[:, 1] += xyz[:, 1]
        mean_y_xyz[:, 2] = np.minimum(mean_y_xyz[:, 2], xyz[:, 2])
       

    mean_y_xyz[:, 1] /= len(class_y_map.keys())
    sorted_y_index = np.argsort(-mean_y_xyz[:, 1])
    for index in range(topn):
        print('m{} top {}: {}, step {}, score {}'.format(ylabel, index, mean_y_xyz[sorted_y_index[index], 1], mean_y_xyz[sorted_y_index[index], 0], mean_y_xyz[sorted_y_index[index], 2]))


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
    return eval_results


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result_dir', help='result dir including inference_*/result_test.json')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--iou-thr', type=float, default=0.35, help='IoU threshold for evaluation')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold for evaluation')
    parser.add_argument('--nproc', type=int, default=4, help='Processes to be used for computing mAP')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    data_test = cfg.data.test

    test_dataset = mmcv.runner.obj_from_dict(data_test, datasets)
    
    result_list = sorted(glob.glob(os.path.join(args.result_dir, '*/result_test.json')), key=lambda x: int(os.path.basename(os.path.dirname(x)).split('-')[1]))

    if hasattr(test_dataset, 'year') and test_dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = test_dataset.CLASSES
    label_names = get_classes(dataset_name)
    
    step_list = []
    class_f1_map = {key: [] for key in label_names}
    class_yewu2_map = {key: [] for key in label_names}

    label_ap_map = {key: [] for key in label_names}
    label_f1_map = {key: [] for key in label_names}
    label_yewu2_map = {key: [] for key in label_names}
    label_score_map = {key: [] for key in label_names}

    for r_i, result in enumerate(result_list):
        # 1. eval one result
        step = int(os.path.dirname(result).split('/')[-1].split('-')[-1])
        # if step < 10000 or step >  60000:
        #     continue
        print(result)
        try:
            eval_results = voc_eval(result, test_dataset, args.iou_thr, args.nproc, args.conf_thresh)
        except Exception as e:
            # print(e)
            continue

        step_list.append(step)

        if isinstance(eval_results[0]['ap'], np.ndarray):
            num_scales = len(eval_results[0]['ap'])
        else:
            num_scales = 1
        num_classes = len(eval_results)

        max_f1 = np.zeros((num_scales, num_classes), dtype=np.float32)
        max_yewu2 = np.zeros((num_scales, num_classes), dtype=np.float32)
        for class_index, cls_result in enumerate(eval_results):
            if cls_result['recall'].size > 0:
                max_f1[:, class_index] = np.array(cls_result['f1'], ndmin=2)[:, -1]
                max_yewu2[:, class_index] = np.array(cls_result['yewu2'], ndmin=2)[:, -1]

        if len(max_f1) == 1:
            for j in range(num_classes):
                class_f1_map[label_names[j]].append(max_f1[0, j])
                class_yewu2_map[label_names[j]].append(max_yewu2[0, j])
        else:
            print('WARNING: current only accept num_scales == 1')

        # temp, using to add score list
        for class_index, cls_result in enumerate(eval_results):
            if cls_result['recall'].size > 0:
                score = np.array(cls_result['scores'], ndmin=2)
                ap = np.array(cls_result['ap'], ndmin=2)
                f1 = np.array(cls_result['f1'], ndmin=2)
                yewu2 = np.array(cls_result['yewu2'], ndmin=2)
            if len(f1) == 1:
                label_ap_map[label_names[class_index]].append(ap[0])
                label_f1_map[label_names[class_index]].append(f1[0])
                label_yewu2_map[label_names[class_index]].append(yewu2[0])
                label_score_map[label_names[class_index]].append(score[0])

        # for debug
        # if r_i > 10:
        #     break

    # draw 3d figure
    # draw_figure_3d(step_list, label_f1_map, label_score_map, ylabel='F1')

    # print
    print_top_n(step_list, label_f1_map, label_score_map, ylabel='F1')
    print_top_n(step_list, label_yewu2_map, label_score_map, ylabel='Yewu2')

    # save result to csv
    class_f1_map['step'] = step_list
    dataframe = pd.DataFrame(class_f1_map)
    dataframe.to_csv('F1_result.csv', index=None)
    step_list = class_f1_map.pop('step')
    class_yewu2_map['step'] = step_list
    dataframe = pd.DataFrame(class_yewu2_map)
    dataframe.to_csv('Yewu2_result.csv', index=None)
    step_list = class_yewu2_map.pop('step')

    y_label = 'F1'
    save_path = './F1_result.png'
    plt.figure(figsize=(10, 5))
    plt.title('{} result analyse'.format(y_label))
    plt.xlabel('step')
    plt.ylabel(y_label)
    lines = []
    for label, f1_list in class_f1_map.items():
        x_list = [int(n) for n in step_list]
        x_y = [[i,j] for i,j in zip(x_list, f1_list)]
        sorted_x_y = sorted(x_y, key=lambda x: x[0])
        new_x, new_y = [], []
        for x_y in sorted_x_y:
            new_x.append(x_y[0])
            new_y.append(x_y[1])

        line = plt.plot(new_x, new_y)
        lines.append(line)
    plt.legend(lines, labels=label_names, loc='best')
    plt.savefig(save_path)


if __name__ == '__main__':
    main()
