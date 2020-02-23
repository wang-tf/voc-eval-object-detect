#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import json
import mmcv


def load_comp4(comp4_path):
    assert os.path.exists(comp4_path), comp4_path

    with open(comp4_path) as f:
        lines = f.read().strip().split('\n')

    image_id_bbox_map = {}
    for line in lines:
        image_id, score, x1, y1, x2, y2 = line.split(' ')
        
        if image_id not in image_id_bbox_map:
            image_id_bbox_map[image_id] = []
        else:
            image_id_bbox_map[image_id].append([float(x1), float(y1), float(x2), float(y2), float(score)])

    return image_id_bbox_map


def main():
    labels_list = ['SteelPipe', 'rebar']
    voc_test_file = '/diskb/GlodonDataset/rebar-steelpipe/v0.1/test-rebar/VOC2007/ImageSets/Main/test.txt'
    image_id_list = mmcv.list_from_file(voc_test_file)

    category_comp4_data_map = {key: {} for key in labels_list}

    comp4_path = '/data/wangtf/Projects/darknet-AlexeyAB/work_dirs/results/comp4_det_test_rebar.txt'
    category = os.path.basename(comp4_path).split('.')[0].split('_')[-1]
    image_id_bbox_map = load_comp4(comp4_path)
    category_comp4_data_map[category] = image_id_bbox_map

    json_data = []
    for image_id in image_id_list:
        one_image_json_data = []
        for label in labels_list:
            if image_id not in category_comp4_data_map[label]:
                one_image_json_data.append([])
            else:
                one_image_json_data.append(category_comp4_data_map[label][image_id])
        json_data.append(one_image_json_data)

    save_path = os.path.dirname(comp4_path) + '/result_test.json'
    with open(save_path, 'w') as f:
        json.dump(json_data, f)


if __name__=='__main__':
    main()
