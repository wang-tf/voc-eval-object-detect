#!/user/bin/env python3
# -*- coding:utf-8 -*-

import os
import argparse
import mmcv
import cv2
import tqdm

# voc_root_path = '/diskb/GlodonDataset/rebar-steelpipe/v0.1/test/VOC2007'
voc_root_path = '/diskb/GlodonDataset/rebar-steelpipe/v0.2/test/VOC2007'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str)
    parser.add_argument('--conf_thresh', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./detect_results')
    args = parser.parse_args()
    return args


def convert(det_results, main_set_file, label_list, conf_thresh, save_dir):
    image_id_list = mmcv.list_from_file(main_set_file)

    for image_id, image_result in tqdm.tqdm(zip(image_id_list, det_results)):
        image_path = os.path.join(os.path.dirname(main_set_file), '../../JPEGImages', image_id+'.jpg')
        assert os.path.exists(image_path), image_path
        
        image = cv2.imread(image_path)
        for label, category_result in zip(label_list, image_result):
            for result in category_result:
                xmin, ymin, xmax, ymax, score = result
                if score < conf_thresh:
                    continue
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                cv2.putText(image, '{}:{:.3f}'.format(label, score), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        save_path = os.path.join(save_dir, image_id+'.jpg')
        cv2.imwrite(save_path, image)



def main():
    args = get_args()
    result_file = args.result
    det_results = mmcv.load(result_file)

    label_list = ['SteelPipe', 'rebar']
    
    main_set_file = os.path.join(voc_root_path, 'ImageSets/Main/test.txt')
    assert os.path.exists(main_set_file), main_set_file

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    convert(det_results, main_set_file, label_list, args.conf_thresh, args.save_dir)

    return


if __name__ == '__main__':
    main()

