#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import cv2
import json
from pascal_voc_tools import XmlParser

from bbox_overlaps import bbox_overlaps 


def load_voc_result(main_path, json_path):
  assert os.path.exists(main_path), main_path
  with open(main_path) as f:
    image_ids = f.read().strip().split('\n')

  assert os.path.exists(json_path), json_path
  with open(json_path) as f:
    json_result = json.load(f)

  return image_ids, json_result


def load_label():
  return ['mask', 'nomask']

label_num = {'mask': 0, 'nomask': 0}
color_map = {'mask':(0, 255, 0), 'nomask': (0, 0, 255)}
def draw_image(image, dets, labels):
  for label, det in zip(labels, dets):
    color = color_map[label]
    for bbox in det:
      x1, y1, x2, y2, s = bbox
      x1 = int(x1)
      y1 = int(y1)
      x2 = int(x2)
      y2 = int(y2)
      cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
      label_num[label] += 1

def load_gt(image_path):
  image_id = os.path.basename(image_path).split('.')[0]
  xml_dir = os.path.join(os.path.dirname(image_path), '../Annotations')
  xml_path = os.path.join(xml_dir, image_id+'.xml')
  assert os.path.exists(xml_path), xml_path

  xml_data = XmlParser().load(xml_path)
  return xml_data['object']
      

def draw_gt(image, gt):
  for bbox in gt:
    label = bbox['name']
    x1 = int(bbox['bndbox']['xmin'])
    y1 = int(bbox['bndbox']['ymin'])
    x2 = int(bbox['bndbox']['xmax'])
    y2 = int(bbox['bndbox']['ymax'])
    if label == 'mask':
      cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0), 2)
    elif label == 'nomask':
      cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    else:
      print(label)


def draw_result(main_path, json_path, save_dir):
  image_ids, json_result = load_voc_result(main_path, json_path)

  jpg_dir = os.path.join(os.path.dirname(main_path), '../../JPEGImages')
  for image_id, image_result in zip(image_ids, json_result):
    image_path = os.path.join(jpg_dir, image_id+'.jpg')
    assert os.path.exists(image_path), image_path
    image = cv2.imread(image_path)

    labels = load_label()
    draw_image(image, image_result, labels)

    gt = load_gt(image_path)
    draw_gt(image, gt)

    save_path = os.path.join(save_dir, image_id+'.jpg')
    cv2.imwrite(save_path, image)
  print(label_num)


def main():
  main_path = '/diskb/GlodonDataset/Mask/test-20200229/VOC2007/ImageSets/Main/test.txt'
  json_path = '/diska/wangtf/service-nfs/Projects/Mask-Classifier/data/test-20200229/result.json'
  save_dir = './test-20200229'

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  draw_result(main_path, json_path, save_dir)


if __name__ == '__main__':
  main()

