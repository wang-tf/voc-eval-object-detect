#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path as osp
import tensorflow as tf
from PIL import Image
import io
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format


@DATASETS.register_module
class TFRecordDataset(CustomDataset):
    def __init__(self, label_map_path, min_size=None, **kwargs):
        self.cat2label = self.load_classes(label_map_path)
        self.CLASSES = list(self.cat2label.keys())
        self.min_size = min_size
        self.filename_key = 'image/filename'
        self.image_key = 'image/encoded'
        self.bbox_name_key = 'image/object/class/text'
        self.bbox_xmin_key = 'image/object/bbox/xmin'
        self.bbox_ymin_key = 'image/object/bbox/ymin'
        self.bbox_xmax_key = 'image/object/bbox/xmax'
        self.bbox_ymax_key = 'image/object/bbox/ymax'
        self.coordinates_in_pixels = False
        self.bboxes = []
        self.labels = []
        self.bboxes_ignore = []
        self.labels_ignore = []
        super(TFRecordDataset, self).__init__(**kwargs)


    def load_classes(self, label_map_path, use_display_name=False):
        with open(label_map_path) as f:
            label_map_string = f.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)

        label_map_dict = {}
        for item in label_map.item:
            if use_display_name:
                label_map_dict[item.display_name] = item.id
            else:
                label_map_dict[item.name] = item.id
        return label_map_dict

    def load_annotations(self, tfrecord_path):
        img_infos = []
        for i, record in enumerate(tf.python_io.tf_record_iterator(tfrecord_path)):
            example = tf.train.Example()
            example.ParseFromString(record)
            feat = example.features.feature

            filename = feat[self.filename_key].bytes_list.value[0].decode("utf-8")
            img_id = osp.basename(filename).split('.')[0]
            img =  Image.open(io.BytesIO(feat[self.image_key].bytes_list.value[0]))
            width, height = img.size

            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            if self.bbox_name_key in feat:
                for ibbox, label in enumerate (feat[self.bbox_name_key].bytes_list.value):
                    name = label.decode('utf-8')
                    label = self.cat2label[name]
                    difficult = 0  # TODO: need load from tfrecord
                    bbox = [
                        feat[self.bbox_xmin_key].float_list.value[ibbox],
                        feat[self.bbox_ymin_key].float_list.value[ibbox],
                        feat[self.bbox_xmax_key].float_list.value[ibbox],
                        feat[self.bbox_ymax_key].float_list.value[ibbox]
                        ]
                    bbox = self.bboxes_to_pixels(bbox, width, height)
                    ignore = False
                    if self.min_size:
                        assert not self.test_mode
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        if w < self.min_size or h < self.min_size:
                            ignore = True
                    if difficult or ignore:
                        bboxes_ignore.append(bbox)
                        labels_ignore.append(label)
                    else:
                        bboxes.append(bbox)
                        labels.append(label)

            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
            self.bboxes.append(bboxes)
            self.labels.append(labels)
            self.bboxes_ignore.append(bboxes_ignore)
            self.labels_ignore.append(labels_ignore)

        return img_infos

    def bboxes_to_pixels(self, bbox, im_width, im_height):
        """
        Convert bounding box coordinates to pixels.
        (It is common that bboxes are parametrized as percentage of image size
        instead of pixels.)

        Args:
            bboxes (tuple): (xmin, ymin, xmax, ymax)
            im_width (int): image width in pixels
            im_height (int): image height in pixels
                                                
        Returns:
            bboxes (tuple): (label, xmin, ymin, xmax, ymax)
        """
        if self.coordinates_in_pixels:
            return bbox
        else:
            xmin, ymin, xmax, ymax = bbox
        return [xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height]

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]
        bboxes_ignore = self.bboxes_ignore[idx]
        labels_ignore = self.labels_ignore[idx]
        
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann
