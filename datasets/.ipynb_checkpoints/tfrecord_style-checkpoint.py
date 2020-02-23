#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path as osp
import tensorflow as tf
from PIL import Image

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class TFRecordDataset(CustomDataset):
    def __init__(self, min_size=None, **kwargs):
        super(TFRecordDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i+1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.filename_key = 'image/filename'
        self.image_key = 'image/encoded'
        self.bbox_name_key = 'image/object/class/text'
        self.coordinates_in_pixels = False
        self.bboxes = []
        self.labels = []
        self.bboxes_ignore = []
        self.labels_ignore = []

    def load_annotations(self, tfrecord_path):
        img_infos = []
        for i, record in enumerate(tf.python_io.tf_record_iterator(tfrecord_path)):
            example = tf.train.Example()
            example.ParseFromString(record)
            feat = example.features.feature

            filename = feat[self.filename_key].bytes_list.value[0].decode("utf-8")
            img_id = osp.basename(filename).split('.')[0]
            img =  Image.open(feat[self.image_key].bytes_list.value[0])
            width, height = img.size

            bboxes = []
            if self.bbox_name_key in feat:
                for ibbox, label in enumerate (feature[self.bbox_name_key].bytes_list.value):
                    name = label.decode('utf-8')
                    label = self.cat2lable[name]
                    difficult = 0  # TODO: need load from tfrecord
                    bbox = [
                        feature[self.args.bbox_xmin_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymin_key].float_list.value[ibbox],
                        feature[self.args.bbox_xmax_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymax_key].float_list.value[ibbox]
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
                        self.bboxes_ignore.append(bbox)
                        self.labels_ignore.append(label)
                    else:
                        self.bboxes.append(bbox)
                        self.labels.append(label)

            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

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
        
        bboxes = self.bboxes
        labels = self.labels
        bboxes_ignore = self.bboxes_ignore
        labels_ignore = self.labels_ignore
        
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
