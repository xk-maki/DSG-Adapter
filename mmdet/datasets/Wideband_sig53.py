# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from typing import List, Union

import numpy as np
import mmcv
from pycocotools.coco import COCO
from mmcv.utils import print_log
import logging

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Wideband_sig53Dataset(CustomDataset):
    """Dataset for Wideband sig53 (COCO style)."""

    CLASSES = ('ask', 'fsk', 'ofdm', 'pam', 'psk', 'qam')
    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = self.COCOAPI(ann_file)
        # 原生COCO API：getCatIds(catNms=[类别名列表])
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # 原生COCO API：getImgIds()获取所有图片ID
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            # 原生COCO API：loadImgs([imgId])
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            # 原生COCO API：getAnnIds(imgIds=[imgId])
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        # 原生COCO API：getAnnIds(imgIds=[imgId])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # 原生COCO API：loadAnns([annId])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # Obtain images that contain annotations
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # Obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for class_id in self.cat_ids:
            # 原生COCO API：getImgIds(catIds=[class_id])获取指定类别图片ID
            ids_in_cat.update(self.coco.getImgIds(catIds=[class_id]))
        # Merge the image id sets
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: Parsed annotation info.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        # Convert to numpy arrays
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')  # 可根据实际情况调整

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style."""
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results, outfile_prefix):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        outfile = f'{outfile_prefix}.bbox.json'
        mmcv.dump(json_results, outfile)
        return outfile

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol."""
        from pycocotools.cocoeval import COCOeval
        from terminaltables import AsciiTable
        import itertools

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        # Format results and save to json files
        if jsonfile_prefix is None:
            jsonfile_prefix = osp.join(tempfile.gettempdir(), 'results')
        result_files = self.results2json(results, jsonfile_prefix)
        if not isinstance(result_files, dict):
            result_files = {metric: result_files}

        eval_results = dict()
        cocoGt = self.coco
        for metric in metrics:
            print_log(f'Evaluating {metric}...', logger=logger)

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                return eval_results

            iou_type = metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs

            # Mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            if classwise:
                # Compute per-category AP
                precisions = cocoEval.eval['precision']
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # 原生COCO API：loadCats([catId])获取类别名称
                    nm = self.coco.loadCats([catId])[0]['name']
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    ap = np.mean(precision) if precision.size else float('nan')
                    results_per_category.append((f'{nm}', f'{ap:.3f}'))

                # Print per-category AP table
                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers] + [list(r) for r in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            # Save evaluation results
            if metric_items is None:
                metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

            for item in metric_items:
                if item not in coco_metric_names:
                    continue
                key = f'{metric}_{item}'
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[key] = val

            # Save mAP_copypaste for easy result comparison
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results