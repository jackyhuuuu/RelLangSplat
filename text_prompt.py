#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
import gc
import re
import scipy.ndimage
import sys

sys.path.append("")
from eval import colormaps
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
from relationship.interpreter import Box

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def calculate_iou(box1, box2):
    x1_max, y1_max, x1_min, y1_min = box1[2], box1[3], box1[0], box1[1]
    x2_max, y2_max, x2_min, y2_min = box2[2], box2[3], box2[0], box2[1]

    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)

    if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
        return 0.0

    intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersect_area

    # handle the case when one box is surrounded by another
    if union_area > 0.9 * box1_area or union_area > 0.9 * box2_area:
        return 1

    return intersect_area / union_area


def detect_bbox(n_head, avg_filtered, score_threshold=0.6, iou_threshold=0.2):
    score_lvl = np.zeros((n_head,))
    coord_lvl = []
    bounding_boxes = []
    # def area_box(boxes):
    #     return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    for i in range(n_head):
        score = avg_filtered[..., i].max()
        coord = np.nonzero(avg_filtered[..., i] == score)
        score_lvl[i] = score
        coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

    selec_head = np.argmax(score_lvl)
    coord_final = coord_lvl[selec_head]

    # obj_threshold=0.55
    # max_score = np.max(score_lvl, axis=0)
    # if max_score < obj_threshold:
    #     return bounding_boxes, coord_final, selec_head
    # else:
    #     score_threshold = score_threshold if max_score > score_threshold else obj_threshold

    for i in range(n_head):
        # 找到所有大于阈值的点
        binary_map = avg_filtered[..., i] >= score_threshold
        # 进行连通域分析
        labeled_array, num_features = scipy.ndimage.label(binary_map)
        for label in range(1, num_features + 1):
            # 获取当前label的坐标
            coords = np.column_stack(np.nonzero(labeled_array == label))
            if coords.size > 0:
                # 计算bounding box
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                # 保存bounding box
                bounding_boxes.append((min_x, min_y, max_x, max_y))

    bounding_boxes = np.array(bounding_boxes)

    # 标记数组，标记是否保留某个 bbox
    keep_bbox = np.ones(len(bounding_boxes), dtype=bool)
    for i in range(len(bounding_boxes)):
        if not keep_bbox[i]:
            continue
        for j in range(i + 1, len(bounding_boxes)):
            if not keep_bbox[j]:
                continue
            box1 = bounding_boxes[i]
            box2 = bounding_boxes[j]
            iou = calculate_iou(box1, box2)

            if iou > iou_threshold:
                area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if area_box1 > area_box2:
                    keep_bbox[j] = False
                else:
                    keep_bbox[i] = False
                    break

    bounding_boxes = bounding_boxes[keep_bbox]

    return bounding_boxes, coord_final, selec_head
    # try:
    #     h, w = avg_filtered.shape[:2]
    #     area = h * w
    #     if bounding_boxes.size != 0:
    #         bounding_boxes = bounding_boxes[area_box(bounding_boxes) > 0.001 * area]
    #         return bounding_boxes, coord_final, selec_head
    # except Exception as e:
    #     # Print the exception message if no valid bounding box is found
    #     print(f"No valid bounding box found: {e}")


def activate_stream(sem_map,
                    image,
                    clip_model,
                    image_name: Path = None,
                    img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None):
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)

            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_lvl_list


def lerf_localization(sem_map, image, clip_model, image_name, text_prompts):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    positives = list(text_prompts)
    results = []
    for k in range(len(positives)):
        select_output = valid_map[:, k]

        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

        bounding_boxes, coord_final, selec_head = detect_bbox(n_head, avg_filtered)
        if len(bounding_boxes) == 0:
            return None

        updated_bounding_boxes = []
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            width, height = x_max - x_min, y_max - y_min
            results.append(Box(obj_name=positives[k], x=x_min, y=y_min, w=width, h=height))
            bbox = np.array([[x_min, y_min, width, height]])
            updated_bounding_boxes.append(bbox)

        bounding_boxes = np.array(updated_bounding_boxes)
        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3

        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    bounding_boxes, save_path)

    return results


def evaluate_with_prompt(text_prompt, image_path, feat_dir, output_path, ae_ckpt_path, mask_thresh, encoder_hidden_dims,
                         decoder_hidden_dims):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    if orig_h > 1080:
        global_down = orig_h / 1080
    else:
        global_down = 1

    scale = float(global_down)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    image = cv2.resize(image, resolution)
    h, w = image.shape[:2]
    image_shape = (h, w)

    filename = os.path.basename(image_path)  # 获取文件名部分："01.JPG"
    index_str = os.path.splitext(filename)[0]  # 去除扩展名后的部分："01"
    image_index = int(index_str[-4:]) - 1  # 转换为整数

    eval_index_list = list([image_index])
    compressed_sem_feats = np.zeros((len(feat_dir), 1, *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # for j, idx in enumerate(tqdm(eval_index_list)):
    for j, idx in enumerate(eval_index_list):
        image_name = Path(output_path) / f'{idx + 1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_path)[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512

        clip_model.set_positives(list(text_prompt))

        bounding_boxes = lerf_localization(restored_feat, rgb_img, clip_model, image_name, text_prompt)
        if bounding_boxes == None:
            return None

        #activate_stream(restored_feat, rgb_img, clip_model, image_name, thresh=mask_thresh, colormap_options=colormap_options)

        return bounding_boxes


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def image_load(image_path):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    if orig_h > 1080:
        global_down = orig_h / 1080
    else:
        global_down = 1

    scale = float(global_down)
    resolution = (int(orig_w / scale), int(orig_h / scale))
    image = cv2.resize(image, resolution)

    return image

if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = ArgumentParser(description="prompt any label")
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()

    dataset_name = "teatime"
    feat_dir = "output"
    ae_ckpt_dir = "autoencoder/ckpt"
    output_dir = "eval_result"
    mask_thresh = 0.4

    feat_dir = [os.path.join(feat_dir, dataset_name + f"_{i}", "train/ours_None/renders_npy") for i in range(1, 4)]
    output_path = os.path.join(output_dir, dataset_name)
    ae_ckpt_path = os.path.join(ae_ckpt_dir, dataset_name, "best_ckpt.pth")

    image_path = "lerf_ovs/teatime/images/frame_00050.jpg"

    text_prompt = ["white sheep"]
    bounding_boxes = evaluate_with_prompt(text_prompt, image_path, feat_dir, output_path, ae_ckpt_path, mask_thresh, args.encoder_dims, args.decoder_dims)
    print(bounding_boxes)





