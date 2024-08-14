import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    # OpenCLIP 使用的均值和标准差是通过计算 LAION-2B 数据集的统计值得到的
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property # 用@property将name转换为只读属性，可通过OpenCLIPNetwork.name访问该方法的返回值
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property # 用@property将embedding_dim转换为只读属性，可通过OpenCLIPNetwork.embedding_dim访问该方法的返回值
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self, element):
        # 假设用户在GUI界面输入 cat;dog;car 点击按钮会触发 gui_cb的方法将输入的字符串拆分成列表 ["cat", "dog", "car"]
        # 并调用 self.set_positives(["cat", "dog", "car"])
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # 函数输入为 embed 输入的嵌入张量和 positive_id 正面样本的索引
        # 评估输入嵌入与不同样本的相关性，并确定最不相关的正面样本
        # 输入的正面样本id应该与输入嵌入的真实id对应，以确保相关性值的正确性

        # 将正面样本嵌入 self.pos_embeds 和负面样本嵌入 self.neg_embeds 沿第一个维度拼接，得到 phrases_embeds
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        # 将拼接后的嵌入张量 phrases_embeds 转换为与输入嵌入 embed 相同的数据类型
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        # 计算输入嵌入 embed 和拼接后的嵌入 p 的点积，得到 output，表示输入嵌入与每个正面和负面样本的相关性
        output = torch.mm(embed, p.T)  # rays x phrases
        # 从 output 中提取与正面样本相关的相关性值
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        # 从 output 中提取与负面样本相关的相关性值
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        # 将正面相关性值 positive_vals 沿第二个维度重复，得到 repeated_pos，使其与负面样本的数量相匹配
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase
        # 将 repeated_pos 和 negative_vals 沿新维度堆叠，得到 sims
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        # 对 sims 乘以10进行放缩，并沿最后一个维度应用softmax函数
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        # 从softmax结果中，找到第一个维度最小的索引 best_id，表示最不相关的正面样本索引
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
        # 举个例子说明 return 结果
        # softmax 的形状为 (2, 3, 2)，即2个输入嵌入，3个样本（正面或负面），2个相关性分布
        # best_id 为 [1, 0]，表示第一个输入嵌入的最不相关正面样本的索引是1，第二个输入嵌入的最不相关正面样本的索引是0
        # best_id[..., None, None] 变为 [[[1]], [[0]]]，形状为 (2, 1, 1)
        # best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2)
        # 变为 [[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]]]，形状为 (2, 3, 2)
        # torch.gather(softmax, 1, ...) 根据索引在维度1上提取元素，结果形状为 (2, 3, 2)
        # [:, 0, :] 提取第一个维度，结果形状为 (2, 2)

    def encode_image(self, input):
        # half这一步将预处理后的图像张量转换为半精度浮点数格式（half precision），即 float16 类型
        # 半精度浮点数可以节省内存并提高计算速度，尤其是在GPU上
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size = 512
    seg_maps = []
    total_lengths = []
    timer = 0
    # 300？，初始化为0，用来储存CLIP嵌入
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    # 4表示四种尺寸的分割地图，*image_list[0].shape[1:] 表示图像的尺寸，取第一张图像的形状，排除了批次维度
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)
        # lengths 列表收集了每个键对应值的长度
        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            # 计算需要填充的数量
            pad = total_length - img_embeds.shape[1]
            # 初始化要填充的部分
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        # 对img_embeds进行填充
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        # 计算累计长度
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]

        for j, (k, v) in enumerate(seg_map.items()):
            # if j == 0:：如果是第一个键值对，直接将其转换为张量并添加到 seg_map_tensor 中
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            # 确保分割地图的最大值等于当前长度减1。这是为了确保分割地图的索引范围在合理的范围内
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            # 将分割地图中不为-1的值（有效的分割区域）加上累积长度，以确保分割地图的索引正确对应扩展后的 img_embeds 张量
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        # 将 seg_map_tensor 中的张量沿着新创建的维度0进行堆叠，以生成最终的分割地图张量
        seg_map = torch.stack(seg_map_tensor, dim=0)
        # 将分割地图张量存储在 seg_maps 中的第 i 个位置，即当前图像的索引位置
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)


def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())


def _embed_clip_sam_tiles(image, sam_encoder):
    # 将输入的图像进行分割，并为每个分割区域生成相应的 CLIP 嵌入，最后返回这些嵌入以及分割地图
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map


def get_seg_img(mask, image):
    # 复制输入图像，以避免直接修改原始图片
    image = image.copy()
    # 根据掩码中的分割区域，将图像中对应的像素值置为全零，即黑色。这样可以将掩码区域以外的部分置为黑色
    image[mask['segmentation'] == 0] = np.array([0, 0,  0], dtype=np.uint8)
    # 获取掩码对应的边界框的坐标和尺寸
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    # 根据边界框的坐标和尺寸，从图像中提取出与掩码对应的分割区域
    return seg_img


def pad_img(img):
    # pad_img 函数的主要作用是将输入的图像填充成正方形，以便后续处理
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    # 计算每个掩码的面积，通过对掩码张量的高度和宽度进行求和得到
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
    # 初始化一个全零的张量，形状为 (num_masks, num_masks)，用于存储计算得到的IoU矩阵
    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    # 初始化一个全零的张量，形状与 iou_matrix 相同，用于存储计算得到的内部IoU矩阵
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    # 循环遍历所有可能的掩码对，但由于IoU矩阵是对称的，因此只需要计算上半部分
    for i in range(num_masks):
        for j in range(i, num_masks):
            # 计算两个掩码的交集。torch.logical_and 用于计算逻辑与，得到一个布尔张量，然后通过 torch.sum 计算其总和，即交集的像素数量
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            # 计算两个掩码的并集。torch.logical_or 用于计算逻辑或，得到一个布尔张量，然后通过 torch.sum 计算其总和，即并集的像素数量
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            # 计算IoU
            iou = intersection / union
            # 把IoU结果储存
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            # inner_iou计算了两个掩码之间的内部 IoU 值，用于衡量一个掩码被另一个掩码完全包含的程度
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou
    # 将 IoU 矩阵的下三角部分（包括对角线）置零，仅保留上三角部分，以避免重复计算 IoU
    iou_matrix.triu_(diagonal=1)
    # 计算每行的最大 IoU 值，即每个掩码与其他掩码之间的最大 IoU 值
    iou_max, _ = iou_matrix.max(dim=0)
    # 将内部 IoU 矩阵的下三角部分（不包括对角线）置零，仅保留上三角部分，以避免重复计算内部 IoU
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    # 计算每行的最大内部 IoU 值，即每个掩码与其他掩码之间的最大内部 IoU 值
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    # 将内部 IoU 矩阵的上三角部分（不包括对角线）置零，仅保留下三角部分，以避免重复计算内部 IoU
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    # 计算每行的最大内部 IoU 值，即每个掩码与其他掩码之间的最大内部 IoU 值
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    # 计算要保留的掩码
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    # 通过将 keep *= keep_conf、keep *= keep_inner_u 和 keep *= keep_inner_l 将以上条件综合起来，
    # 确定最终应该保留的掩码。即只保留 IoU 和内部 IoU 均满足要求且得分高于阈值的掩码
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    # 创建一个空的元组 masks_new，用于存储更新后的掩码
    masks_new = ()
    for masks_lvl in (args):
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        # 将稳定性分数与预测的重叠率相乘，得到掩码的综合得分
        scores = stability * iou_pred
        # 根据掩码的分割区域和综合得分，利用非最大抑制（NMS）算法选择要保留的掩码
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        # 根据非最大抑制的结果，过滤掉不需要保留的掩码，得到更新后的掩码列表
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        # 将更新后的掩码列表添加到 masks_new 元组中
        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image):
    # 图像预处理，将图像张量重新排列，最后一个维度变为通道数，并将tensor转换为numpy，BGR格式转为RGB
    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image): # 将掩码转换为分割图像，并生成分割地图
        seg_img_list = []
        # 初始化分割地图，维度和输入图像相同，初始值为-1
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            # 对于每个掩码，调用 get_seg_img 函数将其转换为分割图像
            seg_img = get_seg_img(mask, image)
            # 使用 pad_img 函数对分割图像进行填充，并调整图像大小为224x224
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            # 将调整大小后的分割图像添加到 seg_img_list 中
            seg_img_list.append(pad_seg_img)
            # 对于每个掩码，将其在 seg_map 中对应的区域标记为该掩码在列表中的索引值
            seg_map[masks[i]['segmentation']] = i
        # 将 seg_img_list 中的分割图像堆叠成一个张量seg_imgs，b 是掩码的数量，3是颜色通道
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        # 将seg_imgs重新排列，颜色通道放到第二个维度上
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    # 计算默认尺寸的分割图像和分割地图
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    # 计算三个不同尺度的分割图像和分割地图
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps


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


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)