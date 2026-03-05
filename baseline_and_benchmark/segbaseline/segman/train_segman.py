
import os

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import sys
import io
import base64
import json
import math
import time
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from contextlib import nullcontext

# 修复导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from models import SegMAN, load_encoder_pretrained
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接导入...")
    from models.model_segman import SegMAN, load_encoder_pretrained

# =========================
# 配置
# =========================
CLASS_ID_TO_NAME = {0: "sky", 1: "cloud", 2: "contamination"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}
NUM_CLASSES = len(CLASS_ID_TO_NAME)
IGNORE_INDEX = 255
TARGET_SIZE = (512, 512)


# =========================
# 数据集（读取 LabelMe JSON）
# =========================
class LabelMeJsonDataset(data.Dataset):
    """
    从包含 LabelMe *.json 的目录构建数据集。
    每个 json 需包含字段：
      - imageData: base64 编码的图像
      - imageWidth, imageHeight (可选，尽量提供)
      - shapes: 列表，每个 shape 为 polygon，字段包括：
            { "label": "sky"/"cloud"/"contamination"/或 "0"/"1"/"2",
              "points": [[x1,y1],[x2,y2],...],
              "shape_type": "polygon" }
    未标注区域像素使用 IGNORE_INDEX。
    """

    def __init__(self, data_dir: str, augment: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.augment = augment

        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        # 收集 json 文件
        self.json_files = sorted(glob.glob(os.path.join(self.data_dir, '*.json')))
        if not self.json_files:
            # 兼容子目录情形
            self.json_files = sorted(glob.glob(os.path.join(self.data_dir, '**', '*.json'), recursive=True))

        if not self.json_files:
            raise ValueError(f"在目录 {self.data_dir} 中未找到 *.json (LabelMe) 文件")

        print(f"在 {self.data_dir} 中找到 {len(self.json_files)} 个 LabelMe JSON 样本")

        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.json_files)

    def _decode_image_from_json(self, js: Dict) -> np.ndarray:
        """从 json 的 imageData 解码为 HWC 的 uint8 RGB 图像"""
        if 'imageData' not in js or js['imageData'] is None:
            # 兼容：如果没给 imageData，尝试 imagePath
            img_path = js.get('imagePath', None)
            if img_path is None:
                raise ValueError("json 中既无 imageData 也无 imagePath")
            # imagePath 可能是相对路径
            candidate = os.path.join(self.data_dir, img_path)
            if not os.path.isfile(candidate):
                raise ValueError(f"找不到图像文件: {candidate}")
            img = Image.open(candidate).convert('RGB')
            return np.array(img)

        # base64 -> bytes -> cv2.imdecode -> BGR -> RGB
        try:
            img_bytes = base64.b64decode(js['imageData'])
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                # 有些标注工具会用 PIL 可读但 cv2 不认的编码，回退 PIL
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return np.array(img)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as e:
            raise ValueError(f"imageData 解码失败: {e}")

    @staticmethod
    def _label_to_class_id(label: str) -> int:
        """将 shape.label 转为类别 id；支持 '0'/'1'/'2' 或 'sky'/'cloud'/'contamination'（大小写不敏感）"""
        if label is None:
            return IGNORE_INDEX
        lab = str(label).strip().lower()
        if lab in ('0', 'sky'):
            return 0
        if lab in ('1', 'cloud'):
            return 1
        if lab in ('2', 'contamination'):
            return 2
        # 未知标签 -> 忽略
        return IGNORE_INDEX

    def _polygons_to_mask(self, w: int, h: int, shapes: List[Dict]) -> np.ndarray:
        """将 shapes(多边形)光栅化为单通道 mask，初值 IGNORE_INDEX"""
        mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
        for shp in shapes or []:
            if shp.get('shape_type', 'polygon') != 'polygon':
                # 只支持 polygon，其它类型可按需扩展
                continue
            pts = np.asarray(shp.get('points', []), dtype=np.float32)
            if pts.size < 6:  # 至少 3 点
                continue
            cls_id = self._label_to_class_id(shp.get('label', None))
            if cls_id == IGNORE_INDEX:
                continue
            # 填充多边形（注意点需为 int）
            pts_int = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts_int], int(cls_id))
        return mask

    def __getitem__(self, idx: int):
        json_path = self.json_files[idx]
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                js = json.load(f)

            image = self._decode_image_from_json(js)  # HWC RGB uint8
            h, w = image.shape[:2]

            mask = self._polygons_to_mask(w, h, js.get('shapes', []))  # HW uint8

            # 调整至目标尺寸
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

            # 可选增强（与掩码一致变换）
            if self.augment:
                image, mask = self._augment(image, mask)

            # 修复负步长，确保连续
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

            # 转张量 & 归一化
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - self.mean) / self.std
            mask_tensor = torch.from_numpy(mask.astype(np.int64))

            return image_tensor, mask_tensor, os.path.basename(json_path)

        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")
            dummy_image = torch.zeros(3, TARGET_SIZE[1], TARGET_SIZE[0])
            dummy_mask = torch.full((TARGET_SIZE[1], TARGET_SIZE[0]), IGNORE_INDEX, dtype=torch.long)
            return dummy_image, dummy_mask, "error"

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # 随机旋转（-15° ~ 15°）
        if np.random.rand() > 0.5:
            angle = float(np.random.uniform(-15, 15))
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        # 随机亮度/对比度
        if np.random.rand() > 0.5:
            alpha = float(np.random.uniform(0.8, 1.2))  # 对比度
            beta = float(np.random.uniform(-20, 20))  # 亮度（像素级别）
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image, mask


# =========================
# 训练与评估
# =========================
def train_one_epoch(model, loader, optimizer, device, epoch, max_epochs, base_lr, total_iters, scaler=None):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0

    if len(loader) == 0:
        print(f"警告: 第 {epoch + 1} 周期的数据加载器为空，跳过训练")
        return 0.0

    use_amp = (device.type == 'cuda') and (scaler is not None)

    for batch_idx, (images, masks, _) in enumerate(loader):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        # 线性衰减学习率 (poly 可自定)
        current_iter = epoch * len(loader) + batch_idx
        lr = base_lr * (1 - current_iter / max(1, total_iters)) ** 0.9
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        ctx = torch.cuda.amp.autocast if use_amp else nullcontext
        with ctx():
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

        if batch_idx % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{max_epochs}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}')

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_and_confusion(model, loader, device) -> Tuple[float, np.ndarray, int]:
    """
    返回：平均损失、混淆矩阵(真实x预测, CxC)、有效像素总数
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0

    # 混淆矩阵
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_valid = 0

    if len(loader) == 0:
        print("警告: 评估数据加载器为空")
        return 0.0, conf, 0

    for images, masks, _ in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=True)

        loss = criterion(logits, masks)
        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=1)  # [B,H,W]
        valid = masks != IGNORE_INDEX
        if valid.any():
            preds_v = preds[valid].view(-1).cpu().numpy()
            gts_v = masks[valid].view(-1).cpu().numpy()
            total_valid += gts_v.size

            # 累加混淆矩阵
            for t, p in zip(gts_v, preds_v):
                if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                    conf[t, p] += 1

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, conf, total_valid


def metrics_from_confusion(conf: np.ndarray) -> Dict:
    """
    基于 CxC 混淆矩阵计算每类 precision/recall/F1、macro/weighted 平均与 overall_accuracy。
    """
    assert conf.shape == (NUM_CLASSES, NUM_CLASSES)
    # 每类 TP、FP、FN、TN（TN 不参与这里的 P/R/F1）
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)  # 每类真实为该类的像素数
    pred_count = conf.sum(axis=0).astype(np.float64)  # 每类预测为该类的像素数
    total = conf.sum().astype(np.float64)

    # overall accuracy
    overall_acc = (tp.sum() / total) if total > 0 else 0.0

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for cid in range(NUM_CLASSES):
        p = (tp[cid] / pred_count[cid]) if pred_count[cid] > 0 else 0.0
        r = (tp[cid] / support[cid]) if support[cid] > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        cname = CLASS_ID_TO_NAME[cid]
        per_class[cname] = {
            "precision": float(p),
            "recall": float(r),
            "f1-score": float(f1),
            "support": float(support[cid])
        }
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support[cid])

    # macro avg
    macro = {
        "precision": float(np.mean(precisions) if len(precisions) else 0.0),
        "recall": float(np.mean(recalls) if len(recalls) else 0.0),
        "f1-score": float(np.mean(f1s) if len(f1s) else 0.0),
        "support": float(np.sum(supports))
    }

    # weighted avg
    sw = np.sum(supports)
    if sw > 0:
        w_precision = float(np.sum(np.array(precisions) * np.array(supports)) / sw)
        w_recall = float(np.sum(np.array(recalls) * np.array(supports)) / sw)
        w_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / sw)
    else:
        w_precision = w_recall = w_f1 = 0.0

    weighted = {
        "precision": w_precision,
        "recall": w_recall,
        "f1-score": w_f1,
        "support": float(sw)
    }

    report = {
        "overall_accuracy": float(overall_acc),
        "per_class_results": {
            **per_class,
            "accuracy": float(overall_acc),
            "macro avg": macro,
            "weighted avg": weighted
        }
    }
    return report


# =========================
# 杂项
# =========================
def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def create_sample_labelme(dir_path: str, idx: int, w: int = 640, h: int = 480):
    """
    生成一个简单的示例 LabelMe json（含 imageData 与几个多边形）
    """
    os.makedirs(dir_path, exist_ok=True)
    # 生成背景渐变图
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)[:, None]
    r = (x * 255).astype(np.uint8)
    g = (y * 255).astype(np.uint8)
    b = np.full((h, w), 180, dtype=np.uint8)
    rgb = np.stack([np.tile(r, (h, 1)), np.tile(g, (1, w)), b], axis=-1)

    # 编码到 imageData
    _, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    # 多边形：上半部 sky、随机云、一个污染小块
    sky_poly = [[0, 0], [w - 1, 0], [w - 1, h // 2], [0, h // 2]]
    cloud_poly = [[w // 4, h // 3], [w // 2, h // 3 - 20], [w // 2 + 60, h // 3 + 10], [w // 4 + 20, h // 3 + 40]]
    contam_poly = [[int(0.7 * w), int(0.7 * h)], [int(0.85 * w), int(0.72 * h)],
                   [int(0.9 * w), int(0.9 * h)], [int(0.75 * w), int(0.88 * h)]]

    js = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [
            {"label": "sky", "points": sky_poly, "group_id": None, "shape_type": "polygon", "flags": {}},
            {"label": "cloud", "points": cloud_poly, "group_id": None, "shape_type": "polygon", "flags": {}},
            {"label": "contamination", "points": contam_poly, "group_id": None, "shape_type": "polygon", "flags": {}},
        ],
        "imagePath": f"sample_{idx:02d}.png",
        "imageData": image_b64,
        "imageWidth": w,
        "imageHeight": h
    }

    out_json = os.path.join(dir_path, f"sample_{idx:02d}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    return out_json


def create_sample_data():
    """创建示例数据（用于测试训练管线）"""
    print("创建示例数据...")
    for split in ['train', 'val', 'test']:
        d = os.path.join('data', split)
        os.makedirs(d, exist_ok=True)
    # 各创建若干 json
    for i in range(4):
        create_sample_labelme(os.path.join('data', 'train'), i + 1)
    for i in range(2):
        create_sample_labelme(os.path.join('data', 'val'), i + 1)
    for i in range(2):
        create_sample_labelme(os.path.join('data', 'test'), i + 1)
    print("示例数据创建完成。数据目录结构：data/train/*.json, data/val/*.json, data/test/*.json")


def validate_data_directory_json(data_dir: str) -> bool:
    """检查目录是否包含足够数量的 *.json"""
    print(f"验证数据目录(JSON): {data_dir}")
    if not os.path.exists(data_dir):
        print("错误: 目录不存在")
        return False
    jsons = glob.glob(os.path.join(data_dir, '*.json'))
    if not jsons:
        jsons = glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True)
    n = len(jsons)
    print(f"找到 {n} 个 json")
    if n < 2:
        print("警告: 样本少于 2，建议至少 2 个样本")
        return False
    return True


def load_config(config_path: str) -> Dict:
    """加载JSON配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 设置默认值
    defaults = {
        "variant": "tiny",
        "epochs": 80,
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "out_dir": "./runs/segman",
        "num_workers": 4,
        "seed": 42,
        "early_stop_patience": 20
    }

    # 用配置文件中的值覆盖默认值
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    return config


class EarlyStopper:
    """早停器类"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.max_validation_accuracy = 0.0

    def early_stop(self, validation_loss: float) -> bool:
        """基于验证损失进行早停判断"""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_accuracy(self, validation_accuracy: float) -> bool:
        """基于验证准确率进行早停判断"""
        if validation_accuracy > self.max_validation_accuracy:
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy < (self.max_validation_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_best_model(model, checkpoint_path: str, device) -> Tuple:
    """加载最佳模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"最佳模型文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    best_epoch = checkpoint.get('epoch', 0)
    best_accuracy = checkpoint.get('best_accuracy', 0.0)

    print(f"加载最佳模型 - 周期: {best_epoch + 1}, 验证准确率: {best_accuracy:.6f}")
    return model, best_epoch, best_accuracy


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description='SegMAN 语义分割（LabelMe JSON imageData）训练脚本')
    parser.add_argument('--config', type=str, required=True, help='JSON配置文件路径')
    parser.add_argument('--create_sample', action='store_true', help='创建示例数据后退出')
    parser.add_argument('--validate_only', action='store_true', help='仅验证数据目录后退出')
    args = parser.parse_args()

    # 创建示例
    if args.create_sample:
        create_sample_data()
        print("示例数据已创建。")
        return

    # 加载配置文件
    config = load_config(args.config)

    # 检查必要的目录配置
    required_dirs = ['train_dir', 'val_dir', 'test_dir']
    for dir_key in required_dirs:
        if dir_key not in config:
            raise ValueError(f"配置文件中缺少必需的目录配置: {dir_key}")

    # 随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 验证数据目录
    train_ok = validate_data_directory_json(config['train_dir'])
    val_ok = validate_data_directory_json(config['val_dir'])
    test_ok = validate_data_directory_json(config['test_dir'])

    if not (train_ok and val_ok and test_ok):
        print("数据不足，自动创建示例数据...")
        create_sample_data()
        config['train_dir'] = 'data/train'
        config['val_dir'] = 'data/val'
        config['test_dir'] = 'data/test'

    if args.validate_only:
        print("数据验证完成")
        return

    # 输出目录
    os.makedirs(config['out_dir'], exist_ok=True)

    try:
        # 数据集 & DataLoader
        print("创建数据集...")
        train_dataset = LabelMeJsonDataset(config['train_dir'], augment=True)
        val_dataset = LabelMeJsonDataset(config['val_dir'], augment=False)
        test_dataset = LabelMeJsonDataset(config['test_dir'], augment=False)

        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")
        print(f"测试样本数: {len(test_dataset)}")

        # 动态 batch
        if len(train_dataset) < config['batch_size']:
            print(f"警告: 批量大小从 {config['batch_size']} 调整为 {len(train_dataset)}（受限于样本数）")
            config['batch_size'] = max(1, len(train_dataset))

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=min(config['num_workers'], os.cpu_count() or 1),
            pin_memory=(device.type == 'cuda'),
            drop_last=False
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=min(config['num_workers'], os.cpu_count() or 1),
            pin_memory=(device.type == 'cuda'),
            drop_last=False
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=min(config['num_workers'], os.cpu_count() or 1),
            pin_memory=(device.type == 'cuda'),
            drop_last=False
        )

        if len(train_loader) == 0:
            print("错误: 训练数据加载器为空，无法训练")
            return

        # 模型
        print(f"创建 SegMAN-{config['variant']} 模型...")
        model = SegMAN(num_classes=NUM_CLASSES, variant=config['variant'])

        # 预训练
        if 'pretrained_encoder' in config and config['pretrained_encoder'] and os.path.exists(
                config['pretrained_encoder']):
            print(f"加载预训练权重: {config['pretrained_encoder']}")
            load_stats = load_encoder_pretrained(model.encoder, config['pretrained_encoder'])
            print(f"预训练权重加载统计: {load_stats}")
        elif 'pretrained_encoder' in config and config['pretrained_encoder']:
            print(f"警告: 预训练权重文件不存在: {config['pretrained_encoder']}")

        model = model.to(device)

        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总数: {total_params:,}")

        # 优化器 & AMP
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        # 早停器
        early_stopper = EarlyStopper(patience=config['early_stop_patience'], min_delta=0.001)

        # 训练日志
        training_log = {
            "config": config,
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_accuracy": 0.0,
            "early_stopped": False,
            "best_epoch": 0
        }

        best_accuracy = 0.0
        best_epoch = 0
        start_time = time.time()
        total_iters = config['epochs'] * max(1, len(train_loader))

        print("开始训练...")
        for epoch in range(config['epochs']):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                epoch, config['epochs'], config['lr'], total_iters, scaler=scaler
            )

            # 验证集评估
            val_loss, conf, valid_pixels = evaluate_and_confusion(model, val_loader, device)
            metrics = metrics_from_confusion(conf)
            val_accuracy = metrics["overall_accuracy"]

            training_log["train_loss"].append(float(train_loss))
            training_log["val_loss"].append(float(val_loss))
            training_log["val_accuracy"].append(float(val_accuracy))

            epoch_time = time.time() - t0
            print(f"Epoch {epoch + 1}/{config['epochs']} [{epoch_time:.1f}s]:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_accuracy:.6f} (有效像素: {valid_pixels})")

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                model_path = os.path.join(config['out_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': float(best_accuracy),
                    'config': config
                }, model_path)
                print(f"  ✓ 保存最佳模型，准确率: {best_accuracy:.6f}")

            # 检查早停
            if early_stopper.early_stop_accuracy(val_accuracy):
                print(f"早停触发! 在周期 {epoch + 1} 停止训练。")
                print(f"最佳验证准确率 {best_accuracy:.6f} 在周期 {best_epoch + 1}")
                training_log["early_stopped"] = True
                training_log["best_epoch"] = best_epoch
                break

            # 定期 checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt = os.path.join(config['out_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': float(val_accuracy)
                }, ckpt)

            # 持久化训练日志
            training_log["best_accuracy"] = float(best_accuracy)
            training_log["best_epoch"] = best_epoch
            save_json(training_log, os.path.join(config['out_dir'], 'training_log.json'))

        total_time = time.time() - start_time
        print("训练完成!")
        print(f"总训练时间: {total_time:.1f}s")
        print(f"最佳验证准确率: {best_accuracy:.6f} (周期: {best_epoch + 1})")

        # 保存最终模型（如果训练完成而非早停）
        if not training_log["early_stopped"]:
            final_model_path = os.path.join(config['out_dir'], 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            print(f'最终模型已保存: {final_model_path}')

        # 加载最佳模型并在测试集上评估
        print("加载最佳模型并在测试集上评估...")
        best_model_path = os.path.join(config['out_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            # 重新创建模型实例以确保架构一致
            best_model = SegMAN(num_classes=NUM_CLASSES, variant=config['variant']).to(device)
            best_model, loaded_epoch, loaded_accuracy = load_best_model(best_model, best_model_path, device)

            # 在测试集上评估最佳模型
            test_loss, test_conf, test_valid_pixels = evaluate_and_confusion(best_model, test_loader, device)
            test_metrics = metrics_from_confusion(test_conf)

            # 构建测试结果报告
            test_report = {
                "best_epoch": int(loaded_epoch + 1),
                "best_val_accuracy": float(loaded_accuracy),
                "test_overall_accuracy": test_metrics["overall_accuracy"],
                "test_loss": float(test_loss),
                "test_valid_pixels": int(test_valid_pixels),
                "per_class_results": test_metrics["per_class_results"]
            }

            # 保存测试结果
            test_metrics_path = os.path.join(config['out_dir'], 'test_metrics.json')
            save_json(test_report, test_metrics_path)

            print("\n" + "=" * 60)
            print("测试集评估结果 (基于最佳模型):")
            print(f"最佳周期: {loaded_epoch + 1}")
            print(f"验证集准确率: {loaded_accuracy:.6f}")
            print(f"测试集准确率: {test_metrics['overall_accuracy']:.6f}")
            print(f"测试集损失: {test_loss:.4f}")
            print(f"测试集有效像素: {test_valid_pixels}")
            print("=" * 60)

            # 打印详细指标
            print("\n详细分类指标:")
            print(json.dumps(test_metrics["per_class_results"], indent=2, ensure_ascii=False))
            print(f"\n测试集指标文件已保存: {test_metrics_path}")
        else:
            print("警告: 未找到最佳模型文件，跳过测试集评估")

    except Exception as e:
        print(f'训练过程中出错: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()