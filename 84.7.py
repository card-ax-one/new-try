import os
import json
import math
import csv
import copy
import random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from PIL import Image

# ======================
# 0. 可复现实验的随机种子
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ======================
# 1. 类别映射（与你之前一致）
# ======================
class_to_idx = {
    "Honeysuckles": 0, "Gardenia": 1, "Tianhukui": 2, "Gouweibacao": 3, "Shuiqincai": 4,
    "Morningglory": 5, "Bosipopona": 6, "Mantuoluo": 7, "Tongquancao": 8, "Perillas": 9,
    "Jicai": 10, "Xiaoji": 11, "Angelica": 12, "Heshouwu": 13, "Yichuanhong": 14,
    "Malan": 15, "Rabdosiaserra": 16, "Zeqi": 17, "Bupleurum": 18, "Plantains": 19,
    "Ginsengs": 20, "Juaner": 21, "Kucai": 22, "Selfheals": 23, "Sedum_sarmentosum": 24,
    "Agastacherugosa": 25, "Xunma": 26, "Boheye": 27, "Hairyveinagrimony": 28, "Feipeng": 29,
    "Guizhencao": 30, "Eichhorniacrassipes": 31, "Dandelions": 32, "Zhajiangcao": 33,
    "Wahlenbergia": 34, "Radixisatidis": 35, "Mangnoliaofficinalis": 36, "Odoratum": 37,
    "Cangerzi": 38, "Commelina_communis": 39, "Chenopodiumalbum": 40, "Monochoriavaginalis": 41,
    "Ziyunying": 42, "Pinellia": 43, "Hongliao": 44, "Moneygrass": 45, "Lotusseed": 46,
    "Ophiopogon": 47, "Qigucao": 48, "Huanghuacai": 49, "Wormwood": 50, "Palms": 51,
    "Denglongcao": 52, "Xiaoqieyi": 53
}
idx_to_class = {v: k for k, v in class_to_idx.items()}
with open("class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)

# ======================
# 2. 数据集定义：强制使用自定义 class_to_idx
# ======================
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_to_idx=None):
        self.custom_class_to_idx = class_to_idx
        super().__init__(root, transform=transform)
        # 强制使用自定义映射
        self.class_to_idx = self.custom_class_to_idx
        self.samples = [
            (path, self.custom_class_to_idx[os.path.basename(os.path.dirname(path))])
            for path, _ in self.samples
        ]
        self.targets = [s[1] for s in self.samples]

class CustomSubset(Subset):
    def __init__(self, dataset, indices, class_to_idx):
        super().__init__(dataset, indices)
        self.class_to_idx = class_to_idx
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        # dataset 已经是数字标签，直接返回
        return image, label

# ======================
# 3. 数据增强（更强：RandAugment + RandomErasing）
# ======================
# 训练增强
train_transforms_list = [
    transforms.Resize((300, 300)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
]
# 尝试加入 RandAugment（老版本 torchvision 可能没有，做兼容）
try:
    train_transforms_list.insert(1, transforms.RandAugment())
except Exception:
    pass
train_transforms_list += [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
]
transform_train = transforms.Compose(train_transforms_list)

# 验证/测试增强（与训练验证阶段保持一致）
transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# 4. Mixup / CutMix 实现（默认开启 Mixup）
# ======================
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 可选：Focal Loss（与 CE 混合）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ======================
# 5. 注意力模块 CBAM + 更强 FC
# ======================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
    def forward(self, x):
        ca = self.channel(x) * x
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return sa * ca

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 在 layer4 末尾加 CBAM（不改动空洞卷积，保持兼容性与稳定性）
        self.backbone.layer4.add_module("cbam", CBAM(2048))
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ======================
# 6. Warmup + Cosine 学习率调度
# ======================
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # 线性 warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # Cosine 衰减
        t = (epoch - self.warmup_epochs) / max(1, (self.max_epochs - self.warmup_epochs))
        return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * t)) / 2 for base_lr in self.base_lrs]

# ======================
# 7. EMA（指数滑动平均）
# ======================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)

# ======================
# 8. 数据准备（分层划分 + 类别均衡采样）
# ======================

def build_loaders(train_root='train_set', batch_size=32):
    full_train = CustomDataset(train_root, transform=transform_train, class_to_idx=class_to_idx)
    full_val   = CustomDataset(train_root, transform=transform_val,   class_to_idx=class_to_idx)

    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_train):
        label_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        tr_idx, va_idx = train_test_split(indices, test_size=0.1, random_state=SEED, shuffle=True)
        train_indices.extend(tr_idx)
        val_indices.extend(va_idx)

    train_dataset = CustomSubset(full_train, train_indices, class_to_idx)
    val_dataset   = CustomSubset(full_val,   val_indices,   class_to_idx)

    # 类别均衡采样器（避免长尾偏差）
    train_labels = [full_train.targets[i] for i in train_indices]
    counts = Counter(train_labels)
    num_classes = len(counts)
    class_weights = torch.tensor([1.0 / (counts[c] + 1e-6) for c in range(num_classes)], dtype=torch.float)
    sample_weights = torch.tensor([class_weights[y] for y in train_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# ======================
# 9. 评估（可选 TTA）
# ======================

def evaluate(model, dataloader, device, use_tta=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if use_tta:
                # 简单 TTA：原图 + 水平翻转，取平均
                outputs = model(imgs)
                outputs_flipped = model(torch.flip(imgs, dims=[3]))
                outputs = (outputs + outputs_flipped) / 2
            else:
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(1, total)

# ======================
# 10. 训练主流程（两阶段 + Mixup + EMA + WarmupCosine）
# ======================

def train(num_classes=54, epochs_stage1=10, epochs_stage2=60, batch_size=32, mixup_alpha=0.4, focal_alpha=0.0,
          lr_stage1=1e-3, lr_stage2=3e-4, warmup_epochs=3, model_out='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = build_loaders(batch_size=batch_size)

    model = ResNet50_CBAM(num_classes=num_classes).to(device)

    # 损失：CE + 可选 Focal 混合
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    focal_loss = FocalLoss(gamma=2.0) if focal_alpha > 0 else None

    def blended_loss(logits, targets):
        loss_ce = ce_loss(logits, targets)
        if focal_alpha > 0:
            loss_focal = focal_loss(logits, targets)
            return (1 - focal_alpha) * loss_ce + focal_alpha * loss_focal
        return loss_ce

    # -------- 阶段1：冻结 backbone，仅训练 fc --------
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.backbone.fc.parameters():
        p.requires_grad_(True)

    opt1 = optim.AdamW(model.backbone.fc.parameters(), lr=lr_stage1, weight_decay=1e-4)
    sched1 = WarmupCosine(opt1, warmup_epochs=warmup_epochs, max_epochs=epochs_stage1)
    ema = ModelEMA(model, decay=0.999)

    best_acc, best_wts = 0.0, None

    for epoch in range(epochs_stage1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Mixup
            imgs, ya, yb, lam = mixup_data(imgs, labels, alpha=mixup_alpha)

            opt1.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = mixup_criterion(blended_loss, logits, ya, yb, lam)
            loss.backward()
            opt1.step()
            ema.update(model)
        sched1.step()

        val_acc = evaluate(ema.ema, val_loader, device, use_tta=False)
        print(f"[Stage1][Epoch {epoch+1}/{epochs_stage1}] Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(ema.ema.state_dict())

    # -------- 阶段2：解冻全模型微调 --------
    for p in model.parameters():
        p.requires_grad_(True)

    opt2 = optim.AdamW(model.parameters(), lr=lr_stage2, weight_decay=2e-4)
    sched2 = WarmupCosine(opt2, warmup_epochs=warmup_epochs, max_epochs=epochs_stage2)

    for epoch in range(epochs_stage2):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam = mixup_data(imgs, labels, alpha=mixup_alpha)

            opt2.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = mixup_criterion(blended_loss, logits, ya, yb, lam)
            loss.backward()
            opt2.step()
            ema.update(model)
        sched2.step()

        val_acc = evaluate(ema.ema, val_loader, device, use_tta=True)
        print(f"[Stage2][Epoch {epoch+1}/{epochs_stage2}] Val Acc (TTA): {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(ema.ema.state_dict())

    # 保存最佳 EMA 权重
    if best_wts is not None:
        torch.save(best_wts, model_out)
    else:
        torch.save(ema.ema.state_dict(), model_out)
    print(f"Training done! Best Val Acc: {best_acc:.2f}% | Saved to {model_out}")

    return model_out

# ======================
# 11. 推理到 CSV（数字标签）
# ======================
class TestDatasetFlat(torch.utils.data.Dataset):
    """遍历 test_set 目录下的所有文件（不分子文件夹），输出 (tensor, path)"""
    def __init__(self, folder, transform):
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if not f.startswith('.')]
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img), self.image_paths[index]
    def __len__(self):
        return len(self.image_paths)

def infer_to_csv(model_path='best_model.pth', test_root='test_set', out_csv='test_predictions.csv', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50_CBAM(num_classes=54).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_ds = TestDatasetFlat(test_root, transform_val)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    results = []
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()
            for p, lab in zip(paths, preds):
                results.append((os.path.basename(p), int(lab)))

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'predicted_class'])
        w.writerows(results)
    print(f"Inference done. Saved: {out_csv}")

# ======================
# 12. 一键运行入口
# ======================
if __name__ == '__main__':
    # 训练 + 保存最佳模型
    best_path = train(
        num_classes=54,
        epochs_stage1=10,
        epochs_stage2=60,
        batch_size=32,
        mixup_alpha=0.4,
        focal_alpha=0.0,      # 如需与 CE 混合 Focal，可调到 0.2~0.4
        lr_stage1=1e-3,
        lr_stage2=3e-4,
        warmup_epochs=3,
        model_out='best_model.pth'
    )

    # 训练完立刻对 test_set 推理（数字标签）并导出 CSV
    if os.path.exists('test_set'):
        infer_to_csv(model_path=best_path, test_root='test_set', out_csv='test_predictions.csv', batch_size=32)
