import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ======================
# 1. 自定义 Subset，保持自定义 class_to_idx
# ======================
class CustomSubset(Subset):
    def __init__(self, dataset, indices, class_to_idx):
        super().__init__(dataset, indices)
        self.class_to_idx = class_to_idx
        self.original_class_to_idx = dataset.class_to_idx

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        original_class = self.dataset.classes[label]
        new_label = self.class_to_idx[original_class]
        return image, new_label


# ======================
# 2. 类别映射
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
# 3. 数据增强
# ======================
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# 4. 数据集划分
# ======================


class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_to_idx=None):
        self.custom_class_to_idx = class_to_idx
        super().__init__(root, transform=transform)
        # 重新赋值，强制使用自定义的索引
        self.class_to_idx = self.custom_class_to_idx
        self.samples = [(path, self.custom_class_to_idx[class_name])
                        for path, class_name in [(path, os.path.basename(os.path.dirname(path)))
                                                 for path, _ in self.samples]]

# 训练集

dataset_train_full = CustomDataset(root='train_set', transform=transform_train,class_to_idx=class_to_idx)
dataset_val_full = CustomDataset(root='train_set', transform=transform_val, class_to_idx=class_to_idx)




label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset_train_full):
    label_to_indices[label].append(idx)

train_indices, val_indices = [], []
for label, indices in label_to_indices.items():
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)

train_dataset = CustomSubset(dataset_train_full, train_indices, class_to_idx)
val_dataset = CustomSubset(dataset_val_full, val_indices, class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 5. 模型定义（带 Dropout）
# ======================
class ResNet50_FC_Dropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = ResNet50_FC_Dropout(num_classes=54).to(device)

# ======================
# 6. 损失函数（Label Smoothing）
# ======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ======================
# 7. 训练函数
# ======================
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def train_model(model, train_loader, val_loader, epochs_stage1=5, epochs_stage2=15):
    best_acc = 0
    best_model_wts = None
    acc_log = {}

    # -------- 阶段 1：冻结主干，只训练 fc --------
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.Adam(model.backbone.fc.parameters(), lr=1e-3)
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=epochs_stage1)

    for epoch in range(epochs_stage1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_stage1.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()
        scheduler_stage1.step()

        val_acc = evaluate(model, val_loader)
        acc_log[f"stage1_epoch_{epoch}"] = val_acc
        print(f"[阶段1][Epoch {epoch+1}] 验证集准确率: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    # -------- 阶段 2：解冻全模型微调 --------
    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-4)
    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=epochs_stage2)

    for epoch in range(epochs_stage2):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_stage2.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()
        scheduler_stage2.step()

        val_acc = evaluate(model, val_loader)
        acc_log[f"stage2_epoch_{epoch}"] = val_acc
        print(f"[阶段2][Epoch {epoch+1}] 验证集准确率: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    # 保存最佳模型
    torch.save(best_model_wts, "best_resnet50.pth")
    print(f"训练完成！最佳验证集准确率: {best_acc:.2f}%")
    return acc_log


# ======================
# 8. 开始训练
# ======================
acc_log = train_model(model, train_loader, val_loader)
print(acc_log)