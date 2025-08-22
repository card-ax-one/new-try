import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import csv

# ======================
# 1. 数据预处理
# ======================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ======================
# 2. 测试集 Dataset
# ======================
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.image_paths = [os.path.join(folder, fname)
                            for fname in os.listdir(folder)
                            if not fname.startswith('.')]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img), self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)


# ======================
# 3. 模型定义
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
        self.backbone = resnet50(weights=None)
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


def main():
    # ======================
    # 4. 主程序逻辑
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 54

    # 初始化模型
    model = ResNet50_CBAM(num_classes).to(device)

    # 加载权重
    try:
        state_dict = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("成功加载模型权重")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 测试集推理
    test_dataset = TestDataset("test_set", transform)
    # 关键修改：在Windows下将num_workers设为0避免多进程问题
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    results = []
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            for p, label in zip(paths, preds):
                results.append((os.path.basename(p), label.item()))

    # 保存结果
    with open("test_predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class"])
        writer.writerows(results)

    print(f"预测完成，结果已保存到 test_predictions.csv (共 {len(results)} 条预测)")


if __name__ == '__main__':
    # 在Windows下需要这行代码
    torch.multiprocessing.freeze_support()
    main()