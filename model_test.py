import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import csv

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 必须与训练时相同
                         std=[0.229, 0.224, 0.225])
])


# 测试集 Dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder)]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img), self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)


# 加载测试集
test_dataset = TestDataset("test_set", transform)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 关键修改：模型定义与加载
# ======================
class ResNet50_FC_Dropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=None)  # 不加载默认权重
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # 必须与训练时结构一致
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# 初始化模型（必须与训练时结构完全相同）
num_classes = 54  # 确保与训练时类别数一致
model = ResNet50_FC_Dropout(num_classes).to(device)

# 加载训练好的权重
try:
    state_dict = torch.load("best_model.pth", map_location=device)
    # 处理可能的键名不匹配（如果保存的是完整模型而非state_dict）
    if not all(k.startswith('backbone.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.'):
                new_state_dict[f'backbone.{k}'] = v  # 调整键名匹配
            else:
                new_state_dict[f'backbone.{k}'] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    print("成功加载预训练权重！")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# ======================
# 推理部分
# ======================
model.eval()
results = []
with torch.no_grad():
    for imgs, paths in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        for p, label in zip(paths, preds):
            results.append((os.path.basename(p), label.item()))

# 保存预测结果
with open("test_predictions4.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "predicted_class"])
    writer.writerows(results)

print("预测完成，结果已保存到 test_predictions4.csv")