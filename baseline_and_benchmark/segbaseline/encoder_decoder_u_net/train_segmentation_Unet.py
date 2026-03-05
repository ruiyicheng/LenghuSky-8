import os
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import argparse


# =====================
# Configuration
# =====================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# =====================
# Data Loading Functions
# =====================
def load_json_image_mask(json_path, label2id):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse image
    image_data = base64.b64decode(data["imageData"])
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate Mask
    height, width = data["imageHeight"], data["imageWidth"]
    mask = 3 * np.ones((height, width), dtype=np.uint8)  # 3 is ignore index
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in label2id:
            continue
        points = np.array(shape["points"], dtype=np.float32)
        points = np.round(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], label2id[label])
    
    return image_rgb, mask


# =====================
# Dataset Class
# =====================
class CloudDataset(Dataset):
    def __init__(self, json_dir, label2id, transform=None):
        self.json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
        self.transform = transform
        self.label2id = label2id
        self.original_imgs = []
        self.masks = []
        
        print(f"Loading {len(self.json_files)} images from {json_dir}")
        for f in self.json_files:
            img, mask = load_json_image_mask(f, label2id)
            self.original_imgs.append(img)
            self.masks.append(mask)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        original_img = self.original_imgs[idx]
        mask = self.masks[idx]
        mask_tensor = torch.from_numpy(mask).long()
        img_pil = Image.fromarray(original_img)
        img_tensor = self.transform(img_pil) if self.transform else T.ToTensor()(img_pil)
        return img_tensor, mask_tensor


# =====================
# U-Net Model Definition
# =====================
class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# =====================
# Evaluation Function
# =====================
def evaluate_model(model, dataloader, device, ignore_index=3):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for img_tensor, mask_tensor in dataloader:
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1)
            
            # Flatten and filter out ignore_index
            preds_flat = preds.flatten().cpu().numpy()
            targets_flat = mask_tensor.flatten().cpu().numpy()
            
            # Remove ignore_index pixels
            valid_mask = targets_flat != ignore_index
            all_preds.extend(preds_flat[valid_mask])
            all_targets.extend(targets_flat[valid_mask])
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    class_report = classification_report(
        all_targets, all_preds, 
        target_names=["sky", "cloud", "contamination"],
        output_dict=True,
        zero_division=0
    )
    
    return accuracy, class_report


# =====================
# Main Training Function
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup device and paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    
    # Label mapping
    label2id = {"sky": 0, "cloud": 1, "contamination": 2}
    IGNORE_INDEX = 3
    
    # Data preprocessing
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    
    # Create datasets
    dataset_train = CloudDataset(cfg["train_labelme_root"], label2id, transform=transform)
    dataset_test = CloudDataset(cfg["test_labelme_root"], label2id, transform=transform)
    dataset_val = CloudDataset(cfg["val_labelme_root"], label2id, transform=transform)
    # Split train into train/val
    # num_train = int(0.889 * len(dataset_train))
    # num_val = len(dataset_train) - num_train
    # dataset_train, dataset_val = torch.utils.data.random_split(
    #     dataset_train, [num_train, num_val],
    #     generator=torch.Generator().manual_seed(cfg["random_state"])
    # )
    
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=12, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=12, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=12, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        weight=torch.tensor([1.0, 1.0, 1.0], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop with early stopping
    print(f"\nStarting training with seed {cfg['random_state']}...")
    patience = 5
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(cfg.get("num_epochs", 500)):
        model.train()
        total_loss = 0.0
        
        for img_tensor, mask_tensor in dataloader_train:
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            optimizer.zero_grad()
            logits = model(img_tensor)
            loss = criterion(logits, mask_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader_train)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for img_tensor, mask_tensor in dataloader_val:
                img_tensor = img_tensor.to(device)
                mask_tensor = mask_tensor.to(device)
                logits = model(img_tensor)
                loss = criterion(logits, mask_tensor)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(dataloader_val)
        
        print(f"Epoch {epoch+1:3d}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience = 5
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break
    
    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_accuracy, test_report = evaluate_model(model, dataloader_test, device, IGNORE_INDEX)
    
    print(f"\nTest Results (Seed {cfg['random_state']}):")
    print(f"Overall Accuracy: {test_accuracy:.4f}")
    print("Per-class metrics:")
    for class_name, metrics in test_report.items():
        if class_name in ["sky", "cloud", "contamination"]:
            print(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Save results to JSON
    results = {
        "overall_accuracy": float(test_accuracy),
        "per_class_results": test_report
    }
    
    results_path = os.path.join(
        cfg["inference_out_dir"], 
        f"test_results_seed_{cfg['random_state']}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(
        cfg["checkpoint_dir"], 
        f"model_seed_{cfg['random_state']}.pth"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'random_state': cfg['random_state']
    }, checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()