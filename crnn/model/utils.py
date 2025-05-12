import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RescaleHeight:
    def __init__(self, img_height):
        self.img_height = img_height

    def __call__(self, img):
        h = self.img_height
        w = int(img.width * (h / img.height))
        return transforms.functional.resize(img, (h, w))

# --- OCR Dataset ---
class OCRDataset(Dataset):
    def __init__(self, csv_path, char2idx, img_height=32, max_samples=None):
        self.data = []
        self.char2idx = char2idx
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            RescaleHeight(img_height),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                path, label = line.strip().split('|')
                if os.path.exists(path):
                    self.data.append((path, label))

        self.samples = self.data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert("L")
            image = self.transform(image)
            label_idx = [self.char2idx[c] for c in text if c in self.char2idx]
            return image, torch.tensor(label_idx, dtype=torch.long), torch.tensor(len(label_idx), dtype=torch.long)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_image = torch.zeros(1, 32, 256)
            dummy_label = torch.tensor([0], dtype=torch.long)
            return dummy_image, dummy_label, torch.tensor(1, dtype=torch.long)

# --- Collate Function ---
def ocr_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    max_width = max([img.shape[-1] for img in images])

    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[-1]
        padded_img = F.pad(img, (0, pad_width), "constant", 0)
        padded_images.append(padded_img)

    padded_images = torch.stack(padded_images)
    label_lengths = torch.stack(label_lengths)
    flat_labels = torch.cat(labels)

    return padded_images, flat_labels, label_lengths