import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from model import CRNN
from utils import OCRDataset, ocr_collate_fn

alphabet = "᠀᠃᠄᠋᠌᠍ᠠᠡᠢᠣᠤᠥᠦᠧᠨᠩᠪᠫᠬᠭᠮᠯᠰᠱᠲᠳᠴᠵᠶᠷᠸᠹᠺᠻᠼᠽᠾ"
char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
char2idx["<BLANK>"] = 0
idx2char = {i: c for c, i in char2idx.items()}


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = OCRDataset("synthetic.csv", char2idx, max_samples=10000)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ocr_collate_fn,
        num_workers=0,  # Keep this 0 for Windows compatibility with custom classes
        pin_memory=True
    )

    model = CRNN(nclass=len(char2idx)).to(device)
    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    for epoch in range(20):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (images, labels, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long, device=device)

            loss = criterion(log_probs, labels, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(epoch_loss)

    torch.save(model.state_dict(), "crnn_model_optimized.pth")
    print("Training complete. Model saved to crnn_model_optimized.pth")

if __name__ == "__main__":
    train()
