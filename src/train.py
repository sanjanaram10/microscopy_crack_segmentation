import json
import time
import torch
import threading
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import Config
from src.model import build_model
from src.dataset import CrackDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import set_seed, collate_fn, save_checkpoint, plot_training_curve


def _move_to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, loader, optimizer, device, scaler, grad_accum_steps=1):
    model.train()
    total_loss = 0
    batch_count = 0
    running_avg = 0.0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        t0 = time.time()
        batch_count += 1

        images, targets = _move_to_device(images, targets, device)
        loss_dict = model(images, targets)

        losses = sum(loss_dict.values()) / grad_accum_steps
        loss_val = losses.item()

        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        if batch_idx % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        iter_time = time.time() - t0
        running_avg = iter_time if running_avg == 0.0 else (running_avg * 0.9 + iter_time * 0.1)
        print(f"Batch {batch_idx:4d} | time={iter_time:.3f}s | avg={running_avg:.3f}s | loss={loss_val:.4f}")

        total_loss += loss_val * grad_accum_steps

    return total_loss / batch_count if batch_count > 0 else 0.0


def mask_iou(pred_mask, gt_mask):
    pred = pred_mask.bool()
    gt = gt_mask.bool()
    intersection = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return intersection / union if union > 0 else 0.0


def image_iou(pred_masks, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    ious = []
    gt_used = set()
    preds = [m.squeeze(0) > 0.5 for m in pred_masks]
    gts = [m.squeeze(0).bool() for m in gt_masks]

    for p in preds:
        best_iou = 0
        best_j = -1
        for j, g in enumerate(gts):
            if j in gt_used:
                continue
            iou = mask_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j != -1:
            gt_used.add(best_j)
            ious.append(best_iou)

    missed = len(gts) - len(gt_used)
    ious.extend([0.0] * missed)
    return sum(ious) / len(gts)


def val_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    iou_scores = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()

            model.eval()
            outputs = model(images)

            for pred, gt in zip(outputs, targets):
                pred_masks = pred["masks"].detach().cpu()
                gt_masks = gt["masks"].detach().cpu()
                iou_scores.append(image_iou(pred_masks, gt_masks))

    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    print(f"Val Loss: {total_loss:.4f}, IoU: {avg_iou:.4f}")
    return total_loss / len(loader), avg_iou


def _save_checkpoint_async(model, optimizer, epoch, val_loss, path):
    thread = threading.Thread(
        target=save_checkpoint,
        args=(model, optimizer, epoch, val_loss, path),
        daemon=True
    )
    thread.start()
    return thread


def main():
    cfg = Config()
    set_seed(42)

    if cfg.DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    print("[Setup] Loading datasets...")
    train_ds = CrackDataset(
        f"{cfg.DATA_DIR}/train/images",
        f"{cfg.DATA_DIR}/train/train.json",
        transforms=get_train_transforms(cfg.IMG_SIZE)
    )
    val_ds = CrackDataset(
        f"{cfg.DATA_DIR}/val/images",
        f"{cfg.DATA_DIR}/val/val.json",
        transforms=get_val_transforms(cfg.IMG_SIZE)
    )

    num_workers = 4 if cfg.DEVICE == "cuda" else 0
    use_pin_memory = cfg.DEVICE == "cuda"
    print(f"Device: {cfg.DEVICE} | Batch size: {cfg.BATCH_SIZE} | Workers: {num_workers}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=use_pin_memory)

    print("[Setup] Building model...")
    model = build_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    print("[Setup] Creating optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )

    log_path = "logs/training.jsonl"
    best_val_loss = float("inf")
    patience = 5
    epochs_no_improve = 0
    grad_accum_steps = getattr(cfg, 'GRAD_ACCUM_STEPS', 1)

    print(f"[Setup] Training for {cfg.NUM_EPOCHS} epochs | Grad accumulation: {grad_accum_steps} steps\n")

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"[Epoch {epoch+1}] Starting training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.DEVICE, scaler, grad_accum_steps)
        print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f}")

        val_loss, val_iou = val_one_epoch(model, val_loader, cfg.DEVICE)
        print(f"[Epoch {epoch+1}] Val loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

        scheduler.step()

        log_entry = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_iou":    round(val_iou, 4),
            "lr":         scheduler.get_last_lr()[0]
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"Epoch {epoch+1:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"iou={val_iou:.4f} | lr={log_entry['lr']:.2e}")

        if (epoch + 1) % 10 == 0:
            _save_checkpoint_async(
                model, optimizer, epoch, val_loss,
                f"{cfg.CHECKPOINT_DIR}/model_epoch_{epoch+1:03d}.pth"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            _save_checkpoint_async(
                model, optimizer, epoch, val_loss,
                f"{cfg.CHECKPOINT_DIR}/model_best.pth"
            )
            print(f"  ✓ New best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    print("\n[Training] Complete!")
    plot_training_curve(log_path, "logs/training_curve.png")
    print("[Setup] Done!")


if __name__ == "__main__":
    main()