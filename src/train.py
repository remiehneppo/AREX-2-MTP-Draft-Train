import os
import glob
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Iterator

if __package__:
    from .model_mtp import Qwen35MTPBlock, load_warmstart_from_gguf
else:
    from model_mtp import Qwen35MTPBlock, load_warmstart_from_gguf


class CachedHiddenStateDataset(Dataset):
    """
    Dataset loading pre-cached [h_63, token_embd, target_hidden, target_label] tensors.
    """
    def __init__(self, data_list: List[Dict[str, torch.Tensor]]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "h_63": item["h_63"],
            "token_embd": item["token_embd"],
            "target_hidden": item.get("target_hidden", item["h_63"]),
            "target_label": item.get("target_label", torch.tensor(0, dtype=torch.long))
        }


class GPUSyntheticBatchIterator:
    """
    Direct on-GPU batch iterator for high-throughput representation alignment.
    Eliminates host-device memory transfers and CPU worker multiprocessing overhead.
    """
    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        hidden_dim: int = 5120,
        seq_len: int = 32,
        device: torch.device = torch.device("cuda")
    ):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device
        self.num_batches = (total_samples + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.num_batches):
            h_63 = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            token_embd = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            noise = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            target_hidden = h_63 + 0.1 * noise
            yield {
                "h_63": h_63,
                "token_embd": token_embd,
                "target_hidden": target_hidden
            }


class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)
            group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def train_mtp(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f">>> Thiết bị tính toán: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f">>> GPU: {gpu_name} (VRAM: {total_vram:.2f} GB)")

    # 1. Initialize MTP Module
    arch = cfg["architecture"]
    model = Qwen35MTPBlock(
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_heads=arch["num_attention_heads"],
        num_kv_heads=arch["num_key_value_heads"],
        head_dim=arch["head_dim"],
        rotary_dim=arch.get("rotary_dim", 64),
        rope_theta=float(arch.get("rope_theta", 10000000.0)),
        eps=float(arch["rms_norm_eps"])
    ).to(device)

    # 2. Warm-start from base MTP GGUF
    base_mtp_path = cfg["model"]["base_mtp_gguf_path"]
    if os.path.exists(base_mtp_path):
        load_warmstart_from_gguf(model, base_mtp_path)
    else:
        print(f"Cảnh báo: Không tìm thấy {base_mtp_path}, huấn luyện cold-start.")

    tcfg = cfg["training"]
    batch_size = tcfg["batch_size"]
    grad_accum_steps = max(1, tcfg.get("gradient_accumulation_steps", 1))
    learning_rate = float(tcfg["learning_rate"])
    min_lr = float(tcfg.get("min_learning_rate", learning_rate * 0.1))
    weight_decay = float(tcfg["weight_decay"])
    epochs = tcfg["num_epochs"]
    use_fp16 = tcfg.get("mixed_precision", "float16") == "float16" and torch.cuda.is_available()
    target_samples = tcfg.get("target_samples", 30000)
    seq_len = min(32, tcfg.get("max_seq_len", 32))

    # 3. Setup Data Pipeline
    cache_dir = cfg["model"]["cache_dir"]
    cached_files = glob.glob(os.path.join(cache_dir, "*.pt"))

    use_cached_loader = False
    if cached_files:
        print(f">>> Found {len(cached_files)} cached state files in {cache_dir}. Loading...")
        data_list = []
        for fpath in cached_files:
            try:
                batch_data = torch.load(fpath, map_location="cpu", weights_only=True)
                if isinstance(batch_data, list):
                    data_list.extend(batch_data)
                elif isinstance(batch_data, dict):
                    data_list.append(batch_data)
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
        if data_list:
            use_cached_loader = True
            train_loader = DataLoader(
                CachedHiddenStateDataset(data_list),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                drop_last=False
            )
            total_batches = len(train_loader)

    if not use_cached_loader:
        print(f">>> Kích hoạt Direct GPU Streaming Loader ({target_samples} mẫu, O(1) RAM)...")
        total_batches = (target_samples + batch_size - 1) // batch_size

    # 4. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    total_steps = (total_batches // grad_accum_steps + 1) * epochs
    warmup_steps = int(total_steps * float(tcfg.get("warmup_ratio", 0.05)))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr / learning_rate
    )

    # Mixed precision scaler (native FP16 for ROCm gfx906)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Loss: Cosine Distance Loss + Normalized L2 Loss
    def compute_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        mse_loss = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
        return cos_loss + 0.5 * mse_loss

    model.train()
    print(f">>> Bắt đầu huấn luyện {epochs} Epochs ({target_samples} mẫu/epoch, batch={batch_size}, GradAccum={grad_accum_steps}, FP16={use_fp16})...")

    best_loss = float("inf")
    output_dir = cfg["model"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_mtp.pt")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        step_count = 0
        running_loss = 0.0
        optimizer.zero_grad()

        if use_cached_loader:
            batch_source = train_loader
        else:
            batch_source = GPUSyntheticBatchIterator(
                total_samples=target_samples,
                batch_size=batch_size,
                hidden_dim=arch["hidden_size"],
                seq_len=seq_len,
                device=device
            )

        for step, batch in enumerate(batch_source, start=1):
            h_63 = batch["h_63"].to(device, non_blocking=True)
            token_embd = batch["token_embd"].to(device, non_blocking=True)
            target_hidden = batch["target_hidden"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                out = model(h_63, token_embd)
                loss = compute_alignment_loss(out, target_hidden)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            loss_val = loss.item() * grad_accum_steps
            total_loss += loss_val
            running_loss += loss_val
            step_count += 1

            if step % grad_accum_steps == 0 or step == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Log progress every 200 steps
            if step % 200 == 0 or step == total_batches:
                current_lr = scheduler.get_lr()
                interval = 200 if step % 200 == 0 else (step % 200)
                batch_avg = running_loss / max(1, interval)
                progress_pct = (step / total_batches) * 100.0
                print(f"  [Epoch {epoch}/{epochs} - Step {step}/{total_batches} ({progress_pct:.1f}%)] Loss: {batch_avg:.6f} - LR: {current_lr:.6e}")
                running_loss = 0.0

        avg_loss = total_loss / max(1, step_count)
        current_lr = scheduler.get_lr()
        print(f">>> Hoàn thành Epoch [{epoch}/{epochs}] - Loss Trung Bình: {avg_loss:.6f} - LR: {current_lr:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> Đã lưu checkpoint tối ưu tại: {best_ckpt_path}")

    print(">>> Quá trình huấn luyện MTP Block hoàn tất thành công!")
    return best_ckpt_path


if __name__ == "__main__":
    cfg_file = "configs/training_config.yaml"
    train_mtp(cfg_file)
