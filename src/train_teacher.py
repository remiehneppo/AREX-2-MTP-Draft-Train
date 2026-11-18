"""Train the MTP sidecar on traces from extract_teacher.cpp."""
import argparse
import random
import re
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model_mtp import Qwen35MTPBlock, gguf, load_warmstart_from_gguf


def read_cache(path):
    with path.open("rb") as cache:
        width, = struct.unpack("<i", cache.read(4))
        if width != 5120:
            raise ValueError(f"Unexpected hidden width: {width}")
        while header := cache.read(8):
            count, prompt_length = struct.unpack("<ii", header)
            tokens = np.frombuffer(cache.read(count * 4), dtype="<i4").copy()
            hidden = np.frombuffer(cache.read((count - 1) * width * 4), dtype="<f4")
            hidden = hidden.reshape(count - 1, width).copy()
            if len(tokens) != count or not 1 <= prompt_length < count:
                raise ValueError("Invalid teacher cache sequence")
            if not np.isfinite(hidden).all():
                raise ValueError("Teacher hidden states contain NaN or Inf")
            yield tokens, hidden, prompt_length


def prompt_family(prompt):
    prompt = re.sub(r"^\[Variant \d+\] ", "", prompt)
    implementation = re.match(
        r"(?:Implement the (.+?) in Python with|Write a high-performance Python implementation of (.+?)\. Focus)",
        prompt,
    )
    return next(filter(None, implementation.groups())) if implementation else prompt


def split_sequences(samples, prompts, rng):
    if len(samples) != len(prompts):
        raise ValueError("Prompt manifest and teacher cache have different sequence counts")
    groups = {}
    for sample, prompt in zip(samples, prompts):
        tokens, _, prompt_length = sample
        if len(tokens) - prompt_length >= 2:
            groups.setdefault(prompt_family(prompt), []).append(sample)
    families = list(groups)
    if len(families) < 12:
        raise ValueError("Need at least 12 prompt families with teacher continuations")
    rng.shuffle(families)
    split = max(4, len(families) // 5)
    validation = [sample for family in families[:split] for sample in groups[family]]
    train = [sample for family in families[split:] for sample in groups[family]]
    return train, validation


def load_shared_weights(path):
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(str(path)).tensors}
    head = tensors["output.weight"]
    head_gpu = torch.empty(tuple(reversed(head.shape)), dtype=torch.float16, device="cuda")
    # Dequantize in row chunks so the full float32 head never occupies host RAM.
    for start in range(0, head_gpu.shape[0], 2048):
        block = gguf.quants.dequantize(head.data[start:start + 2048], head.tensor_type)
        head_gpu[start:start + len(block)].copy_(torch.from_numpy(block))
    return tensors["token_embd.weight"], head_gpu


@torch.inference_mode()
def check_teacher_alignment(sample, teacher, head):
    tokens, hidden, prompt_length = sample
    output_norm = next(tensor for tensor in gguf.GGUFReader(str(teacher)).tensors
                       if tensor.name == "output_norm.weight")
    weight = torch.from_numpy(output_norm.data.copy()).cuda()
    features = torch.from_numpy(hidden[prompt_length - 1:]).cuda()
    features = F.rms_norm(features, (features.shape[-1],), weight, eps=1e-6)
    predicted = torch.cat([F.linear(chunk.half(), head).argmax(-1) for chunk in features.split(8)])
    labels = torch.as_tensor(tokens[prompt_length:], device="cuda")
    matches = (predicted == labels).sum().item()
    print(f"Teacher hidden/label alignment: {matches}/{len(labels)}", flush=True)
    if matches / len(labels) < 0.9:
        raise ValueError("Teacher hidden states and continuation tokens do not align")


def measure(model, samples, embedding, head, optimizer=None, scaler=None):
    total_loss = 0.0
    correct = 0
    total = 0
    training = optimizer is not None
    model.train(training)
    for tokens, hidden, prompt_length in samples:
        input_ids = tokens[:-1]
        token_vectors = gguf.quants.dequantize(embedding.data[input_ids], embedding.tensor_type)
        # The runtime pairs token at position j with teacher hidden state j-1.
        # Position zero receives the initial all-zero pending hidden state.
        shifted = np.zeros_like(hidden)
        shifted[1:] = hidden[:-1]
        start = max(prompt_length - 1, 0)
        targets = torch.as_tensor(tokens[start + 1:], dtype=torch.long, device="cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            output = model(torch.from_numpy(shifted).cuda()[None],
                           torch.from_numpy(token_vectors).cuda()[None])[0, start:]
            # Backpropagate through short vocabulary projections independently,
            # then pass their combined gradient into the MTP graph once.
            features = output.detach().requires_grad_(training)
            for index in range(0, len(targets), 8):
                labels = targets[index:index + 8]
                logits = F.linear(features[index:index + len(labels)].half(), head)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite token loss")
                if training:
                    scaler.scale(loss / len(targets)).backward()
                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)
            if training:
                output.backward(features.grad)
        if training:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    return total_loss / total, correct / total, total


def report(label, metrics):
    loss, accuracy, count = metrics
    print(f"{label}: loss={loss:.4f} top1={accuracy:.2%} n={count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    rng = random.Random(42)
    samples = list(read_cache(args.cache))
    train, validation = split_sequences(samples, args.prompts.read_text().splitlines(), rng)
    print(f"Teacher sequences: {len(train)} train / {len(validation)} validation", flush=True)
    embedding, head = load_shared_weights(args.teacher)
    check_teacher_alignment(validation[0], args.teacher, head)
    model = Qwen35MTPBlock()

    def evaluate(label, batch):
        metrics = measure(model, batch, embedding, head)
        report(label, metrics)
        return metrics

    if args.previous:
        model.load_state_dict(torch.load(args.previous, weights_only=True, mmap=True, map_location="cpu"))
        model.cuda()
        evaluate("old checkpoint validation", validation)
        model.cpu()
    load_warmstart_from_gguf(model, args.warm_start)
    model.cuda()
    warm_start = evaluate("Qwen warm-start validation", validation)

    probe = train[:4]
    evaluate("alignment probe before", probe)
    probe_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, fused=False)
    probe_scaler = torch.amp.GradScaler("cuda", init_scale=128)
    for _ in range(4):
        measure(model, probe, embedding, head, probe_optimizer, probe_scaler)
    evaluate("alignment probe after", probe)
    del probe_optimizer, probe_scaler

    # Start the actual run from the measured baseline. The probe deliberately
    # overfits a few sequences and should have no influence on its optimizer.
    load_warmstart_from_gguf(model, args.warm_start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    best = (-1.0, float("-inf"))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        report(f"epoch {epoch} train", measure(model, train, embedding, head, optimizer, scaler))
        loss, accuracy, _ = evaluate(f"epoch {epoch} validation", validation)
        rank = (accuracy, -loss)
        if rank > best:
            best = rank
            staging = args.output.with_suffix(".tmp")
            torch.save(model.state_dict(), staging)
            staging.replace(args.output)
    print(f"Best validation top-1 {best[0]:.2%}; Qwen warm-start {warm_start[1]:.2%}", flush=True)


if __name__ == "__main__":
    main()
