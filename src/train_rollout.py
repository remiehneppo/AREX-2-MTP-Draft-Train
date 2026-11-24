"""Fine-tune the single MTP block on its own recurrent hidden states."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.model_mtp import Qwen35MTPBlock, gguf
from src.train_teacher import check_teacher_alignment, load_shared_weights, read_cache, split_sequences


def prepare_trace(sample, embedding):
    tokens, hidden, prompt_length = sample
    vectors = gguf.quants.dequantize(embedding.data[tokens[:-1]], embedding.tensor_type)
    shifted = np.zeros_like(hidden)
    shifted[1:] = hidden[:-1]
    return (
        torch.from_numpy(shifted),
        torch.from_numpy(vectors),
        torch.as_tensor(tokens[1:], dtype=torch.long),
        prompt_length - 1,
    )


def rollout(model, head, hidden, embeddings, targets, anchors, depth):
    # Each batch row has its own causal prefix. The first step uses teacher
    # hidden states. Later rows replace only the hidden states at draft positions.
    length = max(anchors) + depth
    anchor_ids = torch.tensor(anchors, device=hidden.device)
    batch_ids = torch.arange(len(anchors), device=hidden.device)
    states = hidden[None, :length].expand(len(anchors), -1, -1)
    embeddings = embeddings[None, :length].expand(len(anchors), -1, -1)
    matches = []
    losses = []
    for step in range(depth):
        positions = anchor_ids + step
        features = model(states, embeddings)[batch_ids, positions]
        logits = F.linear(features.half(), head)
        labels = targets[positions]
        losses.append(F.cross_entropy(logits, labels))
        matches.append(logits.argmax(-1) == labels)
        if step + 1 < depth:
            # A later token can be accepted only if all earlier draft tokens
            # equal the teacher tokens. Feed those IDs and the draft's own hidden.
            index = (positions + 1)[:, None, None].expand(-1, 1, states.shape[-1])
            states = states.scatter(1, index, features[:, None, :].to(states.dtype))
    prefixes = torch.stack(matches).cumprod(dim=0)
    return torch.stack(losses).mean(), prefixes.sum(dim=1)


def run_epoch(model, head, traces, depth, anchor_rng, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    accepted = [0] * depth
    loss_sum = 0.0
    count = 0
    for hidden, embeddings, targets, start in traces:
        eligible = list(range(start, len(targets) - depth + 1))
        if not eligible:
            continue
        if training:
            anchors = anchor_rng.sample(eligible, min(4, len(eligible)))
        else:
            indices = np.linspace(0, len(eligible) - 1, min(12, len(eligible)), dtype=int)
            anchors = [eligible[index] for index in indices]
        hidden, embeddings, targets = hidden.cuda(), embeddings.cuda(), targets.cuda()
        for offset in range(0, len(anchors), 4):
            batch = anchors[offset:offset + 4]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
                loss, counts = rollout(model, head, hidden, embeddings, targets, batch, depth)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite rollout loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(batch)
            accepted = [total + observed for total, observed in zip(accepted, counts.tolist())]
            count += len(batch)
    if not count:
        raise ValueError("No traces are long enough for rollout evaluation")
    return loss_sum / count, accepted, count


def report(name, metrics):
    loss, accepted, count = metrics
    rates = " ".join(f"P(≥{i})={value}/{count} ({value/count:.1%})"
                     for i, value in enumerate(accepted, 1))
    print(f"{name}: loss={loss:.4f} {rates}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(2, 3), default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    rng = random.Random(42)
    samples = list(read_cache(args.cache))
    train, validation = split_sequences(samples, args.prompts.read_text().splitlines(), rng)
    embedding, head = load_shared_weights(args.teacher)
    check_teacher_alignment(validation[0], args.teacher, head)
    train = [prepare_trace(sample, embedding) for sample in train]
    validation = [prepare_trace(sample, embedding) for sample in validation]
    model = Qwen35MTPBlock().cuda()
    model.load_state_dict(torch.load(args.init, weights_only=True, mmap=True, map_location="cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    print(f"Sequences: {len(train)} train / {len(validation)} validation, depth={args.depth}", flush=True)
    report("Initial validation", run_epoch(model, head, validation, args.depth, rng))

    best = None
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        report(f"Epoch {epoch} train", run_epoch(model, head, train, args.depth, rng, optimizer, scaler))
        metrics = run_epoch(model, head, validation, args.depth, rng)
        report(f"Epoch {epoch} validation", metrics)
        loss, accepted, _ = metrics
        rank = (sum(accepted[1:]), accepted[0], -loss)
        if best is None or rank > best:
            best = rank
            args.output.parent.mkdir(parents=True, exist_ok=True)
            staging = args.output.with_suffix(".tmp")
            torch.save(model.state_dict(), staging)
            staging.replace(args.output)
            print(f"Saved epoch {epoch} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
