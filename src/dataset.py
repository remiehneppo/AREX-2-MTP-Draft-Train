import os
import json
import subprocess
import shutil
import itertools
from typing import List, Dict, Optional

DEFAULT_TOKENIZE_BIN = "llama-tokenize"
DEFAULT_MODEL_GGUF = "models/BAAI_AREX-2-Q5_K_M.gguf"
DEFAULT_LIB_PATH = ""


def tokenize_text(
    text: str,
    model_path: str = DEFAULT_MODEL_GGUF,
    tokenize_bin: str = DEFAULT_TOKENIZE_BIN,
    lib_path: str = DEFAULT_LIB_PATH
) -> List[int]:
    """
    Tokenizes text directly using llama-tokenize and the target model vocab.
    """
    tokenizer = shutil.which(tokenize_bin)
    if not tokenizer or not os.path.exists(model_path):
        # Fallback to UTF-8 byte encoding as pseudo-token representation if binary not available
        return list(text.encode("utf-8"))

    env = os.environ.copy()
    if lib_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (lib_path, env.get("LD_LIBRARY_PATH"))))

    cmd = [
        tokenizer,
        "-m", model_path,
        "-p", text,
        "--ids"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {res.stderr}")

    tokens = []
    # llama-tokenize output format can be JSON-like array [1, 2, 3] or space-separated numbers
    raw = res.stdout.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            tokens = json.loads(raw)
            return [int(t) for t in tokens]
        except Exception:
            pass

    for part in raw.replace(",", " ").replace("[", " ").replace("]", " ").split():
        if part.isdigit():
            tokens.append(int(part))
    return tokens


def build_synthetic_agent_dataset(output_path: str, count: int = 1000):
    """
    Generates a high-quality multi-domain synthetic dataset covering:
    Python algorithms, distributed systems, code refactoring, system design, and agent tool calling.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    algos = [
        ("A* Search Algorithm", "heuristic admissible distance and priority queue optimization"),
        ("Dijkstra Shortest Path", "fibonacci heap and graph adjacency list representation"),
        ("LRU Cache with TTL", "O(1) double linked list with automatic time expiration"),
        ("Trie Prefix Tree", "prefix matching, wildcards, and memory-efficient node compression"),
        ("Consistent Hashing Ring", "virtual nodes to minimize migration on node additions"),
        ("Red-Black Tree", "self-balancing binary search tree with left and right rotations"),
        ("Bitmask Dynamic Programming", "Traveling Salesman Problem state compression"),
        ("Segment Tree with Lazy Propagation", "range update and range sum queries in O(log N)"),
        ("KMP Substring Search", "longest prefix-suffix table construction and O(N+M) match"),
        ("Raft Consensus Algorithm", "leader election, log replication, and safety invariants"),
        ("SkipList Index", "probabilistic balancing with multi-level forward pointers"),
        ("Lock-Free Ring Buffer", "atomic compare-and-swap operations for high-throughput SPSC"),
        ("B+ Tree Storage Engine", "disk-friendly page splits, merges, and leaf node chaining"),
        ("HyperLogLog Cardinality Estimator", "64-bit MurmurHash with harmonic mean register estimation"),
        ("Bloom Filter with Double Hashing", "false positive probability bound with bitarray"),
    ]

    systems_topics = [
        ("distributed transaction coordinator", "Two-Phase Commit (2PC) protocol with timeout handling"),
        ("asynchronous task queue", "Redis Streams, consumer groups, and dead-letter queues"),
        ("rate limiter middleware", "token bucket algorithm with distributed Redis sliding window"),
        ("metrics aggregation pipeline", "exponential histogram downsampling and write-ahead log"),
        ("event-driven CQRS event store", "event versioning, optimistic locking, and snapshotting"),
        ("sharded database router", "consistent hashing key derivation and connection pooling"),
    ]

    refactoring_cases = [
        ("multiprocessing pool execution", "replacing synchronous file I/O with asyncio executor"),
        ("memory-intensive pandas pipeline", "vectorized polars expressions and zero-copy Arrow memory"),
        ("SQL N+1 query problem", "eager joins, batch fetching, and index optimization"),
        ("race condition in payment checkout", "pessimistic row-level locking with retry idempotency"),
        ("deep nested callback hierarchy", "async/await coroutines with structured cancellation"),
    ]

    schemas = [
        ("E-commerce multi-vendor platform", "Stripe payment intents, escrow, and idempotent webhook handling"),
        ("Enterprise RBAC permission system", "hierarchical roles, fine-grained ACLs, and JWT claims"),
        ("Real-time collaborative document editor", "Operational Transformation / CRDT sync protocol"),
        ("AI Agent task execution engine", "tool definitions, execution trace graphs, and state checkpointing"),
    ]

    templates = [
        # Domain 1: Algorithmic Implementation
        lambda item: f"Implement the {item[0]} in Python with complete type hints, docstrings, and handling of {item[1]}.",
        lambda item: f"Write a high-performance Python implementation of {item[0]}. Focus especially on {item[1]}.",
        # Domain 2: Distributed Systems
        lambda item: f"Explain how to design and architect a {item[0]} considering {item[1]}. Provide production-ready code examples.",
        # Domain 3: Code Review & Refactoring
        lambda item: f"Perform a comprehensive code review to address {item[0]}. Demonstrate the refactored code using {item[1]}.",
        # Domain 4: API & System Design
        lambda item: f"Design a robust REST and WebSocket architecture for {item[0]} incorporating {item[1]}."
    ]

    samples = []
    idx = 0

    # Generate combinatorial samples across all categories
    categories = [
        (algos, [templates[0], templates[1]]),
        (systems_topics, [templates[2]]),
        (refactoring_cases, [templates[3]]),
        (schemas, [templates[4]])
    ]

    cycle_count = 0
    while len(samples) < count:
        cycle_count += 1
        for items, tmpl_funcs in categories:
            for item in items:
                for tmpl in tmpl_funcs:
                    prompt = tmpl(item)
                    if cycle_count > 1:
                        prompt = f"[Variant {cycle_count}] " + prompt
                    samples.append({"id": idx, "prompt": prompt})
                    idx += 1
                    if len(samples) >= count:
                        break
                if len(samples) >= count:
                    break
            if len(samples) >= count:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f">>> Đã sinh thành công {len(samples)} mẫu huấn luyện đa dạng tại: {output_path}")
    return samples


def load_dataset_prompts(file_path: str) -> List[Dict]:
    """
    Loads JSONL prompts file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tập dữ liệu không tồn tại: {file_path}")
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


if __name__ == "__main__":
    out_file = "data/raw/seed_prompts.jsonl"
    build_synthetic_agent_dataset(out_file, count=500)
