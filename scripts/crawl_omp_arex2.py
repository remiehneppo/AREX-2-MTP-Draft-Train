#!/usr/bin/env python3
"""Export local AREX-2 request prefixes from OMP and Pi session logs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


def visible_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"].strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    )


def clip_text(text, limit):
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    side = max(0, (limit - 7) // 2)
    if not side:
        return ""
    return raw[:side].decode("utf-8", "ignore") + " [...] " + raw[-side:].decode("utf-8", "ignore")


def context_message(message):
    role = message.get("role")
    text = visible_text(message)
    if role == "toolResult":
        name = message.get("toolName") or "tool"
        label = f"Tool ({name}, error)" if message.get("isError") else f"Tool ({name})"
        return {"role": "tool", "name": name, "content": text}, label
    if role not in {"user", "assistant"}:
        return None
    content = message.get("content")
    if role == "assistant" and isinstance(content, list):
        calls = [
            str(block["name"]) for block in content
            if isinstance(block, dict) and block.get("type") == "toolCall" and block.get("name")
        ]
        if calls:
            text = "\n".join(part for part in (text, "Tool calls: " + ", ".join(calls)) if part)
    return {"role": role, "content": text}, role.capitalize()


def build_prefix(entries, parent_id, user_id, history_limit, budget):
    user = entries[user_id]["message"]
    boundary = " | Assistant:"
    latest = []
    visited = set()
    newer_count = None
    while parent_id and parent_id in entries and len(latest) < history_limit - 1:
        if parent_id in visited:
            raise ValueError("cycle in session ancestry")
        visited.add(parent_id)
        entry = entries[parent_id]
        if entry.get("type") == "reset_boundary":
            break
        if parent_id == user_id:
            newer_count = len(latest)
        elif entry.get("type") == "message":
            turn = context_message(entry["message"])
            if turn and turn[0]["content"]:
                latest.append(turn)
        parent_id = entry.get("parentId")

    def render(turn, label, limit):
        content = clip_text(turn["content"], limit)
        part = f'{label}: {" ".join(content.split())}'
        return {**turn, "content": content}, part

    available = budget - len(boundary)
    # Reserve space for the actual task and allocate the rest to the most
    # recent context, prioritizing the tool result immediately before draft.
    user_limit = available - len("User: ")
    if latest:
        user_limit = min(user_limit, max(64, available // 2))
    anchor, anchor_text = render({"role": "user", "content": visible_text(user)}, "User", user_limit)
    available -= len(anchor_text.encode("utf-8"))
    picked = []
    for i, (turn, label) in enumerate(latest):
        limit = min(768, available - len(label.encode("utf-8")) - len(" | : "))
        if i + 1 < len(latest) and limit >= 192:
            limit -= 96
        if limit < 32:
            break
        item, part = render(turn, label, limit)
        if item["content"]:
            picked.append((item, part))
            available -= len(part.encode("utf-8")) + len(" | ")

    index = min(len(picked), newer_count if newer_count is not None else len(latest))
    picked.insert(index, (anchor, anchor_text))
    picked.reverse()
    return [message for message, _ in picked], " | ".join(text for _, text in picked) + boundary


CONTEXT_FIELDS = {
    "compaction": ("summary", "firstKeptEntryId", "systemMessage"),
    "context_edit": ("targetId", "replacement"),
    "custom_message": ("customType", "content", "display"),
    "reset_boundary": (),
    "ttsr_injection": ("injectedRules",),
}
MESSAGE_FIELDS = (
    "role", "content", "sections", "toolsAdded", "toolCallId", "toolName",
    "isError", "synthetic", "prunedAt", "useless",
)


def select_fields(record, fields):
    return {key: record[key] for key in fields if key in record}


def structured_context(entries, parent_id):
    context = []
    visited = set()
    while parent_id is not None:
        if parent_id in visited:
            raise ValueError("cycle in session ancestry")
        if parent_id not in entries:
            raise ValueError(f"missing session ancestor: {parent_id}")
        visited.add(parent_id)
        entry = entries[parent_id]
        kind = entry.get("type")
        record = {"id": parent_id, "type": kind}
        if kind == "message":
            record["message"] = select_fields(entry["message"], MESSAGE_FIELDS)
            context.append(record)
        elif kind in CONTEXT_FIELDS:
            record.update(select_fields(entry, CONTEXT_FIELDS[kind]))
            context.append(record)
        parent_id = entry.get("parentId")
    return context[::-1]


def session_prompts(path, history_limit, budget, stride, per_turn, source="omp"):
    entries = {}
    anchors = {}
    candidates = defaultdict(list)
    session_id = None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            kind = entry.get("type")
            if kind == "session":
                session_id = entry.get("id")
            entry_id = entry.get("id")
            if not entry_id:
                continue
            entries[entry_id] = entry
            parent_id = entry.get("parentId")
            message = entry.get("message", {})
            role = message.get("role")
            if kind == "reset_boundary":
                continue
            user_id = entry_id if kind == "message" and role == "user" else anchors.get(parent_id)
            if user_id:
                anchors[entry_id] = user_id
            if (
                session_id and user_id and kind == "message" and role == "assistant"
                and message.get("provider") == "local"
                and message.get("model") in {"arex-2", "local/arex-2"}
                and message.get("stopReason") not in {"error", "aborted"}
            ):
                candidates[user_id].append(entry_id)

    for user_id, targets in candidates.items():
        if not visible_text(entries[user_id]["message"]):
            continue
        indices = sorted(set(range(0, len(targets), stride)) | {len(targets) - 1})
        if per_turn == 1:
            indices = indices[:1]
        elif len(indices) > per_turn:
            last = len(indices) - 1
            indices = [indices[i * last // (per_turn - 1)] for i in range(per_turn)]
        for index in indices:
            target_id = targets[index]
            target = entries[target_id]
            parent_id = target.get("parentId")
            context = structured_context(entries, parent_id)
            history, prompt = build_prefix(entries, parent_id, user_id, history_limit, budget)
            yield {
                "model": "local/arex-2",
                "source": source,
                "session_id": session_id,
                "target_entry_id": target_id,
                "target_metadata": select_fields(
                    target["message"], ("stopReason", "contextSnapshot", "thinkingLevel", "api"),
                ),
                "context_entries": context,
                "messages": history,
                "prompt": prompt,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("omp", "pi"), default="omp")
    parser.add_argument("--sessions", type=Path, help="override the selected agent's session directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-messages", type=int, default=12,
                        help="maximum messages for the legacy one-line prompt")
    parser.add_argument("--context-bytes", type=int, default=2000,
                        help="maximum UTF-8 bytes for the legacy one-line prompt")
    parser.add_argument("--tool-stride", type=int, default=4,
                        help="sample every N matching assistant responses within a user turn")
    parser.add_argument("--max-per-turn", type=int, default=24,
                        help="maximum prefixes per user turn, spread across the turn")
    parser.add_argument("--dry-run", action="store_true", help="count matching prompts without writing")
    args = parser.parse_args()
    if args.sessions is None:
        args.sessions = Path.home() / f".{args.source}/agent/sessions"
    if args.output is None:
        args.output = Path(f"data/cached_states/{args.source}-arex2-prompts.jsonl")
    if args.history_messages < 1:
        parser.error("--history-messages must be at least 1")
    if args.context_bytes < 256 or args.tool_stride < 1 or args.max_per_turn < 1:
        parser.error("context-bytes must be at least 256 and sampling limits must be positive")
    if not args.sessions.is_dir():
        parser.error(f"session directory not found: {args.sessions}")

    def rows():
        seen = set()
        for path in sorted(args.sessions.rglob("*.jsonl")):
            if path.name == "__advisor.jsonl":
                continue
            for row in session_prompts(
                path, args.history_messages, args.context_bytes, args.tool_stride,
                args.max_per_turn, args.source,
            ):
                key = (row["session_id"], row["target_entry_id"])
                if key not in seen:
                    seen.add(key)
                    yield row

    def consume(output=None):
        sessions = set()
        count = 0
        for row in rows():
            if output is not None:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            sessions.add(row["session_id"])
            count += 1
        return count, len(sessions)

    if args.dry_run:
        count, sessions = consume()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            parser.error(f"output already exists: {args.output}; choose a new path")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                count, sessions = consume(output)
        except BaseException:
            args.output.unlink()
            raise
        print(f"Wrote {args.output}")
    print(f"{count} request prefixes from {sessions} {args.source.upper()} sessions")


if __name__ == "__main__":
    main()
