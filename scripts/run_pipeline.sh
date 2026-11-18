#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
llama_dir=${LLAMA_CPP_DIR:-"$project_dir/../llama.cpp"}
python_bin=${MTP_PYTHON:-"$project_dir/../server-whisper-rocm/.venv/bin/python"}
export LD_LIBRARY_PATH="$llama_dir/build/bin:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.0.6}
export GGML_CUDA_NO_PINNED=1

mapfile -t model_paths < <("$python_bin" - <<'PY'
import yaml
with open("configs/training_config.yaml") as config_file:
    models = yaml.safe_load(config_file)["model"]
for key in ("target_gguf_path", "base_mtp_gguf_path", "raw_data_path"):
    print(models[key])
PY
)
teacher=${model_paths[0]}
warm_start=${model_paths[1]}
seed_prompts=${model_paths[2]}
cache_dir=data/cached_states
output_dir=output
mkdir -p "$cache_dir" "$output_dir"
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT

cache=${MTP_TEACHER_CACHE:-"$cache_dir/teacher.bin"}
prompt_manifest=${MTP_PROMPT_MANIFEST:-"$cache.prompts.txt"}
if [[ ! -s "$cache" ]]; then
    "$python_bin" - "$seed_prompts" "$work_dir/prompts.txt" "${MTP_PROMPT_COUNT:-64}" <<'PY'
import json
import random
import re
import sys
from pathlib import Path

prompts = [json.loads(line)["prompt"] for line in Path(sys.argv[1]).read_text().splitlines()]
# The seed file repeats prompt cycles with a numbered prefix.
prompts = list(dict.fromkeys(re.sub(r"^\[Variant \d+\] ", "", prompt) for prompt in prompts))
random.Random(42).shuffle(prompts)
selected = prompts[:int(sys.argv[3])]
Path(sys.argv[2]).write_text("\n".join(selected) + "\n")
PY
    "${CXX:-g++}" -O2 -std=c++17 -I"$llama_dir/include" -I"$llama_dir/src" \
        -I"$llama_dir/ggml/include" src/extract_teacher.cpp -L"$llama_dir/build/bin" \
        -lllama -o "$work_dir/extract_teacher"
    "$work_dir/extract_teacher" "$teacher" "$work_dir/prompts.txt" "$work_dir/teacher.bin"
    cp "$work_dir/prompts.txt" "$prompt_manifest"
    mv "$work_dir/teacher.bin" "$cache"
fi

previous=()
if [[ -f "$output_dir/best_mtp.pt" ]]; then
    previous=(--previous "$output_dir/best_mtp.pt")
fi
"$python_bin" -m src.train_teacher --cache "$cache" --prompts "$prompt_manifest" --teacher "$teacher" \
    --warm-start "$warm_start" "${previous[@]}" --output "$output_dir/teacher_mtp.pt" \
    --epochs "${MTP_EPOCHS:-4}" --learning-rate "${MTP_LEARNING_RATE:-3e-6}"

"$python_bin" -m src.export_gguf --weights "$output_dir/teacher_mtp.pt" \
    --base-gguf "$warm_start" --target-gguf "$teacher" --keep-f16 \
    --out-f16 "$output_dir/mtp-BAAI_AREX-2-teacher-F16.gguf" \
    --out-q4 "$output_dir/mtp-BAAI_AREX-2-teacher-Q4_0.gguf" \
    --quantize-bin "$llama_dir/build/bin/llama-quantize"
