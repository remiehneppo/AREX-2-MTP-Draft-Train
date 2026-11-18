import os
import subprocess
import shutil
from pathlib import Path
import argparse
from typing import Optional
import torch
import numpy as np

import gguf


def export_mtp_to_gguf(
    mtp_weights_path: str,
    base_mtp_gguf_path: str,
    target_gguf_path: Optional[str],
    output_f16_path: str,
    output_q4_path: str,
    llama_quantize_bin: str = "llama-quantize",
    keep_f16: bool = False,
):
    """
    Combines trained PyTorch MTP weights with GGUF metadata from base MTP and
    shared vocabulary/output weights from the target model (AREX-2) to produce
    a fully compliant llama.cpp sidecar MTP GGUF file.
    """
    print(f">>> Đang đọc cấu trúc từ base MTP: {base_mtp_gguf_path}")
    base_reader = gguf.GGUFReader(base_mtp_gguf_path)
    base_tensor_map = {t.name: t for t in base_reader.tensors}

    target_reader = None
    target_tensor_map = {}
    if target_gguf_path and os.path.exists(target_gguf_path):
        print(f">>> Đang đọc shared weights từ target model: {target_gguf_path}")
        target_reader = gguf.GGUFReader(target_gguf_path)
        target_tensor_map = {t.name: t for t in target_reader.tensors}

    print(f">>> Đang tải trọng số MTP PyTorch: {mtp_weights_path}")
    state_dict = torch.load(mtp_weights_path, map_location="cpu", weights_only=True)

    # Mapping from PyTorch module parameter names to GGUF tensor names
    name_map = {
        "hnorm.weight": "blk.64.nextn.hnorm.weight",
        "enorm.weight": "blk.64.nextn.enorm.weight",
        "eh_proj.weight": "blk.64.nextn.eh_proj.weight",
        "attn_norm.weight": "blk.64.attn_norm.weight",
        "q_proj.weight": "blk.64.attn_q.weight",
        "k_proj.weight": "blk.64.attn_k.weight",
        "v_proj.weight": "blk.64.attn_v.weight",
        "o_proj.weight": "blk.64.attn_output.weight",
        "attn_q_norm.weight": "blk.64.attn_q_norm.weight",
        "attn_k_norm.weight": "blk.64.attn_k_norm.weight",
        "attn_post_norm.weight": "blk.64.post_attention_norm.weight",
        "ffn_gate.weight": "blk.64.ffn_gate.weight",
        "ffn_up.weight": "blk.64.ffn_up.weight",
        "ffn_down.weight": "blk.64.ffn_down.weight",
        "shared_head_norm.weight": "blk.64.nextn.shared_head_norm.weight",
    }

    # Ensure output directory exists
    Path(output_f16_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_q4_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize GGUF Writer with qwen35 architecture
    writer = gguf.GGUFWriter(output_f16_path, arch="qwen35")

    # Copy architecture metadata from base_reader using contents()
    print(">>> Đang sao chép metadata kiến trúc...")
    for field in base_reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        if target_reader is not None and field.name.startswith("tokenizer."):
            continue
        try:
            val = field.contents()
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
        except Exception as e:
            print(f"Warning: Bỏ qua metadata {field.name}: {e}")

    if target_reader is not None:
        for field in target_reader.fields.values():
            if field.name.startswith("tokenizer."):
                value_type = field.types[0]
                element_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
                writer.add_key_value(field.name, field.contents(), value_type, element_type)

    # Ensure required MTP block count and nextn keys exist
    writer.add_uint32("qwen35.block_count", 65)
    writer.add_uint32("qwen35.nextn_predict_layers", 1)

    # 1. Add the 15 trained MTP core tensors in F16
    print(">>> Đang thêm 15 MTP core tensors đã huấn luyện...")
    for pt_name, gguf_name in name_map.items():
        dtype = np.float32 if "norm" in pt_name else np.float16
        if pt_name in state_dict:
            t_data = state_dict[pt_name].float().numpy().astype(dtype)
            writer.add_tensor(gguf_name, t_data)
        else:
            print(f"Cảnh báo: {pt_name} không có trong checkpoint, sao chép từ base GGUF")
            base_t = base_tensor_map[gguf_name]
            f16_data = gguf.quants.dequantize(base_t.data, base_t.tensor_type).astype(dtype)
            writer.add_tensor(gguf_name, f16_data)

    # 2. Add shared embedding & lm_head tensors (zero-copy quantized raw data)
    shared_tensors = ["token_embd.weight", "output.weight", "output_norm.weight"]
    for s_name in shared_tensors:
        if s_name in target_tensor_map:
            print(f">>> Đang sao chép shared tensor từ AREX-2: {s_name}")
            tgt_t = target_tensor_map[s_name]
            if tgt_t.tensor_type == 0:
                # F32 norm tensor
                writer.add_tensor(s_name, tgt_t.data.astype(np.float32))
            else:
                # Direct raw copy of quantized buffer
                writer.add_tensor(
                    s_name,
                    tgt_t.data,
                    raw_shape=tgt_t.data.shape,
                    raw_dtype=tgt_t.tensor_type,
                    tensor_endianess=target_reader.endianess
                )
        elif s_name in base_tensor_map:
            print(f">>> Đang sao chép shared tensor từ base MTP: {s_name}")
            base_t = base_tensor_map[s_name]
            if base_t.tensor_type == 0:
                writer.add_tensor(s_name, base_t.data.astype(np.float32))
            else:
                writer.add_tensor(
                    s_name,
                    base_t.data,
                    raw_shape=base_t.data.shape,
                    raw_dtype=base_t.tensor_type,
                    tensor_endianess=base_reader.endianess
                )
        else:
            raise KeyError(f"Shared tensor {s_name} không tìm thấy trong cả target và base GGUF!")

    print(f">>> Đang ghi file GGUF F16: {output_f16_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(">>> Xuất file F16 thành công!")

    # 3. Quantize to Q4_0
    quantize = shutil.which(llama_quantize_bin)
    if quantize:
        print(f">>> Đang lượng tử hóa sang Q4_0: {output_q4_path}...")
        cmd = [
            quantize,
            "--allow-requantize",
            "--leave-output-tensor",
            "--token-embedding-type", "q5_k",
            output_f16_path,
            output_q4_path,
            "Q4_0"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"llama-quantize thất bại (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}")

        print(f">>> HOÀN TẤT LƯỢNG TỬ HÓA: {output_q4_path}")
        # Clean up temporary F16 file to conserve disk space
        if not keep_f16 and os.path.exists(output_f16_path) and os.path.exists(output_q4_path):
            os.remove(output_f16_path)
            print(">>> Đã dọn dẹp file F16 trung gian.")
    else:
        print(f"Cảnh báo: Không tìm thấy {llama_quantize_bin}, giữ nguyên file F16.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best_mtp.pt checkpoint")
    parser.add_argument("--base-gguf", required=True, help="Path to base MTP GGUF file")
    parser.add_argument("--target-gguf", default=None, help="Path to AREX-2 target GGUF file")
    parser.add_argument("--out-f16", required=True, help="Output path for F16 GGUF")
    parser.add_argument("--out-q4", required=True, help="Output path for Q4_0 GGUF")
    parser.add_argument("--quantize-bin", default="llama-quantize")
    parser.add_argument("--keep-f16", action="store_true")
    args = parser.parse_args()

    export_mtp_to_gguf(
        mtp_weights_path=args.weights,
        base_mtp_gguf_path=args.base_gguf,
        target_gguf_path=args.target_gguf,
        output_f16_path=args.out_f16,
        output_q4_path=args.out_q4,
        llama_quantize_bin=args.quantize_bin,
        keep_f16=args.keep_f16,
    )
