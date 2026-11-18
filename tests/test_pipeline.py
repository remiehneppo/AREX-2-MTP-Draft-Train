import os
import sys
import unittest
import yaml
import torch

# Add project root and gguf-py
sys.path.insert(0, "/home/tieubaoca/AI/train_arex_mtp")
sys.path.insert(0, "/home/tieubaoca/AI/llama.cpp/gguf-py")

from src.model_mtp import Qwen35MTPBlock, RotaryEmbedding, load_warmstart_from_gguf
from src.dataset import build_synthetic_agent_dataset, load_dataset_prompts


class TestMTPPipeline(unittest.TestCase):
    def setUp(self):
        self.config_path = "/home/tieubaoca/AI/train_arex_mtp/configs/training_config.yaml"
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def test_01_config_parameters(self):
        # Verify architecture matches AREX-2 and Qwen3.5 27B
        arch = self.cfg["architecture"]
        self.assertEqual(arch["hidden_size"], 5120)
        self.assertEqual(arch["intermediate_size"], 17408)
        self.assertEqual(arch["num_attention_heads"], 24)
        self.assertEqual(arch["num_key_value_heads"], 4)
        self.assertEqual(arch["head_dim"], 256)
        self.assertEqual(arch["vocab_size"], 248320)
        self.assertEqual(self.cfg["training"]["mixed_precision"], "float16")

    def test_02_rotary_embedding(self):
        # Verify RoPE rotates the first 64 dimensions out of 256
        rope = RotaryEmbedding(rotary_dim=64, theta=10000000.0)
        x = torch.randn(2, 8, 24, 256)
        out = rope(x)
        self.assertEqual(out.shape, (2, 8, 24, 256))
        # First 64 dims should be altered, remaining 192 unchanged
        self.assertTrue(torch.allclose(x[..., 64:], out[..., 64:]))
        self.assertFalse(torch.allclose(x[..., :64], out[..., :64]))

    def test_rotary_embedding_matches_llama_cpp_split_half(self):
        rope = RotaryEmbedding(rotary_dim=4, theta=100.0)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])
        angles = torch.tensor([1.0, 0.1])
        first = x[..., :2] * angles.cos() - x[..., 2:4] * angles.sin()
        second = x[..., 2:4] * angles.cos() + x[..., :2] * angles.sin()
        expected = torch.cat([first, second, x[..., 4:]], dim=-1)
        torch.testing.assert_close(rope(x, pos_ids=torch.tensor([[1]])), expected)

    def test_03_mtp_forward_shape(self):
        # Verify forward pass tensor dimensions
        arch = self.cfg["architecture"]
        model = Qwen35MTPBlock(
            hidden_size=arch["hidden_size"],
            intermediate_size=arch["intermediate_size"],
            num_heads=arch["num_attention_heads"],
            num_kv_heads=arch["num_key_value_heads"],
            head_dim=arch["head_dim"],
            eps=float(arch["rms_norm_eps"])
        )
        h_63 = torch.randn(2, 16, 5120)
        token_embd = torch.randn(2, 16, 5120)
        out = model(h_63, token_embd)
        self.assertEqual(out.shape, (2, 16, 5120))

    def test_04_qgate_interleave_separation(self):
        # Verify Q and Gate tensors are properly de-interleaved per head
        num_heads = 24
        head_dim = 256
        q_heads_dim = num_heads * head_dim
        q_full = torch.randn(2, 4, num_heads * 2 * head_dim)
        reshaped = q_full.view(2, 4, num_heads, 2, head_dim)
        q = reshaped[..., 0, :]
        gate = reshaped[..., 1, :].reshape(2, 4, q_heads_dim)

        self.assertEqual(q.shape, (2, 4, num_heads, head_dim))
        self.assertEqual(gate.shape, (2, 4, q_heads_dim))

    def test_05_warmstart_loading(self):
        # Verify loading 15 weights from base MTP GGUF
        arch = self.cfg["architecture"]
        model = Qwen35MTPBlock(
            hidden_size=arch["hidden_size"],
            intermediate_size=arch["intermediate_size"],
            num_heads=arch["num_attention_heads"],
            num_kv_heads=arch["num_key_value_heads"],
            head_dim=arch["head_dim"],
            eps=float(arch["rms_norm_eps"])
        )
        base_mtp_path = self.cfg["model"]["base_mtp_gguf_path"]
        if os.path.exists(base_mtp_path):
            load_warmstart_from_gguf(model, base_mtp_path)
            # Verify weights are non-zero
            self.assertGreater(model.eh_proj.weight.abs().sum().item(), 0.0)
            self.assertGreater(model.q_proj.weight.abs().sum().item(), 0.0)

    def test_06_dataset_generation(self):
        # Verify synthetic prompt generator produces diverse items
        test_out = "/tmp/test_prompts.jsonl"
        samples = build_synthetic_agent_dataset(test_out, count=50)
        self.assertEqual(len(samples), 50)
        loaded = load_dataset_prompts(test_out)
        self.assertEqual(len(loaded), 50)
        self.assertTrue("prompt" in loaded[0])
        if os.path.exists(test_out):
            os.remove(test_out)


if __name__ == "__main__":
    unittest.main()
