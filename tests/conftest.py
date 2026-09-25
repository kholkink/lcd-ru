import copy

import pytest
import torch

TOKENIZER = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def tok():
    from src.models import load_tokenizer

    return load_tokenizer(TOKENIZER)


def _tiny(arch: str, vocab: int, dtype=torch.float32, seed: int = 0, layers: int = 3):
    from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

    torch.manual_seed(seed)
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
    cfg = Qwen2Config(**common) if arch == "qwen" else LlamaConfig(**common)
    cfg._attn_implementation = "sdpa"
    m = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    # random init is tiny (std 0.02): scale up so logits are not flat and tests are meaningful
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
    return m.eval()


@pytest.fixture
def tiny(tok):
    def make(arch="qwen", dtype=torch.float32, seed=0, layers=3):
        return _tiny(arch, len(tok), dtype, seed, layers)

    return make


@pytest.fixture
def clone():
    return copy.deepcopy
