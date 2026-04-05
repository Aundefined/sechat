"""
export_onnx.py

Exporta instruct_model.pt a ONNX para servir con onnxruntime (sin PyTorch).

Uso:
    python export_onnx.py

Genera: checkpoints/instruct_model.onnx
"""

import os
import sys
import torch
import torch.utils.checkpoint as cp_module

# Añadir src/ al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import GPT
from tokenizer import TiktokenWrapper

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "instruct_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "vocab.json")
ONNX_OUT   = os.path.join(BASE_DIR, "checkpoints", "instruct_model.onnx")

CONFIG = {
    "n_embd"     : 1024,
    "n_head"     : 16,
    "n_layer"    : 24,
    "block_size" : 1024,
    "dropout"    : 0.1,
}

# ── Parchear torch.utils.checkpoint para que no interfiera con el tracing ──
# Durante la exportación ONNX no necesitamos gradient checkpointing;
# lo reemplazamos por una llamada directa.
cp_module.checkpoint = lambda fn, *args, use_reentrant=False, **kwargs: fn(*args)

print("Cargando tokenizador...")
tokenizer = TiktokenWrapper()
tokenizer.load(VOCAB_PATH)

print(f"Cargando modelo desde {CHECKPOINT}...")
model = GPT(
    vocab_size  = tokenizer.vocab_size,
    n_embd      = CONFIG["n_embd"],
    n_head      = CONFIG["n_head"],
    n_layer     = CONFIG["n_layer"],
    block_size  = CONFIG["block_size"],
    dropout     = CONFIG["dropout"],
)

ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt.get("model_state", ckpt)

unwrapped = {}
for k, v in state_dict.items():
    clean_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
    unwrapped[clean_key] = v

model.load_state_dict(unwrapped)
model.eval()
print(f"Parámetros: {model.num_params():,}")

# ── Exportar ──
print(f"Exportando a {ONNX_OUT} ...")
dummy_input = torch.zeros(1, 16, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUT,
        input_names  = ["input_ids"],
        output_names = ["logits"],
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits":    {0: "batch", 1: "seq_len"},
        },
        opset_version = 17,
    )

print(f"✓ Exportado correctamente: {ONNX_OUT}")
size_mb = os.path.getsize(ONNX_OUT) / 1024 / 1024
print(f"  Tamaño: {size_mb:.0f} MB")
