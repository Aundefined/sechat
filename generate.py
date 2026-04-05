"""
generate.py

Carga el modelo entrenado y genera texto a partir de un prompt.

Parámetros de generación:
  temperature : controla la aleatoriedad. <1 más conservador, >1 más creativo.
  top_k       : solo muestrea entre los k tokens más probables.
  max_tokens  : número de tokens a generar.
"""

import os
import torch
import sys

# Añadir src/ al path para poder importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import GPT
from tokenizer import TiktokenWrapper


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "vocab.json")

MAX_TOKENS  = 100
TEMPERATURE = 0.2
TOP_K       = 10
REP_PENALTY = 1.3
PROMPT      = "La democracia es un sistema"


# ----------------------------------------------------------------------
# Generación
# ----------------------------------------------------------------------

@torch.no_grad()
def generate(model: GPT, tokenizer: TiktokenWrapper, prompt: str,
             max_tokens: int, temperature: float, top_k: int,
             device: str, repetition_penalty: float = 1.0) -> str:
    """
    Genera texto a partir de un prompt usando sampling con temperatura y top-k.

    Temperature:
      Divide los logits antes del softmax. Valores bajos (<1) hacen la
      distribución más peaked (el modelo elige lo más probable). Valores
      altos (>1) la aplanan (más aleatoriedad y creatividad).

    Top-k:
      Antes de samplear, descartamos todos los tokens excepto los k más
      probables. Evita que el modelo elija tokens raros con poca probabilidad.

    Repetition penalty:
      Valores > 1.0 penalizan tokens que ya aparecieron en la secuencia,
      reduciendo bucles repetitivos. 1.2–1.5 funciona bien.
    """
    model.eval()

    # Codificar el prompt
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_tokens):
        # Si la secuencia es más larga que block_size, recortamos por la derecha
        idx_cond = idx[:, -model.block_size:]

        # Forward pass — solo nos interesan los logits del último token
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Repetition penalty: penalizar tokens ya generados
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Aplicar temperatura
        logits = logits / temperature

        # Top-k: poner -inf a todo lo que no esté en el top-k
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float('-inf')

        # Softmax → probabilidades
        probs = torch.softmax(logits, dim=-1)

        # Samplear el siguiente token
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Añadir a la secuencia y continuar
        idx = torch.cat([idx, next_id], dim=1)

    # Decodificar toda la secuencia (prompt + generado)
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"Dispositivo: {device}")

    # Cargar tokenizador (vocabulario compacto)
    tokenizer = TiktokenWrapper()
    tokenizer.load(VOCAB_PATH)

    # Cargar checkpoint
    print(f"Cargando modelo desde {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg  = ckpt["config"]

    model = GPT(
        vocab_size = cfg["vocab_size"],
        n_embd     = cfg["n_embd"],
        n_head     = cfg["n_head"],
        n_layer    = cfg["n_layer"],
        block_size = cfg["block_size"],
        dropout    = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Modelo cargado (iter {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f})\n")

    # Generar
    print(f"Prompt: '{PROMPT}'")
    print(f"{'─' * 60}")
    text = generate(
        model       = model,
        tokenizer   = tokenizer,
        prompt      = PROMPT,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
        top_k       = TOP_K,
        device      = device,
        repetition_penalty = REP_PENALTY,
    )
    print(text)
    print(f"{'─' * 60}")
