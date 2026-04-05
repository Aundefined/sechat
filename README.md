# Sechat — GPT entrenado desde cero

Transformer decoder-only de **400M parámetros** implementado en PyTorch puro, preentrenado sobre 7.2B tokens en español y ajustado con SFT Instruct. Sin modelos preentrenados, sin Hugging Face.

🖥️ **[Probar el modelo en vivo](https://arnaud-tartulier.com)**

---

## Arquitectura

| Parámetro | Valor |
|---|---|
| Parámetros | 400M |
| Capas (`n_layer`) | 24 |
| Cabezas de atención (`n_head`) | 16 |
| Dimensión de embedding (`n_embd`) | 1024 |
| Tamaño de contexto (`block_size`) | 1024 |
| Tokenizador | tiktoken `cl100k_base` (vocab compacto 94.770 tokens) |

**Detalles de implementación:**
- Atención multi-cabeza con Flash Attention (`F.scaled_dot_product_attention`)
- Gradient checkpointing para eficiencia de memoria
- Weight tying entre token embedding y LM head
- BF16 mixed precision + `torch.compile`
- Gradient accumulation (batch efectivo 128)

---

## Entrenamiento

### Pretraining

- **Corpus:** Wikipedia en español + CulturaX (~7.2B tokens)
- **Hardware:** Google Colab (RTX 6000 Pro)
- **val_loss final:** 2.4301

### SFT Instruct

Fine-tuning supervisado sobre el modelo base usando el dataset [Alpaca Spanish](https://huggingface.co/datasets/bertin-project/alpaca-spanish).

**Formato instrucción-respuesta:**
```
### Instrucción:
{instrucción}
### Respuesta:
{respuesta}
###
```

**Hiperparámetros SFT:**
- Learning rate: `1e-5` (30x menor que el pretraining)
- Epochs: 3
- Gradient accumulation: 2

---

## Estructura del repositorio

```
sechat/
├── src/
│   ├── model.py          # Arquitectura GPT completa
│   └── tokenizer.py      # TiktokenWrapper (vocab compacto)
├── trainings/
│   ├── base_training/
│   │   └── train_colab_v2.ipynb   # Notebook de pretraining
│   └── instruct-training/
│       └── sft_instruct_colab.ipynb  # Notebook de SFT
├── chat_sft.py           # Chat interactivo por terminal
├── export_onnx.py        # Exportación del modelo a ONNX
└── generate.py           # Generación de texto
```

---

## Uso

### Chat interactivo

```bash
python chat_sft.py
```

Requiere `checkpoints/instruct_model.pt` y `data/vocab.json`.

### Exportar a ONNX

```bash
python export_onnx.py
```

Genera `checkpoints/instruct_model.onnx` para servir con `onnxruntime` sin depender de PyTorch.

---

## Dependencias

```bash
pip install torch tiktoken numpy onnx
```

---

## Notas

Este es un modelo experimental con fines educativos. Sus respuestas pueden ser incorrectas, incoherentes o directamente falsas. No tomes ninguna decisión basándote en lo que diga.
