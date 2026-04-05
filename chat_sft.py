"""
chat_sft.py

Inicia un chat interactivo por terminal con el modelo SFT Instruct (instruct_model.pt).
Permite conversar de forma continua sin tener que reiniciar el script.
Escribe 'salir' o 'exit' para terminar la conversación.
"""

import os
import sys
import torch

# Añadir src/ al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import GPT
from tokenizer import TiktokenWrapper

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "instruct_model.pt")
#CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "personalized_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "vocab.json")
USE_HISTORY = True

# Configuración del modelo (debe coincidir con la usada en el entrenamiento)
CONFIG = {
    "n_embd"      : 1024,
    "n_head"      : 16,
    "n_layer"     : 24,
    "block_size"  : 1024,
    "dropout"     : 0.1,
}

# Parámetros de generación por defecto
GEN_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.2,
    "top_k": 20,
    "repetition_penalty": 1.15
}

# ----------------------------------------------------------------------
# Lógica de Generación SFT
# ----------------------------------------------------------------------

@torch.no_grad()
def generate_sft(model, tokenizer, prompt, max_new_tokens=200, temperature=0.6, top_k=30, repetition_penalty=1.1, device="cpu"):
    """
    Genera una respuesta en formato instruct a partir del prompt dado.
    """
    model.eval()
        
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Recortar el contexto si excede el block_size
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Penalización por repetición
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
                    
        logits = logits / temperature
        
        # Top-K sampling
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float('-inf')
            
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        
        # Comprobar condición de parada (el modelo genera el delimitador de fin)
        generated_text = tokenizer.decode(idx[0].tolist())
        if "###" in generated_text.split("### Respuesta:\n")[-1]:
            break

    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

# ----------------------------------------------------------------------
# Interfaz de Chat Interactiva
# ----------------------------------------------------------------------

def chat_loop(model, tokenizer, device):
    print("\n" + "="*60)
    print("🤖 CHAT SFT INSTRUCT INICIADO (CON MEMORIA)")
    print("Escribe tu instrucción y presiona Enter.")
    print("Escribe 'salir', 'exit' o 'quit' para terminar.")
    print("="*60 + "\n")
    
    historial = []
    
    while True:
        try:
            user_input = input("\nUsuario> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo del chat...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("Saliendo del chat. ¡Hasta luego!")
            break
            
        # Construir el prompt completo con el historial
        prompt = ""
        
        if USE_HISTORY:
            # Añadir al historial
            historial.append({"role": "user", "content": user_input})
             # Mantener solo los últimos 10 intercambios (20 mensajes)
            if len(historial) > 4:
                historial = historial[-4:]
                
            for msg in historial:
                if msg["role"] == "user":
                    prompt += f"### Instrucción:\n{msg['content']}\n### Respuesta:\n"
                else:
                    prompt += f"{msg['content']}\n###\n"
                    
        prompt += f"### Instrucción:\n{user_input}\n### Respuesta:\n"
                
        print("Asistente> Pensando...", end="\r", flush=True)
        #print("prompt", prompt)
        
        # Generar respuesta
        respuesta_completa = generate_sft(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=GEN_CONFIG["max_new_tokens"],
            temperature=GEN_CONFIG["temperature"],
            top_k=GEN_CONFIG["top_k"],
            repetition_penalty=GEN_CONFIG["repetition_penalty"],
            device=device
        )
        
        # Extraer solo la nueva respuesta
        try:
            # Separamos por "### Respuesta:\n" y cogemos el último bloque
            solo_respuesta = respuesta_completa.split("### Respuesta:\n")[-1]
            solo_respuesta = solo_respuesta.replace("###", "").strip()
            
            # Guardar en historial
            historial.append({"role": "assistant", "content": solo_respuesta})
            
            # Sobrescribir "Pensando..."
            print(" " * 30, end="\r") 
            print(f"Asistente> {solo_respuesta}")
        except Exception:
            print(" " * 30, end="\r")
            print(f"Asistente (Crudo)> {respuesta_completa}")

# ----------------------------------------------------------------------
# Main: Carga e Inicialización
# ----------------------------------------------------------------------

if __name__ == "__main__":
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Iniciando... Dispositivo: {device}")

    # Cargar tokenizador
    tokenizer = TiktokenWrapper()
    try:
        tokenizer.load(VOCAB_PATH)
    except Exception as e:
        print(f"ERROR al cargar tokenizador en {VOCAB_PATH}: {e}")
        exit(1)

    # Verificar si existe el checkpoint
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: No se encontró el checkpoint SFT en {CHECKPOINT}.")
        exit(1)

    print(f"Cargando modelo instruct desde {CHECKPOINT}...")
    model = GPT(
        vocab_size = tokenizer.vocab_size,
        n_embd     = CONFIG["n_embd"],
        n_head     = CONFIG["n_head"],
        n_layer    = CONFIG["n_layer"],
        block_size = CONFIG["block_size"],
        dropout    = CONFIG["dropout"],
    ).to(device)
    
    ckpt = torch.load(CHECKPOINT, map_location=device)
    
    # Manejar posibles formatos de guardado (state_dict directo o diccionario anidado)
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Limpiar prefijos de torch.compile ("_orig_mod.") si los hay
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        unwrapped_state_dict[clean_key] = v

    model.load_state_dict(unwrapped_state_dict)
    print("Modelo cargado exitosamente.")
    
    # Iniciar el bucle de chat
    chat_loop(model, tokenizer, device)
