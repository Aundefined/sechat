import json
import tiktoken
import numpy as np
import os

class TiktokenWrapper:

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.tiktoken_to_compact: dict[int, int] = {}
        self.compact_to_tiktoken: dict[int, int] = {}
        self.vocab_size: int = 0

    def build_vocab_from_files(self, corpus_paths: list[str], token_out_paths: list[str],
                               chunk_chars: int = 100_000_000):
        """
        Tokeniza múltiples corpus en chunks de ~100MB (bajo consumo de RAM).
        Pass 1: tokeniza por chunks → escribe IDs raw de tiktoken a .raw
        Pass 2: construye vocab compacto, remapea con lookup vectorizado → .bin final
        """
        all_unique = set()
        raw_paths = []
        token_counts = []

        # Pass 1: tokenizar por chunks y escribir IDs raw
        for path in corpus_paths:
            name = os.path.basename(path)
            raw_path = os.path.join('/tmp', os.path.basename(path) + '.tokens.raw')
            raw_paths.append(raw_path)

            n_tokens = 0
            print(f'[Tokenizer] Pass 1 — tokenizando {name}...')
            with open(path, 'r', encoding='utf-8') as fin, \
                 open(raw_path, 'wb') as fout:
                while True:
                    chunk = fin.read(chunk_chars)
                    if not chunk:
                        break
                    tids = self.enc.encode(chunk)
                    all_unique.update(tids)
                    np.array(tids, dtype=np.int32).tofile(fout)
                    n_tokens += len(tids)
                    print(f'  {n_tokens/1e6:.0f}M tokens...', end='\r')

            token_counts.append(n_tokens)
            print(f'  {name}: {n_tokens:,} tokens          ')

        # Construir vocab compacto
        unique_ids = sorted(all_unique)
        self.tiktoken_to_compact = {orig: compact for compact, orig in enumerate(unique_ids)}
        self.compact_to_tiktoken = {compact: orig for orig, compact in self.tiktoken_to_compact.items()}
        self.vocab_size = len(unique_ids)
        print(f'[Tokenizer] Vocab compacto: {self.vocab_size:,} tokens '
              f'(de {self.enc.n_vocab:,})')

        # Lookup table vectorizado para remapeo rápido
        max_tid = max(self.tiktoken_to_compact.keys()) + 1
        lookup = np.zeros(max_tid + 1, dtype=np.int32)
        for orig, compact in self.tiktoken_to_compact.items():
            lookup[orig] = compact

        # Pass 2: remapear raw → compact en chunks
        total_tokens = 0
        remap_chunk = 10_000_000  # 10M tokens a la vez (~40MB RAM)

        for raw_path, out_path, n_tokens in zip(raw_paths, token_out_paths, token_counts):
            print(f'[Tokenizer] Pass 2 — remapeando → {os.path.basename(out_path)}...')
            raw = np.memmap(raw_path, dtype=np.int32, mode='r', shape=(n_tokens,))
            out = np.memmap(out_path, dtype=np.int32, mode='w+', shape=(n_tokens,))

            for i in range(0, n_tokens, remap_chunk):
                end = min(i + remap_chunk, n_tokens)
                out[i:end] = lookup[np.array(raw[i:end])]

            out.flush()
            del raw, out
            os.remove(raw_path)  # borrar archivo temporal
            total_tokens += n_tokens
            print(f'  {os.path.basename(out_path)}: {n_tokens:,} tokens')

        del lookup
        print(f'[Tokenizer] Total: {total_tokens:,} tokens')
        return total_tokens

    def encode(self, text: str) -> list[int]:
        # allowed_special="all" permite codificar tokens especiales si se necesitan (útil en SFT)
        tids = self.enc.encode(text, allowed_special="all")
        return [self.tiktoken_to_compact[t] for t in tids if t in self.tiktoken_to_compact]

    def decode(self, compact_ids: list[int]) -> str:
        tids = [self.compact_to_tiktoken[c] for c in compact_ids if c in self.compact_to_tiktoken]
        return self.enc.decode(tids)

    def save(self, path: str):
        data = {
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
            "tiktoken_to_compact": {str(k): v for k, v in self.tiktoken_to_compact.items()},
            "compact_to_tiktoken": {str(k): v for k, v in self.compact_to_tiktoken.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"[Tokenizer] Vocab guardado en {path}")

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.encoding_name = data["encoding_name"]
        self.enc = tiktoken.get_encoding(self.encoding_name)
        self.vocab_size = data["vocab_size"]
        self.tiktoken_to_compact = {int(k): v for k, v in data["tiktoken_to_compact"].items()}
        self.compact_to_tiktoken = {int(k): v for k, v in data["compact_to_tiktoken"].items()}
        print(f"[Tokenizer] Vocab cargado ({self.vocab_size:,} tokens)")
