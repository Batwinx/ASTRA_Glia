# src/llm/brain.py
from llama_cpp import Llama
import json
import torch
import gc
from pathlib import Path

class PrefrontalCortex:
    """
    Das Qwen3-1.7B Gehirn von ASTRA_Glia.
    Wird nur bei Astrozyten-Trigger aktiviert, um VRAM zu sparen.
    """
    def __init__(self, model_path: str, verbose: bool = False):
        print(f"🧠 Initialisiere Präfrontalen Kortex (Qwen3-1.7B Q8_0)...")
        
        # VRAM-Check
        if torch.cuda.is_available():
            free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_vram_gb = free_vram / (1024**3)
            print(f"   📊 Freier VRAM: {free_vram_gb:.2f} GB")
            if free_vram_gb < 2.0:
                print("   ⚠️ Warnung: Weniger als 2 GB VRAM frei. Bereinige...")
                torch.cuda.empty_cache()
                gc.collect()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,     # Alle Layer auf GPU
            n_threads=4,
            n_batch=512,
            verbose=verbose,
            # Qwen3 nutzt automatisch das richtige Template bei GGUF
        )
        print("✅ Präfrontaler Kortex online (Thinking Mode aktiv).")

    def wake_up_and_think(self, snn_context: str, astro_state: dict, user_query: str = "") -> str:
        """
        Wird von den Astrozyten aufgerufen bei Trigger (Kanal 1 oder 2).
        """
        system_prompt = f"""Du bist ASTRA_Glia, ein hybrider KI-Agent mit Spiking Neural Network.
Dein limbisches System hat folgende Situation erkannt:
- Astrozyten-Zustand: {json.dumps(astro_state)}
- SNN-Kontext: {snn_context}

Deine Aufgabe: Analysiere die Situation. Wenn ein Tool-Call nötig ist (wetter.api, search.api, wikipedia.api, mail.api), antworte NUR mit einem JSON-Objekt:
{{"action": "tool_name", "query": "suchbegriff"}}

Ansonsten antworte kurz, präzise und hilfsbereit auf Deutsch. /no_think"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query or "Analysiere den Zustand und entscheide."}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=512,
            stop=["<|im_end|>"]
        )
        
        return response['choices'][0]['message']['content'].strip()

    def unload(self):
        """Gibt VRAM sofort frei."""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()
        gc.collect()
        print("💤 Präfrontaler Kortex im Schlafmodus (VRAM freigegeben).")