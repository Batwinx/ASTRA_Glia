"""ASTRA_Glia · Interaktiver Chat-Agent mit laufendem SNN + Tool-Loop."""
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*sm_120.*")
warnings.filterwarnings("ignore", message=".*not compatible.*")

import json
import math
import threading
import time
from pathlib import Path

import torch
import yaml

from src.neurons.izhikevich_adaptive import AstroModulatedIzhikevich
from src.glia.astrocyte_modulator import AstrocyteModulator
from src.stemcell.neurogenesis import StemCellNeurogenesis
from src.tools.router import ToolRouter
from core.cortex_gateway import CortexGateway, embed_text

# --- Hyperparameter ---
INPUT_SIZE = 64
HIDDEN = 128
TIME_STEPS = 50
INPUT_SCALE = 5.0
SLEEP_S = 0.005
TAU_E = 20.0
ETA = 0.001
TARGET_HZ = 7.0
TARGET_BIAS = 4.3

# --- Stimulus / Gateway ---
BURST_DC = 8.0
BURST_DURATION_S = 2.0  # Wanduhrzeit, nicht Hirnzeit
USER_SALIENCE = 0.8
IDLE_TIMEOUT_S = 60.0
CORTEX_MODEL = "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
CONFIG_PATH = Path("config/hardware_profiles.yaml")
PREVIEW_CHARS = 200

DEVICE = torch.device("cpu")


def load_config():
    """Hardware-Profil anhand der erkannten GPU wählen (None, wenn keine Config)."""
    if not CONFIG_PATH.exists():
        return None
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        profiles = (yaml.safe_load(f) or {}).get("profiles", {})

    default = profiles.get("school_5070_mobile")
    if not torch.cuda.is_available():
        return default

    gpu = torch.cuda.get_device_name(0)
    print(f"[GPU] Erkannt: {gpu}")
    if "5070" in gpu:
        return profiles.get("school_5070_mobile")
    if "2080" in gpu:
        return profiles.get("home_2080ti")
    return default


class ChatAgent:
    def __init__(self):
        self.running = True
        self.brain_s = 0.0
        self.burst_nr = 0
        self.consol_total = 0
        self.replaced_total = 0

        self._init_snn()
        self._init_gateway()

        self.snn_thread = threading.Thread(target=self._snn_loop, daemon=True)
        self.snn_thread.start()

        self._print_banner()

    # ------------------------------------------------------------------ Init

    def _init_snn(self):
        self.neuron = AstroModulatedIzhikevich(
            input_size=INPUT_SIZE, hidden_size=HIDDEN
        ).to(DEVICE)
        self.astro = AstrocyteModulator(
            hidden_size=HIDDEN, target_hz=TARGET_HZ
        ).to(DEVICE)
        self.stem = StemCellNeurogenesis(
            hidden_size=HIDDEN, target_bias=TARGET_BIAS
        ).to(DEVICE)

        # Lernregel-State
        self.e_trace = torch.zeros(HIDDEN, INPUT_SIZE, device=DEVICE)
        self.last_mod = None
        self.last_fast = None

        # Stimulus-State: Lock + Token pro Domäne, damit ein alter Reset
        # einen neueren Burst auf derselben Domäne nicht vorzeitig beendet.
        self._stim_lock = threading.Lock()
        self._stim_tokens = [0] * self.astro.n_domains

    def _init_gateway(self):
        self.router = ToolRouter()
        self.gateway = CortexGateway(
            model_path=Path("checkpoints") / "llm" / CORTEX_MODEL,
            on_action=self._on_answer,
            idle_timeout=IDLE_TIMEOUT_S,
            tool_executor=self._execute_tool,
        )

    @staticmethod
    def _print_banner():
        print("\n" + "=" * 60)
        print("🧠 ASTRA_Glia Chat-Agent bereit")
        print("   Ich suche automatisch in Wikipedia/Search, wenn nötig.")
        print("   Befehle: /quit, /stats, /tools, /memory")
        print("=" * 60 + "\n")

    # ------------------------------------------------------- Gateway-Callbacks

    def _execute_tool(self, tool_name, query):
        """Tool-Aufruf mit sichtbarem Feedback."""
        print(f"\n🔍 Suche nach: {query} ({tool_name})")
        result = self.router.execute(tool_name, query)
        if result:
            preview = result[:PREVIEW_CHARS].replace("\n", " ")
            print(f"   ✅ Gefunden: {preview}...")
        else:
            print(f"   ❌ Keine Ergebnisse für '{query}'")
        return result

    def _on_answer(self, ev, answer_json):
        """Callback vom Gateway: Antwort auf stdout."""
        try:
            answer = json.loads(answer_json).get("answer", answer_json)
        except (ValueError, AttributeError):
            answer = answer_json

        print(f"\n{'=' * 60}")
        print(f"🤖 ASTRA: {answer}")
        print(f"{'=' * 60}\n> ", end="", flush=True)

    # --------------------------------------------------------------- SNN-Loop

    def _snn_loop(self):
        """Hintergrund-Loop: SNN + Astrozyten + Stammzellen."""
        torch.set_grad_enabled(False)

        while self.running:
            x = torch.rand(1, TIME_STEPS, INPUT_SIZE, device=DEVICE) * INPUT_SCALE
            spikes = self.neuron(x, astro_mod=self.last_mod, astro_fast=self.last_fast)

            consol = self._astro_feedback(spikes)
            if consol:
                self._learn(x, spikes)
            self._stem_cell_check(spikes)

            self.brain_s += TIME_STEPS * self.neuron.dt / 1000.0
            time.sleep(SLEEP_S)

    def _astro_feedback(self, spikes):
        astro_mod, consol = self.astro(spikes, dt=self.neuron.dt)
        self.last_mod = astro_mod.detach()
        self.last_fast = self.astro.fast_signal.detach()
        self.consol_total += len(consol)
        return consol

    def _learn(self, x, spikes):
        """3-Faktor-Lernregel + Konsolidierungsmeldung ans Gateway."""
        pre = x[0].mean(0)
        post = spikes[0].sum(0) / spikes.shape[1]
        ca_plast = self.astro.ca_slow.item()

        decay = math.exp(-self.neuron.dt / TAU_E)
        self.e_trace = decay * self.e_trace + torch.outer(post, pre)
        self.neuron.weight.weight.data += ETA * self.e_trace * ca_plast

        self.gateway.submit(
            f"Konsolidierung nach Salienz-Reiz {self.burst_nr}, "
            f"Hirnzeit {self.brain_s:.1f}s, ca_slow {ca_plast:.2f}",
            salience=float(self.astro.fast_signal.max().item()),
            channel="konsolidierung",
        )

    def _stem_cell_check(self, spikes):
        n = self.neuron
        v_new, u_new, _, _, _, n_rep = self.stem.check_and_repair(
            spikes.any(dim=1).float(),
            n.v_history[:, -1, :], n.u_history[:, -1, :],
            n.weight.weight, n.c_base, n.d_base,
            i_bias_param=n.I_bias,
            a_base_param=n.a_base, b_base_param=n.b_base,
            overheat_neurons=self.astro.overheat_neurons,
            dt=n.dt, steps_per_call=TIME_STEPS,
        )
        n.v.copy_(v_new.detach())
        n.u.copy_(u_new.detach())
        self.replaced_total += n_rep

    # --------------------------------------------------------------- Stimulus

    def inject_salience(self):
        """Nutzer-Eingabe als Salienz-Burst in eine rotierende Domäne injizieren."""
        self.burst_nr += 1
        dom = self.burst_nr % self.astro.n_domains
        lo = dom * self.astro.domain_size
        hi = min(lo + self.astro.domain_size, HIDDEN)

        with self._stim_lock:
            self._stim_tokens[dom] += 1
            token = self._stim_tokens[dom]
            self.neuron.I_ext[lo:hi] = BURST_DC

        timer = threading.Timer(
            BURST_DURATION_S, self._clear_stimulus, args=(dom, lo, hi, token)
        )
        timer.daemon = True
        timer.start()

    def _clear_stimulus(self, dom, lo, hi, token):
        with self._stim_lock:
            if self._stim_tokens[dom] == token:  # kein neuerer Burst aktiv
                self.neuron.I_ext[lo:hi] = 0.0

    # ------------------------------------------------------------------- Chat

    def _cmd_stats(self):
        print("\n📊 Stats:")
        print(f"   Hirnzeit: {self.brain_s:.1f}s")
        print(f"   Konsolidierungen: {self.consol_total}")
        print(f"   Neuronen ersetzt: {self.replaced_total}")
        print(f"   Gateway: {self.gateway.stats}\n")

    def _cmd_tools(self):
        print("\n🔧 Verfügbare Tools:")
        for name in self.router.tools:
            print(f"   - {name}")
        print()

    def _cmd_memory(self):
        hippo = self.gateway.hippocampus
        print("\n💾 Wissensbestand:")
        print(f"   {hippo.long_term_count()} Einträge")
        print(f"   Arten: {hippo.kind_stats()['kinds']}\n")

    def _handle_command(self, text):
        """Führt einen /-Befehl aus. Gibt True zurück, wenn der Text einer war."""
        commands = {
            "/stats": self._cmd_stats,
            "/tools": self._cmd_tools,
            "/memory": self._cmd_memory,
        }
        handler = commands.get(text)
        if handler is None:
            return False
        handler()
        return True

    def chat_loop(self):
        try:
            while self.running:
                user_input = input("> ").strip()

                if not user_input:
                    continue
                if user_input == "/quit":
                    break
                if self._handle_command(user_input):
                    continue

                self.inject_salience()
                print("\n⏳ Denke nach...")
                self.gateway.submit(user_input, salience=USER_SALIENCE, channel="user")
                time.sleep(0.5)

        except (KeyboardInterrupt, EOFError):
            print("\n[Stop] Vom Benutzer beendet.")
        finally:
            self.running = False
            self.gateway.shutdown()
            print("\n👋 ASTRA_Glia heruntergefahren.")


def main():
    print("[INIT] ASTRA_Glia Chat-Agent")
    embed_text(None, "Warmup")  # Embedder vorwärmen
    print("[INIT] Embedder vorgewaermt")

    load_config()  # Ergebnis wird aktuell nicht genutzt (SNN läuft auf CPU)

    ChatAgent().chat_loop()


if __name__ == "__main__":
    main()
