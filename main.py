"""ASTRA_Glia - Hauptsimulation mit 3-Faktor-Lernregel, RAG und Tool-Loop.
Kalibriert: bias 4.3, Ziel 7 Hz, scale 5.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*sm_120.*")
warnings.filterwarnings("ignore", message=".*not compatible.*")

import math
import time
from pathlib import Path

import torch
import yaml

from src.neurons.izhikevich_adaptive import AstroModulatedIzhikevich
from src.glia.astrocyte_modulator import AstrocyteModulator
from src.stemcell.neurogenesis import StemCellNeurogenesis
from src.tools.router import ToolRouter
from core.cortex_gateway import CortexGateway, embed_text

# Embedder vorwärmen (verhindert SNN-Stottern beim ersten Wake)
embed_text(None, "Warmup")
print("[INIT] Embedder vorgewaermt")

# --- Hyperparameter ---
HIDDEN = 128
TIME_STEPS = 50
INPUT_SCALE = 5.0
SLEEP_S = 0.005
DURATION_S = 300
BURST_EVERY_S = 12.0
BURST_LEN_S = 2.0
BURST_DC = 8.0
DOMAIN_SIZE = 8
CORTEX_MODEL = "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"

# --- 3-Faktor-Plastizität ---
TAU_E = 20.0
ETA = 0.001

# --- Demo: Tool -> Wissen -> RAG ---
DEMO_TOOL = True
DEMO_TOOL_AT = 3      # nach dem 3. Reiz: Frage, die ein Tool braucht
DEMO_RAG_AT = 6       # nach dem 6. Reiz: Frage, die das gespeicherte Wissen nutzt


def load_config():
    p = Path("config/hardware_profiles.yaml")
    if not p.exists():
        print("[GPU] Kein Profil gefunden, fahre mit CPU fort")
        return None
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prof = cfg.get("profiles", {})
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"[GPU] Erkannt: {gpu}")
        if "5070" in gpu:
            return prof.get("school_5070_mobile")
        if "2080" in gpu:
            return prof.get("home_2080ti")
    return prof.get("school_5070_mobile")


def main():
    print("[INIT] ASTRA_Glia")
    torch.set_grad_enabled(False)
    load_config()
    dev = torch.device("cpu")

    neuron = AstroModulatedIzhikevich(input_size=64, hidden_size=HIDDEN).to(dev)
    astro = AstrocyteModulator(hidden_size=HIDDEN, target_hz=7.0).to(dev)
    stem = StemCellNeurogenesis(hidden_size=HIDDEN, target_bias=4.3).to(dev)

    assert hasattr(neuron, "I_bias"), "Neuron-Patch fehlt (I_bias)"
    assert hasattr(neuron, "I_ext"), "Neuron-Patch fehlt (I_ext)"
    assert hasattr(astro, "salience_threshold"), "Glia-Totzone fehlt"
    assert hasattr(stem, "silent_ms"), "Stammzell-Patch fehlt"
    print(f"[INIT] Patches OK | Totzone {astro.salience_threshold} | "
          f"Konsolid-Schwelle {astro.consolidation_threshold}")

    router = ToolRouter()
    print(f"[INIT] Tools: {', '.join(router.tools.keys())}")

    gateway = CortexGateway(
        model_path=Path("checkpoints") / "llm" / CORTEX_MODEL,
        on_action=lambda ev, ans: print(f"\n  [Kortex #{ev['seq']}] {ans.strip()[:400]}"),
        idle_timeout=30.0,
        tool_executor=router.execute,
    )

    e_trace = torch.zeros(HIDDEN, 64, device=dev)

    def burst_dom(d, an):
        lo = d * DOMAIN_SIZE
        hi = min(lo + DOMAIN_SIZE, HIDDEN)
        neuron.I_ext[lo:hi] = BURST_DC if an else 0.0

    brain_s, step = 0.0, 0
    next_burst, burst_until, active_dom = BURST_EVERY_S, -1.0, None
    burst_nr = 0
    consol_total, replaced_total = 0, 0
    last_ping, last_diag = 0.0, time.time()
    demo_tool_sent, demo_rag_sent = False, False
    last_mod = last_fast = None
    counts, nsteps = torch.zeros(HIDDEN, device=dev), 0
    hz_last = 0.0

    t0 = time.time()
    print("Wandzeit | Hirnzeit |    Hz | still% | ca_slow | consol | ersetzt")

    try:
        while (time.time() - t0) < DURATION_S:
            step += 1

            # 1. Salienz-Reize steuern
            if active_dom is None and brain_s >= next_burst:
                active_dom = burst_nr % astro.n_domains
                burst_nr += 1
                burst_dom(active_dom, True)
                burst_until = brain_s + BURST_LEN_S
                next_burst = brain_s + BURST_EVERY_S
                print(f"\n[Reiz] Domaene {active_dom} +{BURST_DC} DC, {BURST_LEN_S}s Hirnzeit")
            elif active_dom is not None and brain_s >= burst_until:
                burst_dom(active_dom, False)
                active_dom = None

            # 2. Forward Pass
            x = torch.rand(1, TIME_STEPS, 64, device=dev) * INPUT_SCALE
            spikes = neuron(x, astro_mod=last_mod, astro_fast=last_fast)

            # 3. Astrozyten-Feedback
            astro_mod, consol = astro(spikes, dt=neuron.dt)
            last_mod = astro_mod.detach()
            last_fast = astro.fast_signal.detach()
            consol_total += len(consol)

            # 4. 3-Faktor-Lernregel
            if consol:
                pre = x[0].mean(0)
                post = spikes[0].sum(0) / spikes.shape[1]
                ca_plast = astro.ca_slow.item()
                decay = math.exp(-neuron.dt / TAU_E)
                e_trace = decay * e_trace + torch.outer(post, pre)
                dw = ETA * e_trace * ca_plast
                neuron.weight.weight.data += dw

            # 5. Kortex bei Konsolidierung (Telemetrie, keine Generierung)
            if consol and (time.time() - last_ping > 10.0):
                last_ping = time.time()
                gateway.submit(
                    f"Konsolidierung nach Salienz-Reiz {burst_nr}, "
                    f"Hirnzeit {brain_s:.1f}s, ca_slow {astro.ca_slow.item():.2f}",
                    salience=float(astro.fast_signal.max().item()),
                    channel="konsolidierung",
                )

            # 5b. Demo 1: externe Frage -> Tool -> Wissens-Episode
            if DEMO_TOOL and not demo_tool_sent and burst_nr >= DEMO_TOOL_AT:
                demo_tool_sent = True
                print("\n[Demo] Frage an den Kortex: Wetter morgen in Gelsenkirchen")
                gateway.submit("Wie wird das Wetter morgen in Gelsenkirchen?",
                               salience=0.8, channel="user")

            # 5c. Demo 2: stabiles Wissen -> sollte RAG statt Tool nutzen
            if DEMO_TOOL and not demo_rag_sent and burst_nr >= DEMO_RAG_AT:
                demo_rag_sent = True
                print("\n[Demo] Recall-Frage: Rolle der Astrozyten")
                gateway.submit("Was weisst du ueber die Rolle von Astrozyten?",
                               salience=0.5, channel="user")
                
                # FIX: Diese Zeilen standen vorher außerhalb des if-Blocks
                gateway.submit("Unterschied Axon vs. Dendrit",
                               salience=0.9, channel="demo")
                print("\n[Demo] Vergleichsfrage: Unterschied Axon vs. Dendrit")

            # 6. Stammzellen-Check
            v_new, u_new, _, _, _, n_rep = stem.check_and_repair(
                spikes.any(dim=1).float(),
                neuron.v_history[:, -1, :], neuron.u_history[:, -1, :],
                neuron.weight.weight, neuron.c_base, neuron.d_base,
                i_bias_param=neuron.I_bias,
                a_base_param=neuron.a_base, b_base_param=neuron.b_base,
                overheat_neurons=astro.overheat_neurons,
                dt=neuron.dt, steps_per_call=TIME_STEPS,
            )
            neuron.v.copy_(v_new.detach())
            neuron.u.copy_(u_new.detach())
            replaced_total += n_rep

            # 7. Dashboard & Metriken
            counts += spikes[0].sum(0)
            nsteps += TIME_STEPS
            brain_s += TIME_STEPS * neuron.dt / 1000.0

            if step % 100 == 0:
                hz = counts / nsteps / neuron.dt * 1000.0
                hz_last = hz.mean().item()
                sil = (hz == 0).float().mean().item() * 100
                el = time.time() - t0
                print(f"{int(el//60)}:{int(el%60):02d}     | {brain_s:7.1f}s | "
                      f"{hz_last:5.1f} | {sil:5.0f}  | {astro.ca_slow.item():7.2f} | "
                      f"{consol_total:6d} | {replaced_total}")
                counts.zero_(); nsteps = 0

            if time.time() - last_diag >= 30.0:
                last_diag = time.time()
                print(f"  [Diag] Reize {burst_nr} | Stufe1 {stem.stats['repaired']} | "
                      f"Overheat {astro.stats['overheat_neurons']} | "
                      f"Kortex {gateway.stats['processed']} | "
                      f"Tools {gateway.stats['tool_calls']} | "
                      f"RAG {gateway.stats['rag_used']}")

            time.sleep(SLEEP_S)

    except KeyboardInterrupt:
        print("\n[Stop] Vom Benutzer beendet.")
    finally:
        if active_dom is not None:
            burst_dom(active_dom, False)
        gateway.shutdown()

    el = time.time() - t0
    turnover = replaced_total / HIDDEN
    print("\n" + "=" * 70)
    print("SIMULATION ABGESCHLOSSEN")
    print("=" * 70)
    print(f"Wandzeit:            {el/60:.2f} min")
    print(f"Hirnzeit simuliert:  {brain_s:.1f} s  (Faktor {brain_s/el:.2f})")
    print(f"Salienz-Reize:       {burst_nr}")
    print(f"Konsolidierungen:    {consol_total}")
    print(f"Stufe 1 Reparatur:   {stem.stats['repaired']}")
    print(f"Stufe 2 Ersatz:      {replaced_total}  (Turnover {turnover:.2f})")
    print(f"Overheat-Neuronen:   {astro.stats['overheat_neurons']}")
    print(f"Letzte Feuerrate:    {hz_last:.1f} Hz")
    print(f"Gateway-Stats:       {gateway.stats}")

    if consol_total == 0:
        status = "KEINE Konsolidierung - Reize zu schwach oder Schwelle zu hoch"
    elif turnover > 0.2:
        status = "Hoher Turnover - Kalibrierung pruefen"
    elif 3.0 <= hz_last <= 15.0:
        status = "Homoeostase aktiv, Konsolidierung salienz-gekoppelt, LERNREGEL AKTIV"
    else:
        status = "Feuerrate ausserhalb Zielband"
    print(f"Status:              {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()