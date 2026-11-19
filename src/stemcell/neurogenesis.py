"""ASTRA_Glia - Stammzellen: zweistufige Reparatur statt Schweigen-als-Tod."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


class StemCellNeurogenesis(nn.Module):
    """
    Stufe 1 (Reparatur): I_bias um bias_step anheben (bis bias_max), v/u reset, Karenz grace_ms;
                     bis zu max_strikes Versuche (ein Neuron mit negativem Eingangs-Offset
                     wird so gerettet statt durch zufaellig neue Gewichte ersetzt).
    Stufe 2 (Ersatz):    Gewichte + a/b/c/d + I_bias neu, nur bei Wiederholung.

    Zeitbasis in ms (nicht Aufrufe). Rate-Limit gegen Massen-Turnover.
    Signatur rueckwaertskompatibel: neue Argumente sind optional.
    """

    def __init__(self, hidden_size, inactivity_threshold=300,
                 repair_log_path="memory/short_term/repair_log.json",
                 inactivity_ms=2000.0, grace_ms=1000.0, target_bias=4.3,
                 bias_jitter=0.5, max_repair_frac=0.05, strike_decay_ms=30000.0,
                 max_strikes=3, bias_step=1.5, bias_max=9.0, bias_floor=4.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.inactivity_ms = float(inactivity_ms)
        self.grace_ms = float(grace_ms)
        self.target_bias = float(target_bias)
        self.bias_jitter = float(bias_jitter)
        self.strike_decay_ms = float(strike_decay_ms)
        self.max_strikes = int(max_strikes)
        self.bias_step = float(bias_step)
        self.bias_max = float(bias_max)
        self.bias_floor = float(bias_floor)
        self.max_repair = max(1, int(hidden_size * max_repair_frac))

        self.repair_log_path = Path(repair_log_path)
        self.repair_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.register_buffer("silent_ms", torch.zeros(hidden_size))
        self.register_buffer("strike", torch.zeros(hidden_size, dtype=torch.int8))
        self.register_buffer("since_repair_ms", torch.zeros(hidden_size))
        self.stats = {"repaired": 0, "replaced": 0, "overheat": 0}

    def check_and_repair(self, spikes, v, u, weight_param, c_base_param, d_base_param,
                         i_bias_param=None, a_base_param=None, b_base_param=None,
                         overheat_neurons=None, dt=0.2, steps_per_call=1):
        if spikes.dim() == 3:
            act = spikes.any(dim=1).float()          # Spike irgendwo im Fenster
        else:
            act = (spikes > 0).float()
        act = act.mean(dim=0) if act.dim() > 1 else act
        step_ms = float(dt) * int(steps_per_call)

        self.silent_ms = torch.where(act > 0, torch.zeros_like(self.silent_ms),
                                     self.silent_ms + step_ms)
        # Strike verfaellt, wenn das Neuron nach Reparatur lange genug stabil lief
        on_probation = self.strike > 0
        self.since_repair_ms = torch.where(on_probation, self.since_repair_ms + step_ms,
                                           torch.zeros_like(self.since_repair_ms))
        healed = on_probation & (act > 0) & (self.since_repair_ms >= self.strike_decay_ms)
        self.strike = torch.where(healed, torch.zeros_like(self.strike), self.strike)
        limit = torch.where(self.strike > 0,
                            torch.full_like(self.silent_ms, self.grace_ms),
                            torch.full_like(self.silent_ms, self.inactivity_ms))
        candidates = self.silent_ms >= limit

        forced = torch.zeros_like(candidates)
        if overheat_neurons is not None and overheat_neurons.numel() > 0:
            forced[overheat_neurons] = True
            self.stats["overheat"] += int(overheat_neurons.numel())

        stage1 = candidates & (self.strike < self.max_strikes) & ~forced
        stage2 = (candidates & (self.strike >= self.max_strikes)) | forced

        n2 = int(stage2.sum().item())
        if n2 > self.max_repair:                     # Rate-Limit: haerteste zuerst
            idx = torch.nonzero(stage2).flatten()
            worst = idx[torch.argsort(self.silent_ms[idx], descending=True)[:self.max_repair]]
            stage2 = torch.zeros_like(stage2)
            stage2[worst] = True

        b, h = v.shape
        touch = stage1 | stage2

        if bool(stage1.any()):
            if i_bias_param is not None:
                raised = (i_bias_param.data.clamp(min=self.target_bias)
                          + self.bias_step).clamp(max=self.bias_max)
                i_bias_param.data.copy_(torch.where(stage1, raised, i_bias_param.data))
            self.strike = torch.where(stage1, self.strike + 1, self.strike)
            self.since_repair_ms = torch.where(stage1, torch.zeros_like(self.since_repair_ms),
                                               self.since_repair_ms)
            self.stats["repaired"] += int(stage1.sum().item())

        if bool(stage2.any()):
            fan_in = weight_param.data.shape[1]
            bound = 1.0 / (fan_in ** 0.5)
            new_w = torch.empty_like(weight_param.data).uniform_(-bound, bound)
            new_w = new_w - new_w.mean(dim=1, keepdim=True)   # zentriert wie beim Init
            weight_param.data.copy_(torch.where(
                stage2.unsqueeze(1).expand_as(weight_param.data), new_w, weight_param.data))
            for p, val in ((c_base_param, -65.0), (d_base_param, 8.0),
                           (a_base_param, 0.02), (b_base_param, 0.2)):
                if p is not None:
                    p.data.copy_(torch.where(stage2, torch.full_like(p.data, val), p.data))
            if i_bias_param is not None:
                fresh = (self.target_bias + torch.randn_like(i_bias_param.data) * self.bias_jitter
                         ).clamp(min=self.bias_floor)
                i_bias_param.data.copy_(torch.where(stage2, fresh, i_bias_param.data))
            self.strike = torch.where(stage2, torch.zeros_like(self.strike), self.strike)
            self.stats["replaced"] += int(stage2.sum().item())
            self._log_repair_event(int(stage2.sum().item()),
                                   "excitability_unrecoverable" if bool(forced.any())
                                   else "silence_after_repair")

        if bool(touch.any()):
            m = touch.unsqueeze(0).expand(b, h)
            v = torch.where(m, torch.full_like(v, -65.0), v)
            u = torch.where(m, torch.full_like(u, -13.0), u)
            self.silent_ms = torch.where(touch, torch.zeros_like(self.silent_ms), self.silent_ms)

        return v, u, weight_param, c_base_param, d_base_param, int(stage2.sum().item())

    def _log_repair_event(self, count, reason):
        entry = {"timestamp": datetime.now().isoformat(),
                 "event": "stemcell_neurogenesis",
                 "replaced_neurons": int(count), "reason": reason}
        try:
            with self.repair_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Stammzell] Repair-Log nicht schreibbar: {e}")
