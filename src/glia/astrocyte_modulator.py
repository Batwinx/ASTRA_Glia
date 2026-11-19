"""ASTRA_Glia - Astrozyten: drei Calcium-Kanaele, getrennte Zeitskalen."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AstrocyteModulator(nn.Module):
    """
    ca_fast    tau=150 ms   pro Domaene   -> exzitatorischer Gain  (fast_signal)
    ca_medium  tau=5 s      pro Neuron    -> Daempfung/Homoeostase (astro_mod)
    ca_slow    tau=10 s     global        -> Konsolidierungs-Gate  (events)
    
    Alle Ca-Werte normiert: 1.0 == target_hz.
    Rueckgabe: (astro_mod, consolidation_events) - kompatibel zu main.py.
    Zusaetzlich: self.fast_signal, self.overheat_neurons, self.stats
    """

    def __init__(self, hidden_size,
                 tau_fast=150.0, tau_medium=5000.0, tau_slow=10000.0,
                 target_hz=7.0, domain_size=8, ki_homeo=0.6, gain_fast=1.0,
                 salience_threshold=1.8,
                 consolidation_threshold=0.15, overheat_threshold=2.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_fast = float(tau_fast)
        self.tau_medium = float(tau_medium)
        self.tau_slow = float(tau_slow)
        self.target_hz = float(target_hz)
        self.domain_size = max(1, int(domain_size))
        self.ki_homeo = float(ki_homeo)
        self.gain_fast = float(gain_fast)
        self.consolidation_threshold = float(consolidation_threshold)
        self.overheat_threshold = float(overheat_threshold)
        self.salience_threshold = float(salience_threshold)

        self.n_domains = (hidden_size + self.domain_size - 1) // self.domain_size
        dom = torch.arange(hidden_size) // self.domain_size
        M = torch.zeros(hidden_size, self.n_domains)
        M[torch.arange(hidden_size), dom] = 1.0
        self.register_buffer("domain_avg", M / M.sum(0).clamp(min=1.0))
        self.register_buffer("domain_onehot", M)

        self.register_buffer("ca_fast", torch.zeros(self.n_domains))
        self.register_buffer("ca_medium", torch.zeros(hidden_size))
        self.register_buffer("ca_slow", torch.zeros(1))

        self.fast_signal = None
        self.overheat_neurons = None
        self.stats = {"consolidations": 0, "overheat_neurons": 0}

    def reset_state(self):
        self.ca_fast.zero_()
        self.ca_medium.zero_()
        self.ca_slow.zero_()

    def forward(self, spikes, dt=0.2):
        B, T, H = spikes.shape
        if self.ca_fast.device != spikes.device:
            self.to(spikes.device)
        dt = float(dt)

        k = max(1, min(9, T))
        if k % 2 == 0:
            k -= 1
        rate = F.avg_pool1d(spikes.permute(0, 2, 1), kernel_size=k, stride=1,
                            padding=k // 2, count_include_pad=False)
        rate = rate.permute(0, 2, 1)[:, :T, :]
        norm = rate / dt * 1000.0 / self.target_hz

        a_fast = 1.0 - math.exp(-dt / self.tau_fast)
        a_med = 1.0 - math.exp(-dt / self.tau_medium)
        a_slow = 1.0 - math.exp(-dt / self.tau_slow)

        astro_mod = torch.zeros_like(spikes)
        fast_out = torch.zeros_like(spikes)
        consol_flags, exhaust_flags = [], []

        for t in range(T):
            r = norm[:, t, :].mean(0)

            r_dom = r @ self.domain_avg
            self.ca_fast = (1.0 - a_fast) * self.ca_fast + a_fast * r_dom
            gain = ((self.ca_fast - self.salience_threshold) * self.gain_fast).clamp(0.0, 1.0)
            fast_out[:, t, :] = (gain @ self.domain_onehot.t()).unsqueeze(0)

            err = r - 1.0
            self.ca_medium = (self.ca_medium
                              + a_med * (self.ki_homeo * err - self.ca_medium)).clamp(0.0, 1.0)
            astro_mod[:, t, :] = self.ca_medium.unsqueeze(0)

            self.ca_slow = (1.0 - a_slow) * self.ca_slow + a_slow * gain.max()
            hit = self.ca_slow > self.consolidation_threshold
            self.ca_slow = torch.where(hit, self.ca_slow * 0.5, self.ca_slow)
            consol_flags.append(hit)
            exhaust_flags.append((self.ca_medium >= 0.99) & (r > self.overheat_threshold))

        consol = torch.nonzero(torch.stack(consol_flags).flatten()).flatten().tolist()
        self.overheat_neurons = torch.nonzero(torch.stack(exhaust_flags).any(0)).flatten()
        self.stats["consolidations"] += len(consol)
        self.stats["overheat_neurons"] = int(self.overheat_neurons.numel())
        self.fast_signal = fast_out
        return astro_mod, consol
