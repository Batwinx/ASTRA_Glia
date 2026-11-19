"""ASTRA_Glia - Adaptives Izhikevich-Neuron mit Astrozyten-Modulation."""
from __future__ import annotations

import torch
import torch.nn as nn


class AstroModulatedIzhikevich(nn.Module):
    """
    Persistenter Membran-State: v/u ueberleben forward()-Aufrufe.

    Neu gegenueber V1:
      I_bias      lernbarer DC-Grundtrieb (Rheobase liegt bei I=4)
      clamp       Astro-Signale auf [0,1] begrenzt -> keine Divergenz
      astro_fast  exzitatorischer Gain-Kanal (Gliotransmitter)
      History     detached -> kein Graph-Leak
      I_ext       externer Reiz-Buffer, getrennt von I_bias (Reparatur kann ihn nicht ueberschreiben)
      reset_state(), firing_hz()

    Signatur bleibt kompatibel: forward(x, astro_mod=None) wie bisher.
    """

    def __init__(self, input_size, hidden_size, dt=0.2,
                 bias_init=4.3, bias_jitter=0.5,
                 gain_fast=6.0, damp_slow=15.0, learnable_bias=True,
                 center_weights=True, bias_floor=4.1):
        super().__init__()
        self.dt = float(dt)
        self.hidden_size = hidden_size
        self.gain_fast = float(gain_fast)
        self.damp_slow = float(damp_slow)

        self.weight = nn.Linear(input_size, hidden_size, bias=False)
        if center_weights:   # Zeilensumme 0 -> kein fester DC-Offset durch den Eingangsmittelwert
            with torch.no_grad():
                self.weight.weight.sub_(self.weight.weight.mean(dim=1, keepdim=True))

        self.a_base = nn.Parameter(torch.full((hidden_size,), 0.02))
        self.b_base = nn.Parameter(torch.full((hidden_size,), 0.2))
        self.c_base = nn.Parameter(torch.full((hidden_size,), -65.0))
        self.d_base = nn.Parameter(torch.full((hidden_size,), 8.0))

        drive = torch.full((hidden_size,), float(bias_init))
        if bias_jitter > 0:
            drive = drive + torch.randn(hidden_size) * float(bias_jitter)
        if bias_floor is not None:   # Rheobase ~4: kein Neuron startet unterschwellig
            drive = drive.clamp(min=float(bias_floor))
        if learnable_bias:
            self.I_bias = nn.Parameter(drive)
        else:
            self.register_buffer("I_bias", drive)

        self.register_buffer("I_ext", torch.zeros(hidden_size))
        self.register_buffer("v", torch.full((1, hidden_size), -65.0))
        self.register_buffer("u", torch.full((1, hidden_size), -13.0))
        self.v_history = None
        self.u_history = None

    def reset_state(self, batch_size=1, device=None):
        device = device if device is not None else self.v.device
        self.v = torch.full((batch_size, self.hidden_size), -65.0, device=device)
        self.u = self.b_base.detach() * self.v

    def forward(self, x, astro_mod=None, astro_fast=None):
        batch_size, time_steps, _ = x.shape
        if self.v.shape[0] != batch_size or self.v.device != x.device:
            self.reset_state(batch_size, x.device)

        v, u = self.v, self.u
        spikes_out, v_hist, u_hist = [], [], []

        for t in range(time_steps):
            I_syn = self.weight(x[:, t, :]) + self.I_bias + self.I_ext

            if astro_fast is not None:
                ca_fast = astro_fast[:, t, :].clamp(0.0, 1.0)
                I_syn = I_syn + self.gain_fast * ca_fast

            if astro_mod is not None:
                ca_slow = astro_mod[:, t, :].clamp(0.0, 1.0)
                d_eff = self.d_base + self.damp_slow * ca_slow
                a_eff = self.a_base * (1.0 - 0.5 * ca_slow)
            else:
                d_eff = self.d_base
                a_eff = self.a_base

            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I_syn
            du = a_eff * (self.b_base * v - u)
            v_next = v + self.dt * dv
            u_next = u + self.dt * du

            fired = v_next >= 30.0
            spike = fired.to(x.dtype)
            v = torch.where(fired, self.c_base, v_next)
            u = torch.where(fired, u_next + d_eff, u_next)

            spikes_out.append(spike)
            v_hist.append(v.detach())
            u_hist.append(u.detach())

        self.v = v.detach()
        self.u = u.detach()
        self.v_history = torch.stack(v_hist, dim=1)
        self.u_history = torch.stack(u_hist, dim=1)
        return torch.stack(spikes_out, dim=1)

    def firing_hz(self, spikes):
        return spikes.detach().mean().item() / self.dt * 1000.0


