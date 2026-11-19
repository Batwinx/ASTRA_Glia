"""ASTRA_Glia - 3-Faktor-STDP mit Eligibility Traces."""
import torch
import torch.nn as nn


class STDP3Factor(nn.Module):
    """
    Eligibility Traces + 3-Faktor-Regel.
    
    Faktor 1: Pre-Synapse (Input x)
    Faktor 2: Post-Synapse (Spike oder Rate)
    Faktor 3: Ca_plast (Salienz-Signal, z.B. astro.ca_slow)
    
    Δw = η * e(t) * ca_plast
    e(t) = e(t-1) * exp(-dt/tau_e) + pre ⊗ post
    """
    
    def __init__(self, input_size, hidden_size, tau_e=20.0, eta=0.001):
        super().__init__()
        self.tau_e = float(tau_e)
        self.eta = float(eta)
        self.register_buffer("e_trace", torch.zeros(input_size, hidden_size))
    
    def forward(self, pre, post, ca_plast, dt=0.2):
        """
        pre:      (input_size,) oder (batch, input_size)
        post:     (hidden_size,) oder (batch, hidden_size)
        ca_plast: scalar oder (batch,)
        dt:       Zeitschritt in ms
        """
        if pre.dim() == 2:
            pre = pre.mean(0)
        if post.dim() == 2:
            post = post.mean(0)
        if isinstance(ca_plast, torch.Tensor):
            ca_plast = ca_plast.item()
        
        # Eligibility Trace
        decay = torch.exp(torch.tensor(-dt / self.tau_e, device=self.e_trace.device))
        self.e_trace = decay * self.e_trace + torch.outer(pre, post)
        
        # Weight Update
        dw = self.eta * self.e_trace * ca_plast
        return dw
    
    def reset(self):
        self.e_trace.zero_()