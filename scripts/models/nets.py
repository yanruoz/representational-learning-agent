"""Small trainable networks on fixed input features, and the training losses.

   Towers: `linear` (one Linear layer) or `mlp` (Linear -> BatchNorm -> ReLU -> Dropout per hidden
   layer, then Linear). Every method embeds each modality with its own tower, so a model can embed
   one modality alone at prediction time.

   Losses:
     infonce         symmetric CLIP-style cross-entropy over in-batch pairs, learnable temperature
     vicreg          invariance (MSE between paired embeddings) + variance hinge + covariance penalty
     supcon          supervised contrastive over both modalities' embeddings; a sample's partner
                     and same-label samples are positives; unlabeled samples (label -1) have only
                     their partner as positive
     reconstruction  per-modality MSE from the fused latent (modality dropout)
     cross_entropy   classifier on the fused embedding (modality dropout)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def tower(in_dim, head, hidden_dims, out_dim, dropout):
    if head == "linear":
        return nn.Linear(in_dim, out_dim)
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def infonce(za, zb, logit_scale):
    za, zb = F.normalize(za, dim=1), F.normalize(zb, dim=1)
    logits = logit_scale.exp().clamp(max=100) * za @ zb.T
    target = torch.arange(len(za), device=za.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def _vicreg_terms(z):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, off.pow(2).sum() / d


def vicreg(za, zb, w_inv=25.0, w_var=25.0, w_cov=1.0):
    va, ca = _vicreg_terms(za)
    vb, cb = _vicreg_terms(zb)
    return w_inv * F.mse_loss(za, zb) + w_var * (va + vb) + w_cov * (ca + cb)


def supcon(za, zb, labels, temperature):
    z = F.normalize(torch.cat([za, zb]), dim=1)
    n = len(za)
    idx = torch.arange(n, device=z.device)
    pair = torch.cat([idx, idx])
    lab = torch.cat([labels, labels])
    same_pair = pair[:, None] == pair[None, :]
    same_label = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    pos = (same_pair | same_label) & ~self_mask
    sim = (z @ z.T / temperature).masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


class MultiTower(nn.Module):
    """One tower per modality into a shared embed_dim space, plus optional decoders (autoencoder)
    or classifier (supervised fusion)."""

    def __init__(self, in_dims, head, hidden_dims, embed_dim, dropout, decoders=False, n_classes=0, temperature_init=0.07):
        super().__init__()
        self.names = list(in_dims)
        self.towers = nn.ModuleDict({m: tower(d, head, hidden_dims, embed_dim, dropout) for m, d in in_dims.items()})
        self.decoders = nn.ModuleDict({m: tower(embed_dim, head, list(reversed(hidden_dims)), d, dropout)
                                       for m, d in in_dims.items()}) if decoders else None
        self.classifier = nn.Linear(embed_dim, n_classes) if n_classes else None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

    def embed(self, xs):
        return {m: self.towers[m](x) for m, x in xs.items()}

    def fused(self, zs, modality_dropout):
        """Mean of the modalities' embeddings; in training, each sample drops each modality with
        probability modality_dropout (always keeping at least one)."""
        Z = torch.stack([zs[m] for m in self.names])               # (M, B, D)
        keep = torch.ones(Z.shape[:2], device=Z.device)
        if self.training and modality_dropout > 0 and len(self.names) > 1:
            keep = (torch.rand_like(keep) >= modality_dropout).float()
            none = keep.sum(0) == 0
            if none.any():
                pick = torch.randint(len(self.names), (int(none.sum()),), device=Z.device)
                keep[:, none] = 0
                keep[pick, none.nonzero().ravel()] = 1
        return (Z * keep[..., None]).sum(0) / keep.sum(0)[:, None]
