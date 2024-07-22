import torch
from torch import nn
from transformers import WhisperForAudioClassification


class PITLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            reduction="none", weight=torch.tensor([1.6, 1.0, 1.0]).to("cuda")
        )

    def forward(self, preds, labels):
        preds = preds.transpose(1, 2)

        losses = []
        for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
            loss = self.ce_loss(preds, labels[..., perm]).mean(dim=1)
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        loss = torch.min(losses, dim=1).values.mean()
        return loss


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=5, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, _, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=inputs.device)

        inputs = self.norm_input(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            k = self.to_k(inputs)
            v = self.to_v(inputs)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class WhisperWithSlotAttention(WhisperForAudioClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_slots = 3
        self.slot_attn = SlotAttention(self.num_slots, config.classifier_proj_size)

    def forward(self, input_features, labels=None):
        hidden_states = self.encoder(input_features)[0]
        hidden_states = self.projector(hidden_states)

        slots = self.slot_attn(hidden_states)
        logits = self.classifier(slots)

        loss = None
        if labels is not None:
            loss_fct = PITLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
