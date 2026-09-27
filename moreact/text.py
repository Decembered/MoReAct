from __future__ import annotations

import torch


class FrozenCLIP:
    def __init__(self, model="ViT-B/32", device="cpu"):
        import clip
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(model, device=device, jit=False)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        if not texts:
            return torch.empty(0, 512)
        result = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            tokens = self.clip.tokenize(batch, truncate=True).to(self.device)
            embedding = self.model.encode_text(tokens).float()
            embedding[[not s.strip() for s in batch]] = 0
            result.append(embedding.cpu())
        return torch.cat(result)
