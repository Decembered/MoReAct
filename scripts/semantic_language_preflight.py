#!/usr/bin/env python3
"""Full-cache length audit and maximum-input/maximum-target optimizer preflight."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from moreact.config import dump_json
from moreact.semantics.common import read_manifest
from moreact.semantics.language import build_language, texts_for_cache, audit_lengths, batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    m = read_manifest(a.cache, 'tokens')
    rows = texts_for_cache(a.cache, m)
    tok, model = build_language(str(ROOT.parent / 'models/flan-t5-large'), m['rvq_config'], m['identity']['future'], a.device)
    audit_lengths(tok, rows, Path(a.output).with_name('preflight_lengths.json'))
    longest = max(rows, key=lambda r: len(tok.encode(r['text'])))
    caption = max((c for r in rows for c in r['captions']), key=lambda c: len(tok.encode(c)))
    # This is only a capacity check; the modified model is discarded, including validation captions.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = batch(tok, [longest['text']], [caption], a.device)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = model(**x, labels=y).loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite preflight loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    dump_json(a.output, dict(input_tokens=x.input_ids.shape[1], target_tokens=y.shape[1], loss=float(loss.detach()),
              max_memory_allocated=torch.cuda.max_memory_allocated(a.device), updates=2,
              note='Capacity-only model discarded; no preflight weights enter formal training.'))

if __name__ == '__main__':
    main()
