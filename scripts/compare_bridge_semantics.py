#!/usr/bin/env python3
"""Fixed common validation examples; native-system comparison, not tokenizer isolation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moreact.config import dump_json
from moreact.semantics.common import sha256, read_manifest, load_arrays
from moreact.semantics.language import text_metrics, load_language, serialize, decode_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['prepare', 'bridge', 'moreact', 'summarize'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bridge-root', default=str(ROOT.parent / 'ttr_remogen_bridge'))
    p.add_argument('--motion-cache', default=str(ROOT / 'data/interx_h2_f8'))
    p.add_argument('--cache')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=64)
    args = p.parse_args()
    torch.set_num_threads(4)
    out, bridge = Path(args.output), Path(args.bridge_root)
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / 'selection.json'
    if args.mode == 'prepare':
        if selection_path.exists():
            raise FileExistsError(selection_path)
        examples_path = bridge / 'artifacts/interaction_rvq_v5/data/examples.json'
        examples = {r['episode']: r for r in json.loads(examples_path.read_text()) if r['split'] == 'val'}
        source = json.loads((Path(args.motion_cache) / 'manifest.json').read_text())
        groups, excluded = {}, []
        for r in source['records']:
            if r['split'] != 'val':
                continue
            other = examples.get(r['episode'])
            if other is None or not other['id'].endswith('_a%s_r%s' % (r['actor_id'], r['reactor_id'])):
                excluded.append(r['episode'])
                continue
            with np.load(Path(args.motion_cache) / r['file'], allow_pickle=False) as z:
                refs = [str(c) for c in z['captions']]
            row = dict(episode=r['episode'], bridge_id=other['id'], references=refs,
                       frames=r['frames'], actor_id=r['actor_id'], reactor_id=r['reactor_id'])
            groups.setdefault(other['category'], []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda r: hashlib.sha256(('42'+r['episode']).encode()).hexdigest())
        selected = []
        while any(groups.values()) and len(selected) < args.limit:
            for key in sorted(groups):
                if groups[key] and len(selected) < args.limit:
                    selected.append(groups[key].pop(0))
        dump_json(selection_path, dict(split='val', seed=42, count=len(selected), records=selected,
                   excluded=excluded, bridge_examples_sha256=sha256(examples_path),
                   notes=['Native systems: training tasks and budgets differ.',
                          'Common MoReAct multi-reference captions; identical greedy decoding limit 512.',
                          'FPS metadata differs (MoReAct 30, bridge 10); no temporal-rate conclusion.']))
        print('Selected', len(selected), 'validation episodes', flush=True)
        return
    selection = json.loads(selection_path.read_text())
    if args.mode == 'bridge' and sha256(bridge / 'artifacts/interaction_rvq_v5/data/examples.json') != selection['bridge_examples_sha256']:
        raise ValueError('Bridge examples changed since selection was fixed')
    if args.mode == 'summarize':
        reports = {name: json.loads((out / (name+'.json')).read_text()) for name in ['bridge', 'moreact']}
        if any(r['selection_sha256'] != sha256(selection_path) for r in reports.values()):
            raise ValueError('Selection identity mismatch')
        if [r['episode'] for r in reports['bridge']['samples']] != [r['episode'] for r in reports['moreact']['samples']]:
            raise ValueError('Episode order mismatch')
        summary = {name: report['summary'] for name, report in reports.items()}
        delta = {k: summary['moreact']['normal'][k]-summary['bridge']['normal'][k]
                 for k in ['word_f1', 'rouge_l_f1', 'valid_fraction']}
        dump_json(out / 'comparison.json', dict(summary=summary, moreact_minus_bridge=delta,
                  selection_sha256=sha256(selection_path), notes=selection['notes']))
        print(json.dumps(summary, indent=2))
        return
    if (out / (args.mode+'.json')).exists():
        raise FileExistsError(out / (args.mode+'.json'))
    checkpoint = args.checkpoint
    if args.mode == 'bridge':
        sys.path[:0] = [str(bridge / 'src'), str(bridge), str(bridge / 'scripts')]
        from evaluate_remogen_generated_token_descriptions import load_model, decode, PROMPT_SPECS
        checkpoint = checkpoint or str(bridge / 'artifacts/interaction_rvq_v5_flan_large/train_4gpu/best_text.pt')
        model, tokenizer, mapping, suppressed, step = load_model(checkpoint, str(ROOT.parent / 'models/flan-t5-large'), args.device)
        examples = {r['episode']: r for r in json.loads((bridge / 'artifacts/interaction_rvq_v5/data/examples.json').read_text())}
    else:
        manifest = read_manifest(args.cache, 'tokens')
        tokenizer, model, saved = load_language(checkpoint, args.device)
        from moreact.semantics.common import require_identity
        require_identity(saved['contract']['identity'], manifest['identity'])
        require_identity(saved['contract']['rvq_id'], manifest['rvq_id'])
        examples = {r['episode']: r for r in manifest['records']}
    samples = []
    checkpoint_id = sha256(checkpoint)
    for row in selection['records']:
        r = examples[row['episode']]
        modes = ['normal'] if args.mode == 'bridge' else ['normal', 'base_only', 'shuffle', 'remove_actor', 'remove_reactor']
        predictions = {}
        for mode in modes:
            if args.mode == 'bridge':
                text, eos = decode(model, tokenizer, suppressed, PROMPT_SPECS['pair_m2t']['instruction'],
                                   r['actor_pair'], r['reactor_pair'], args.device, 512)
                pred = dict(text=text, valid=bool(text.strip()) and eos, eos=eos)
            else:
                data = load_arrays(args.cache, r)
                text = serialize(data['ids'], data['spans'], manifest['rvq_config'], manifest['identity']['future'], mode, 42)
                pred = decode_text(model, tokenizer, text, saved['contract'])
            predictions[mode] = dict(prediction=pred, metrics=text_metrics(pred['text'], row['references']))
        samples.append(dict(episode=row['episode'], references=row['references'], modes=predictions))
        dump_json(out / (args.mode+'_progress.json'), dict(completed=len(samples), total=selection['count'], samples=samples))
        print(args.mode, len(samples), row['episode'], flush=True)
    summary = {}
    for mode in samples[0]['modes']:
        vals = [r['modes'][mode] for r in samples]
        summary[mode] = {k: float(np.mean([v['metrics'][k] for v in vals])) for k in ['word_f1', 'rouge_l_f1']}
        summary[mode]['valid_fraction'] = float(np.mean([v['prediction']['valid'] for v in vals]))
    dump_json(out / (args.mode+'.json'), dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_id,
              selection_sha256=sha256(selection_path), summary=summary, samples=samples))


if __name__ == '__main__':
    main()
