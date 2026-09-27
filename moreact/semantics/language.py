"""Text-only supervision on complete paired motion tokens. No generation feedback."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from ..config import digest, dump_json
from ..train import load_checkpoint, save_checkpoint, seed_all, rng_state, restore_rng
from .common import (VERSION, load_arrays, read_manifest, require_identity, new_output,
                     sha256)
from .rvq import load_rvq

PROMPT = "Describe the overall interaction between the actor and the reactor in these complete motions.\n"


def alphabet(config, future):
    return (["<rvq%d_%d>" % (level, code) for level in range(config["levels"]) for code in range(config["size"])]
            + ["<step>", "<actor>", "<reactor>"] + ["<valid_%d>" % n for n in range(1, future+1)])


def serialize(ids, spans, config, future, mode="normal", seed=0):
    ids, spans = np.asarray(ids), np.asarray(spans)
    if ids.ndim != 3 or ids.shape[1:] != (2, config["levels"]) or len(ids) == 0:
        raise ValueError("Expected nonempty [windows, actor/reactor, RVQ levels] codes")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= config["size"]):
        raise ValueError("Invalid motion codes")
    if spans.shape != (len(ids), 3) or np.any(spans[:, 2] < 1) or np.any(spans[:, 2] > future):
        raise ValueError("Invalid window spans")
    if mode not in ("normal", "shuffle", "remove_actor", "remove_reactor", "base_only"):
        raise ValueError("Unknown understanding ablation")
    if mode == "shuffle":
        # Permute paired windows together; keep level/time groups intact.
        order = np.random.default_rng(seed).permutation(len(ids))
        ids = ids[order]
    result = [PROMPT]
    for group, span in zip(ids, spans):
        result.append("<step><valid_%d>" % span[2])
        for role, name in enumerate(("actor", "reactor")):
            if mode == "remove_" + name:
                continue
            result.append("<" + name + ">")
            for level, code in enumerate(group[role]):
                if mode == "base_only" and level > 0:
                    continue
                result.append("<rvq%d_%d>" % (level, code))
    return "".join(result)


def token_identity(tokenizer):
    # The fast tokenizer JSON includes segmentation/normalization, not just ID mapping.
    return digest(dict(vocab=tokenizer.get_vocab(), special=tokenizer.special_tokens_map,
                       backend=tokenizer.backend_tokenizer.to_str()))


def build_language(base_model, config, future, device):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True, use_fast=True)
    symbols = alphabet(config, future)
    if any(s in tokenizer.get_vocab() for s in symbols):
        raise ValueError("Expected original language vocabulary, not a previous motion tokenizer")
    tokenizer.add_tokens(symbols)
    model = T5ForConditionalGeneration.from_pretrained(str(base_model), local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    for symbol in symbols:
        if tokenizer.encode(symbol, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(symbol)]:
            raise ValueError("Motion symbol is not atomic")
    tokenizer.model_max_length = 4096
    return tokenizer, model


def texts_for_cache(cache, manifest):
    rows = []
    for record in manifest["records"]:
        data = load_arrays(cache, record)
        text = serialize(data["ids"], data["spans"], manifest["rvq_config"], manifest["identity"]["future"])
        rows.append(dict(record, text=text))
    return rows


def audit_lengths(tokenizer, rows, output, max_input=4096, max_target=512):
    results, errors = [], []
    for row in rows:
        ni = len(tokenizer.encode(row["text"], truncation=False))
        nt = [len(tokenizer.encode(c, truncation=False)) for c in row["captions"]]
        result = dict(episode=row["episode"], split=row["split"], input_tokens=ni, target_tokens=nt)
        results.append(result)
        if ni > max_input or not nt or max(nt) > max_target or not all(c.strip() for c in row["captions"]):
            errors.append(result)
    dump_json(output, dict(max_input=max_input, max_target=max_target, records=results, errors=errors))
    if errors:
        raise ValueError("Token budget exceeded or missing captions; see " + str(output))


def batch(tokenizer, texts, targets, device, max_input=4096, max_target=512):
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False).to(device)
    labels = tokenizer(targets, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False).to(device)
    if inputs.input_ids.shape[1] > max_input or labels.input_ids.shape[1] > max_target:
        raise ValueError("Token budget exceeded; truncation is forbidden")
    return inputs, labels.input_ids.masked_fill(labels.attention_mask == 0, -100)


@torch.no_grad()
def validate(model, tokenizer, rows, device, batch_size):
    model.eval()
    total, tokens = 0., 0
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start+batch_size]
        inputs, labels = batch(tokenizer, [r["text"] for r in selected], [r["captions"][0] for r in selected], device)
        result = model(**inputs, labels=labels)
        n = int((labels != -100).sum())
        total += float(result.loss) * n
        tokens += n
    if not tokens:
        raise ValueError("Empty validation set")
    return total / tokens


def train_captioner(cache, output, steps, base_model="/data/autovla/projects/models/flan-t5-large",
                    device="cpu", batch_size=1, learning_rate=1e-4, validate_every=100, seed=0, resume=None,
                    gradient_accumulation=1, gradient_checkpointing=False):
    if min(steps, batch_size, validate_every, gradient_accumulation) < 1 or learning_rate <= 0:
        raise ValueError("Training budgets/batch size/LR must be positive")
    manifest = read_manifest(cache, "tokens")
    rows = texts_for_cache(cache, manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val:
        raise ValueError("Need train and validation splits")
    out = Path(output)
    protocol = dict(cache_id=manifest["manifest_id"], batch_size=batch_size, learning_rate=learning_rate,
                    seed=seed, validate_every=validate_every)
    if gradient_accumulation != 1 or gradient_checkpointing:
        protocol.update(gradient_accumulation=gradient_accumulation,
                        gradient_checkpointing=gradient_checkpointing)
    seed_all(seed)
    chooser = random.Random(seed)
    start, best = 0, float("inf")
    if resume:
        tokenizer, model, saved = load_language(resume, device)
        if saved["protocol"] != protocol or out.resolve() != Path(resume).resolve().parent:
            raise ValueError("Caption resume config/cache/output mismatch")
        require_identity(saved["contract"]["identity"], manifest["identity"])
        require_identity(saved["contract"]["rvq_id"], manifest["rvq_id"], "RVQ")
        start, best = saved["step"], saved["best"]
    else:
        new_output(out)
        tokenizer, model = build_language(base_model, manifest["rvq_config"], manifest["identity"]["future"], device)
        tokenizer.save_pretrained(str(out / "tokenizer"))
        saved = None
    if steps <= start:
        raise ValueError("steps is an absolute target beyond the saved step")
    # Audit every split and every reference before any parameter update.
    audit_lengths(tokenizer, rows, out / "length_audit.json")
    model.requires_grad_(True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        chooser.setstate(saved["chooser"])
    contract = dict(version=VERSION, identity=manifest["identity"], rvq_id=manifest["rvq_id"],
                    rvq_config=manifest["rvq_config"], tokenizer_id=token_identity(tokenizer),
                    prompt=PROMPT, max_input=4096, max_target=512, base_model=str(base_model),
                    task="complete_pair_interaction_description")
    if saved:
        contract = saved["contract"]
    dump_json(out / "contract.json", contract)
    for step in range(start, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.
        for micro_step in range(gradient_accumulation):
            chosen = [chooser.choice(train) for _ in range(batch_size)]
            inputs, labels = batch(tokenizer, [r["text"] for r in chosen],
                                   [chooser.choice(r["captions"]) for r in chosen], device)
            loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite caption loss")
            (loss / gradient_accumulation).backward()
            mean_loss += float(loss.detach()) / gradient_accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        metrics = dict(step=step+1, loss=mean_loss, grad_norm=float(norm))
        if (step+1) % validate_every == 0 or step+1 == steps:
            nll = validate(model, tokenizer, val, device, batch_size)
            if not np.isfinite(nll):
                raise FloatingPointError("Nonfinite validation NLL")
            metrics["val_nll"] = nll
            improved = nll < best
            best = min(best, nll)
            state = dict(kind="semantic_captioner", contract=contract, protocol=protocol,
                         model=model.state_dict(), model_config=model.config.to_dict(),
                         optimizer=optimizer.state_dict(), rng=rng_state(), chooser=chooser.getstate(),
                         step=step+1, best=best)
            save_checkpoint(out / "last.pt", state)
            if improved:
                save_checkpoint(out / "best.pt", state)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    # No motion model is loaded by this trainer; it cannot update either frozen asset.
    return metrics


def load_language(checkpoint, device="cpu"):
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    checkpoint_id = sha256(checkpoint)
    saved = load_checkpoint(checkpoint)
    if checkpoint_id != sha256(checkpoint):
        raise ValueError("Caption checkpoint changed during loading")
    saved["checkpoint_id"] = checkpoint_id
    if saved.get("kind") != "semantic_captioner" or saved["contract"].get("version") != VERSION:
        raise ValueError("Invalid captioner checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(checkpoint).parent / "tokenizer"), local_files_only=True, use_fast=True)
    require_identity(saved["contract"]["tokenizer_id"], token_identity(tokenizer), "vocabulary")
    if saved["contract"]["prompt"] != PROMPT:
        raise ValueError("Caption prompt version mismatch")
    with torch.random.fork_rng(devices=[]):
        model = T5ForConditionalGeneration(T5Config.from_dict(saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return tokenizer, model, saved


@torch.no_grad()
def decode_text(model, tokenizer, text, contract):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=False, return_token_type_ids=False).to(device)
    if inputs.input_ids.shape[1] > contract["max_input"]:
        raise ValueError("Input token budget exceeded")
    forbidden = tokenizer.convert_tokens_to_ids(alphabet(contract["rvq_config"], contract["identity"]["future"]))
    ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=contract["max_target"],
                         suppress_tokens=forbidden, use_cache=True)
    complete = int(ids[0, -1]) == tokenizer.eos_token_id
    text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return dict(text=text, complete=complete, valid=bool(text) and complete)


class InteractionCaptioner:
    def __init__(self, motion_tokenizer, rvq_checkpoint, caption_checkpoint, device="cpu"):
        self.motion = motion_tokenizer
        self.rvq, rq = load_rvq(rvq_checkpoint, motion_tokenizer.device, motion_tokenizer.identity)
        self.tokenizer, self.model, saved = load_language(caption_checkpoint, device)
        self.contract = saved["contract"]
        self.captioner_id = saved["checkpoint_id"]
        require_identity(self.contract["identity"], self.motion.identity)
        require_identity(self.contract["rvq_id"], rq["rvq_id"], "RVQ")
        require_identity(self.contract["rvq_config"], rq["config"], "RVQ configuration")

    def describe(self, features, betas, genders, offsets, batch_size=64):
        # The API has no reference-caption or future-label argument.
        encoded = self.motion.encode_pair(features, betas, genders, offsets, batch_size, self.rvq)
        text = serialize(encoded["ids"], encoded["spans"], self.rvq.config, self.motion.identity["future"])
        result = decode_text(self.model, self.tokenizer, text, self.contract)
        return dict(result, identity=self.motion.identity, rvq_id=self.contract["rvq_id"],
                    captioner_id=self.captioner_id, tokenizer_id=self.contract["tokenizer_id"],
                    frames=int(encoded["frames"]), fps=int(encoded["fps"]),
                    windows=len(encoded["spans"]), spans=encoded["spans"].tolist(),
                    task="offline_complete_pair_description")


def text_metrics(prediction, references):
    def words(s):
        return re.findall(r"[\w]+", s.lower())
    p = words(prediction)
    scores = []
    for reference in references:
        r = words(reference)
        common = sum((Counter(p) & Counter(r)).values())
        f1 = 2 * common / max(1, len(p)+len(r))
        previous = [0] * (len(r)+1)
        for word in p:
            current = [0]
            for j, target in enumerate(r):
                current.append(previous[j]+1 if word == target else max(previous[j+1], current[-1]))
            previous = current
        rouge = 2 * previous[-1] / max(1, len(p)+len(r))
        scores.append((f1, rouge))
    return dict(word_f1=max(s[0] for s in scores), rouge_l_f1=max(s[1] for s in scores))


def evaluate_captioner(cache, checkpoint, rvq, output, split="test", limit=0, device="cpu", seed=0):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    manifest = read_manifest(cache, "tokens")
    model_rvq, rq = load_rvq(rvq, "cpu", manifest["identity"])
    require_identity(manifest["rvq_id"], rq["rvq_id"], "RVQ cache")
    tokenizer, model, saved = load_language(checkpoint, device)
    contract = saved["contract"]
    require_identity(contract["identity"], manifest["identity"])
    require_identity(contract["rvq_id"], manifest["rvq_id"], "RVQ")
    require_identity(contract["rvq_config"], manifest["rvq_config"], "RVQ configuration")
    out = new_output(output)
    rows = [r for r in manifest["records"] if r["split"] == split]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty evaluation split")
    modes = ["normal", "shuffle", "remove_actor", "remove_reactor"]
    if model_rvq.config["levels"] == 2:
        modes.append("base_only")
    results = []
    for row in rows:
        data = load_arrays(cache, row)
        for mode in modes:
            text = serialize(data["ids"], data["spans"], model_rvq.config, manifest["identity"]["future"], mode, seed)
            prediction = decode_text(model, tokenizer, text, contract)
            results.append(dict(episode=row["episode"], mode=mode, prediction=prediction,
                                references=row["captions"], metrics=text_metrics(prediction["text"], row["captions"]),
                                review=dict(action_error=None, role_error=None, temporal_error=None, notes="")))
    summary = {}
    for mode in modes:
        selected = [r for r in results if r["mode"] == mode]
        summary[mode] = {k: float(np.mean([r["metrics"][k] for r in selected])) for k in ("word_f1", "rouge_l_f1")}
        summary[mode]["valid_fraction"] = float(np.mean([r["prediction"]["valid"] for r in selected]))
    report = dict(task=contract["task"], split=split, limit=limit, seed=seed, summary=summary, samples=results,
                  captioner_id=saved["checkpoint_id"],
                  contract=contract, notes=["Metrics are lexical multi-reference proxies, not semantic correctness.",
                  "Review fields require human annotation; they are not automatic error-rate measurements.",
                  "base_only removes residual symbols from a K=2 captioner; a matched K=1 baseline requires separate RVQ/T5 training."])
    dump_json(out / "report.json", report)
    return summary
