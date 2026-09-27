import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from moreact.data import load_stats, InterXDataset, condition_window
from moreact.geometry import transform_features
from moreact.models import ReactionVAE, ReactionDenoiser, position
from moreact.generate import ReactionGenerator, rollout_arrays
from moreact.train import save_checkpoint, load_checkpoint
from moreact.semantics.tokenizer import SharedMotionTokenizer, windows
from moreact.semantics.rvq import ResidualQuantizer, load_rvq
from moreact.semantics.common import load_arrays, read_manifest, tensor_digest
from moreact.semantics.cache import cache_latents, fit_rvq, cache_tokens
from moreact.semantics.language import serialize, audit_lengths, token_identity


@pytest.fixture
def checkpoint(tiny_config, tmp_path):
    torch.set_num_threads(1)
    vae, denoiser = ReactionVAE(tiny_config), ReactionDenoiser(tiny_config)
    p = tmp_path / "motion.pt"
    save_checkpoint(p, dict(kind="diffusion", config=tiny_config, vae=vae.state_dict(),
                            ema=denoiser.state_dict(), stats=load_stats(tiny_config["data"]["cache"], "cpu"),
                            data_digest="synthetic-only", step=0))
    return p


def test_mean_preserves_rng_and_legacy_encode(tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["dropout"] = .1
    vae = ReactionVAE(cfg).eval()
    a, r, future = torch.randn(2, 4, 276), torch.randn(2, 4, 276), torch.randn(2, 2, 276)
    before = torch.get_rng_state().clone()
    mean = vae.encode_mean(a, r, future)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(mean, vae.encode_mean(a, r, future), rtol=0, atol=0)
    # Reproduce the pre-refactor encode expression, including its one RNG draw.
    h = vae.history(a, r)
    f = vae.future_embed(future)
    f = f + vae.future_role + position(f.shape[1], f.shape[2], f.device, f.dtype)
    params = vae.to_latent(vae.encoder(torch.cat((vae.distribution_tokens.expand(len(h), -1, -1), h, f), 1))[:, :2])
    expected_mu, logvar = params[:, :1], params[:, 1:2].clamp(-10, 10)
    expected_z = expected_mu + torch.exp(.5 * logvar) * torch.randn_like(expected_mu)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    z, mu, actual_lv = vae.encode(a, r, future)
    for actual, expected in ((z, expected_z), (mu, expected_mu), (actual_lv, logvar)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="eval"):
        vae.train().encode_mean(a, r, future)


def test_windows_cover_tail_without_padding():
    assert windows(10, 2, 8).tolist() == [[2, 10, 8]]
    assert windows(11, 2, 8).tolist() == [[2, 10, 8], [3, 11, 1]]
    assert windows(19, 2, 8).tolist() == [[2, 10, 8], [10, 18, 8], [11, 19, 1]]
    with pytest.raises(ValueError):
        windows(9, 2, 8)


def test_shared_encoding_role_mapping_and_chunking(checkpoint, tiny_config):
    rng = torch.get_rng_state().clone()
    tokenizer = SharedMotionTokenizer.from_checkpoint(checkpoint)
    assert torch.equal(rng, torch.get_rng_state())
    record = dict(file="episodes/sample0.npz")
    data = load_arrays(tiny_config["data"]["cache"], record)
    data["features"] = data["features"][:, :15]
    data["offsets"] = np.array([[.1, .02, .2], [-.1, .03, -.2]], dtype=np.float32)
    kwargs = {k: data[k] for k in ("features", "betas", "genders", "offsets")}
    a = tokenizer.encode_pair(**kwargs, batch_size=1)
    b = tokenizer.encode_pair(**kwargs, batch_size=64)
    np.testing.assert_allclose(a["mu"], b["mu"], rtol=1e-5, atol=1e-6)
    assert int(a["spans"][:, 2].sum()) == 15 - 4
    x = torch.from_numpy(data["features"])
    off = torch.from_numpy(data["offsets"])
    for role in (0, 1):
        other = 1-role
        a_hist, r_hist, origin, basis = condition_window(x[other, :4][None], x[role, :4][None],
                                                        off[[other, role]][None], tokenizer.stats)
        gt = transform_features(x[role, 4:6][None], origin, basis, off[role][None])
        mu = tokenizer.vae.encode_mean(a_hist, r_hist, (gt-tokenizer.stats["mean"])/tokenizer.stats["std"])
        np.testing.assert_allclose(a["mu"][0, role], mu[0, 0].numpy(), rtol=1e-5, atol=1e-6)
    tokenizer.assert_frozen(full=True)
    with torch.no_grad():
        next(tokenizer.vae.parameters()).add_(1)
    with pytest.raises(ValueError, match="mutated"):
        tokenizer.encode_pair(**kwargs)


def test_rvq_residual_codes_and_no_eval_update():
    q = ResidualQuantizer(dim=1, size=2, levels=2)
    q.codebooks.copy_(torch.tensor([[[0.], [10.]], [[0.], [2.]]]))
    q.initialized.fill_(True)
    q.eval()
    x = torch.tensor([[2.], [12.]])
    codes = q.ids(x)
    assert codes.tolist() == [[0, 1], [1, 1]]
    torch.testing.assert_close(q.decode_ids(codes), x)
    before = tensor_digest(q.state_dict())
    q.ids(x)
    assert before == tensor_digest(q.state_dict())
    with pytest.raises(RuntimeError):
        q.update(x)
    with pytest.raises(ValueError):
        q.decode_ids(torch.tensor([[0, 2]]))


@pytest.fixture
def caches(checkpoint, tiny_config, tmp_path):
    latent = tmp_path / "latent"
    cache_latents(checkpoint, tiny_config["data"]["cache"], latent)
    rq = tmp_path / "rq"
    fit_rvq(latent, rq, epochs=2, batch_size=3, size=4)
    tokens = tmp_path / "tokens"
    cache_tokens(latent, rq / "best.pt", tokens)
    return latent, rq, tokens


def test_caches_identity_frozen_snapshot_and_rvq_resume(caches, tmp_path):
    latent, rq, tokens = caches
    m = read_manifest(tokens, "tokens")
    assert m["counts"] == dict(train=1, val=1, test=1)
    assert load_arrays(tokens, m["records"][0])["ids"].shape[1:] == (2, 2)
    partial = tmp_path / "partial"
    fit_rvq(latent, partial, epochs=1, batch_size=3, size=4)
    fit_rvq(latent, partial, epochs=2, batch_size=3, size=4, resume=partial / "last.pt")
    first, resumed = load_checkpoint(rq / "last.pt"), load_checkpoint(partial / "last.pt")
    assert first["rvq_id"] == resumed["rvq_id"]
    k1 = tmp_path / "k1"
    fit_rvq(latent, k1, epochs=1, batch_size=3, size=4, levels=1)
    k1_tokens = tmp_path / "k1_tokens"
    k1_manifest = cache_tokens(latent, k1 / "best.pt", k1_tokens)
    assert load_arrays(k1_tokens, k1_manifest["records"][0])["ids"].shape[-1] == 1
    for k in first["model"]:
        torch.testing.assert_close(first["model"][k], resumed["model"][k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_rvq(rq / "best.pt", identity=dict(m["identity"], fps=99))
    with pytest.raises(FileExistsError):
        cache_tokens(latent, rq / "best.pt", tokens)
    f = tokens / m["records"][0]["file"]
    with f.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        read_manifest(tokens)


def test_validation_latents_never_update_codebook(caches, tmp_path):
    from moreact.semantics.common import write_manifest, sha256
    latent, rq, _ = caches
    m = read_manifest(latent)
    for r in m["records"]:
        if r["split"] != "train":
            data = load_arrays(latent, r)
            data["mu"] += 123.
            np.savez_compressed(latent / r["file"], **data)
            r["sha256"] = sha256(latent / r["file"])
    m.pop("manifest_id")
    write_manifest(latent, m)
    other = tmp_path / "changed_validation"
    fit_rvq(latent, other, epochs=2, batch_size=3, size=4)
    assert load_checkpoint(rq / "last.pt")["rvq_id"] == load_checkpoint(other / "last.pt")["rvq_id"]


def test_generation_identical_after_shared_understanding(checkpoint, tiny_config):
    class Body:
        def model(self, gender):
            return None

        def repair(self, generated, previous, betas, genders):
            return generated

    data = load_arrays(tiny_config["data"]["cache"], dict(file="episodes/sample0.npz"))
    g1, g2 = ReactionGenerator(checkpoint, seed=19), ReactionGenerator(checkpoint, seed=19)
    g1.body = g2.body = Body()
    shared = SharedMotionTokenizer.from_generator(g2)
    assert shared.vae is g2.vae
    before = torch.get_rng_state().clone()
    shared.encode_pair(*(data[k] for k in ("features", "betas", "genders", "offsets")))
    assert torch.equal(before, torch.get_rng_state())
    args = [torch.from_numpy(data["features"][0]), torch.from_numpy(data["features"][1, :4])]
    args += [torch.from_numpy(data[k]) for k in ("betas", "genders", "offsets")]
    a, _ = rollout_arrays(g1, *args, frames=6)
    b, _ = rollout_arrays(g2, *args, frames=6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.fixture
def tiny_t5(tmp_path):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    vocab = {word: i for i, word in enumerate(["<pad>", "</s>", "<unk>", "A", "approaches", "B", "Describe", "the", "actor", "reactor"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    raw.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    out = tmp_path / "language_base"
    tok.save_pretrained(str(out))
    cfg = transformers.T5Config(vocab_size=len(vocab), d_model=16, d_ff=32, d_kv=8,
                                num_layers=1, num_decoder_layers=1, num_heads=2,
                                dropout_rate=0., pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    transformers.T5ForConditionalGeneration(cfg).save_pretrained(str(out))
    return out


@pytest.mark.parametrize("accumulation,checkpointing", [(1, False), (2, True)])
def test_language_train_reload_resume_and_audit(caches, tiny_t5, tmp_path, accumulation, checkpointing):
    from moreact.semantics.language import train_captioner as train_impl, load_language, InteractionCaptioner, evaluate_captioner
    from functools import partial
    train_captioner = partial(train_impl, gradient_accumulation=accumulation,
                              gradient_checkpointing=checkpointing)
    latent, rq, tokens = caches
    snapshot_before = (latent / "vae_snapshot.pt").read_bytes()
    rvq_before = (rq / "best.pt").read_bytes()
    out = tmp_path / "captioner"
    result = train_captioner(tokens, out, steps=1, base_model=tiny_t5, validate_every=1)
    assert np.isfinite(result["val_nll"])
    train_captioner(tokens, out, steps=2, base_model=tiny_t5, validate_every=1, resume=out / "last.pt")
    tok, model, saved = load_language(out / "last.pt")
    assert saved["step"] == 2
    uninterrupted = tmp_path / "captioner_whole"
    train_captioner(tokens, uninterrupted, steps=2, base_model=tiny_t5, validate_every=1)
    whole = load_checkpoint(uninterrupted / "last.pt")
    for key, value in saved["model"].items():
        torch.testing.assert_close(value, whole["model"][key], rtol=0, atol=0)
    assert token_identity(tok) == saved["contract"]["tokenizer_id"]
    assert snapshot_before == (latent / "vae_snapshot.pt").read_bytes()
    assert rvq_before == (rq / "best.pt").read_bytes()
    # Budget rejection writes an audit and never silently truncates.
    with pytest.raises(ValueError, match="budget"):
        audit_lengths(tok, [dict(episode="long", split="train", text="A "*5000, captions=["A"])], tmp_path / "audit.json")
    assert json.loads((tmp_path / "audit.json").read_text())["errors"]
    m = read_manifest(tokens)
    data = load_arrays(tokens, m["records"][0])
    text = serialize(data["ids"], data["spans"], m["rvq_config"], m["identity"]["future"])
    assert len(tok.encode(text)) < 4096
    assert "<rvq1_" not in serialize(data["ids"], data["spans"], m["rvq_config"], m["identity"]["future"], "base_only")
    assert "<actor>" not in serialize(data["ids"], data["spans"], m["rvq_config"], m["identity"]["future"], "remove_actor")
    motion = SharedMotionTokenizer.from_checkpoint(latent / "vae_snapshot.pt")
    rng_before_captioner = torch.get_rng_state().clone()
    captioner = InteractionCaptioner(motion, rq / "best.pt", out / "last.pt")
    # Exercise real T5 inference with a bounded tiny decoder in this unit test.
    captioner.contract["max_target"] = 2
    source = load_arrays(read_manifest(latent)["source_cache"], dict(file="episodes/sample2.npz"))
    pred = captioner.describe(*(source[k] for k in ("features", "betas", "genders", "offsets")))
    assert pred["frames"] == 16 and pred["windows"] == 6
    assert torch.equal(rng_before_captioner, torch.get_rng_state())
    assert pred["captioner_id"] == saved["checkpoint_id"]
    assert pred["tokenizer_id"] == saved["contract"]["tokenizer_id"]
    motion.assert_frozen(full=True)
    tok.add_tokens(["<tampered>"])
    tok.save_pretrained(str(out / "tokenizer"))
    with pytest.raises(ValueError, match="vocabulary"):
        load_language(out / "last.pt")
