from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moreact.losses import GLOBAL_MASK_NORMALIZATION, masked_huber


def _mask_worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        parameter = torch.tensor(.2, requires_grad=True)
        # Rank zero has NO contacts; rank one has three. Both must participate.
        inputs = parameter * (torch.empty(0) if rank == 0 else torch.tensor([1.,2.,3.]))
        token = GLOBAL_MASK_NORMALIZATION.set(True)
        loss = masked_huber(inputs, torch.zeros_like(inputs))
        GLOBAL_MASK_NORMALIZATION.reset(token)
        loss.backward()
        result = torch.stack((loss.detach(), parameter.grad))
        dist.all_reduce(result)
        result /= 2
        if rank == 0:
            torch.save(result, output)
    finally:
        dist.destroy_process_group()


def test_contact_normalization_matches_global_batch_with_empty_rank(tmp_path):
    target = str(tmp_path / 'result.pt')
    mp.spawn(_mask_worker, args=(str(tmp_path / 'rendezvous'),target), nprocs=2, join=True)
    actual = torch.load(target, weights_only=False)
    parameter = torch.tensor(.2, requires_grad=True)
    x = parameter * torch.tensor([1.,2.,3.])
    loss = torch.nn.functional.huber_loss(x, torch.zeros_like(x))
    loss.backward()
    torch.testing.assert_close(actual,torch.stack((loss.detach(),parameter.grad)))
