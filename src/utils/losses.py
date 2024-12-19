import torch
from itertools import permutations
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse = torch.nn.MSELoss()

def pit_reorder_batch(gen: torch.Tensor, src : torch.Tensor) -> torch.Tensor: 

    batch_size, num_sources = src.shape[:2]
    reordered_signals = torch.zeros_like(gen)

    for batch_idx in range(batch_size):
        max_si_sdr_sum = -float('inf')
        best_permutation = None
        for permutation in permutations(range(num_sources)):
            si_sdr_sum = 0
            for i, j in enumerate(permutation):
                si_sdr_sum += si_sdr(src[batch_idx, i], gen[batch_idx, j]).item()
            if si_sdr_sum > max_si_sdr_sum:
                max_si_sdr_sum = si_sdr_sum
                best_permutation = permutation
        for i, j in enumerate(best_permutation):
            reordered_signals[batch_idx, i] = gen[batch_idx, j]
    return reordered_signals

def calc_losses(gen: torch.Tensor, src : torch.Tensor) -> torch.Tensor:
    return -si_sdr(gen, src).mean()

def si_sdri(gen: torch.Tensor, src : torch.Tensor, mix : torch.Tensor) -> torch.Tensor:
    mix = mix.unsqueeze(1).repeat(1, 2, 1)
    improved = si_sdr(gen, src)
    baseline = si_sdr(mix, src)
    return (improved-baseline)