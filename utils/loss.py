
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), mu.size(1), num_samples, mu.size(2), dtype=mu.dtype, device=mu.device)
    samples_sigma = eps.mul(torch.exp(logsigma.unsqueeze(2) * 0.5))
    samples = samples_sigma.add_(mu.unsqueeze(2))
    return samples

def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)
    The following broadcasting operation will be computed:
    (N x Nc x 1 x K x D) - (N x Nc x K x 1 x D) = (N x Nc x K x K x D)
    Parameters
    ----------
    samples1: torch.Tensor (shape: N x Nc x K x D)
    samples2: torch.Tensor (shape: N x Nc x K x D)
    Returns
    -------
    batchwise distance: N x Nc x K ** 2
    """
    if len(samples1.size()) not in [3, 4, 5] or len(samples2.size()) not in [3, 4, 5]:
        raise RuntimeError('expected: 4-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    elif samples1.shape[1] == samples2.shape[1]:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(0)
        result = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
        return result.view(*result.shape[:-2], -1)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')
    if len(samples1.size()) == 5:
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
    elif len(samples1.size()) == 4:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, samples1.size(1), -1)
    else:
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(2)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)


class MCBCELoss(nn.Module):
    def __init__(self, reduction='sum', criterion=nn.BCELoss(reduction='none'), **kwargs):
        super().__init__()
        if reduction not in {'mean', 'sum', None}:
            raise ValueError('unknown reduction {}'.format(reduction))
        self.reduction = reduction

        self.vib_beta = kwargs.get('vib_beta', 0)

        self.criterion = criterion

    def kl_divergence(self, mu, logsigma, reduction='sum'):
        kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
        if reduction == 'sum':
            return kl.sum()
        else:
            return kl.sum(dim=-1).mean()

    def _compute_loss(self, probs, label):
        loss = self.criterion(probs, label)
        loss = loss.sum() if self.reduction == 'sum' else loss.mean()
        if loss != loss:
            print("NaN")

        return {
            'loss': loss
        }

    def forward(self, probs, image_mean, image_logsigma, concept_labels, negative_scale,  **kwargs):
        vib_loss = 0

        loss_dict = {}

        if self.vib_beta != 0:
            vib_loss = self.kl_divergence(image_mean, image_logsigma, reduction=self.reduction)
            loss_dict['vib_loss'] = vib_loss.item()

        t2i_loss = self._compute_loss(probs, concept_labels)
        t2i_loss_l = t2i_loss['loss']
        loss = t2i_loss_l +self.vib_beta * vib_loss

        loss_dict['t2i_loss'] = t2i_loss_l.item()
        loss_dict['negative_scale'] = negative_scale.mean().item()
        loss_dict['loss'] = loss.mean().item()

        return loss, loss_dict
