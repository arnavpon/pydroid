"""
Code citation: https://github.com/lyprince/sdtw_pytorch/blob/master/sdtw.py
paper citation: Soft-DTW: a Differentiable Loss Function for Time-Series
"""

import torch

def softmin(x, gamma):
    dims = tuple([len(x), *x[0].shape])
    x = -torch.cat(x).reshape(dims) / gamma
    return -gamma * torch.logsumexp(x, dim=0)

def soft_dtw_loss_gradients_hessians(y_true, y_pred, gamma=1.0):
    device = 'cuda' if y_true.is_cuda else 'cpu'
    y_true = y_true.unsqueeze(0).unsqueeze(0)
    y_pred = y_pred.unsqueeze(0).unsqueeze(0)

    x_time_dim, y_time_dim = y_true.shape[-1], y_pred.shape[-1]
    R = torch.full((x_time_dim + 2, y_time_dim + 2), float('inf'), device=device)
    R[0, 0] = 0

    # Forward pass
    for i in range(x_time_dim):
        for j in range(y_time_dim):
            D_ij = (y_true[..., i] - y_pred[..., j]).pow(2).sum()
            R[i + 1, j + 1] = D_ij + softmin([R[i, j], R[i + 1, j], R[i, j + 1]], gamma)

    loss = R[-2, -2]

    # Backward pass
    E = torch.zeros((x_time_dim + 2, y_time_dim + 2), device=device)
    E[-1, -1] = 1
    R[R == float('inf')] = -float('inf')
    R[-1, -1] = R[-2, -2]

    for i in range(x_time_dim, 0, -1):
        for j in range(y_time_dim, 0, -1):
            a = torch.exp((R[i - 1, j] - R[i, j] - (y_true[..., i - 1] - y_pred[..., j]).pow(2).sum()) / gamma)
            b = torch.exp((R[i, j - 1] - R[i, j] - (y_true[..., i] - y_pred[..., j - 1]).pow(2).sum()) / gamma)
            c = torch.exp((R[i - 1, j - 1] - R[i, j] - (y_true[..., i - 1] - y_pred[..., j - 1]).pow(2).sum()) / gamma)
            E[i, j] = a * E[i - 1, j] + b * E[i, j - 1] + c * E[i - 1, j - 1]

    gradients = 2 * (E[1:-1, 1].unsqueeze(1) * y_true - E[1:-1, :-1].unsqueeze(1) * y_pred).sum(dim=-1)
    hessians = torch.ones_like(gradients)
    return loss, gradients, hessians
