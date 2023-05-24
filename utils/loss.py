import numpy as np
import torch as t


def raise_and_suppress_loss(a:t.Tensor, b:t.Tensor, score_a:t.Tensor, score_b:t.Tensor) -> t.Tensor:
    """
    a: NodeIdx in Top-K Beam Search (batch_size, num_nodes)
    b: NodeIdx in Top-2K Beam Search (batch_size, num_nodes)
    return: Intersect NodeIdx (batch_size, num_nodes)
    """

    # Note: We use B to approximate the ground-truth TopK set

    # The basic idea are two folds:
    # 1. Suppress the Nodes in A but not in B
    # 2. Raise    the Nodes in B but not in A

    # Calculate the TopK Gatekeeper
    topk_gatekeeper = t.zeros(a.shape[0])


    # Total R&S Loss
    loss = 0

    a = a.cpu().numpy()
    b = b.cpu().numpy()

    for i in range(a.shape[0]):
        pred_set = a[i]
        real_set = b[i]

        intersect_set, a_ind, b_ind = np.intersect1d(pred_set, real_set, return_indices=True)

        # If the result is identical, we can skip this
        if len(np.setdiff1d(pred_set, real_set)) == 0:
            continue

        # TODO::这个守门员的值到底该设置为多少？
        # 被抑制的值，不能进入TopK List，则守门员的值就应该设置为GT中TopK的最小值
        # 被提升的值，需要进入TopK List，则守门员的值就应该设置为GT中TopK的最小值
        topk_gatekeeper[i] = score_b[i].min()

        suppress_idx = np.ones_like(pred_set)
        suppress_idx[a_ind] = 0
        raise_idx = np.ones_like(real_set)
        raise_idx[b_ind] = 0

        # Suppress Loss: Suppress the Nodes in A but not in B, the score should be lower
        suppress_thsh = score_a[i][suppress_idx].min()
        suppress_loss = (score_a[i][suppress_idx == 1] - suppress_thsh).pow(2).mean()

        # Raise Loss: Raise the Nodes in B but not in A, the score should be higher
        raise_loss = (topk_gatekeeper[i] - score_b[i][raise_idx == 1]).relu().pow(2).mean()

        loss += suppress_loss + raise_loss

    return loss

