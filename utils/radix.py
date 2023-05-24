import torch as t


def tree_path(x, narys):

    res = []

    # 所有值都不大于narys为False
    flag = True

    cnt = 0
    while flag:
        remainder = x % narys
        res.append(cnt * narys + remainder)
        x //= narys
        if t.sum(x >= narys) == 0:
            flag = False
    res.reverse()

    # stack them => [bsize, num_nodes, path_len]
    res = t.stack(res, dim=-1)
    return res

