#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 09/05/2023 12:52
# @Author : YuHui Li(MerylLynch)
# @File : mock.py
# @Comment : Created By Liyuhui,12:52
# @Completed : No
# @Tested : No

import numpy as np


def mock_data(size, rank):

    tsize, usize, isize = size

    C = np.random.rand(tsize, rank)
    A = np.random.rand(usize, rank)
    B = np.random.rand(isize, rank)

    tensor = np.einsum('tr,ur,ir->tui', C, A, B)

    tensor /= np.max(tensor)

    np.save('mock.npy', tensor)

    return tensor

if __name__ == '__main__':
    mock_data((128, 32, 32), 20)
