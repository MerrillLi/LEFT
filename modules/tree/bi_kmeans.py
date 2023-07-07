import math
import time

import torch
from sklearn.cluster import KMeans
import numpy as np

def left_child(node):
    return 2 * node + 1

def right_child(node):
    return 2 * node + 2

class BiKMeans:

    def __init__(self, data):
        self.data = data
        self.leaf_size = 2 ** math.ceil(math.log2(len(self.data)))
        self.leaf_startIdx = 2 ** math.ceil(math.log2(len(self.data))) - 1
        self.allocation = {}

    @torch.no_grad()
    def _cluster(self, node, index):
        # 处理边界条件
        if len(index) <= 2:
            if len(index) == 1:
                while node < self.leaf_startIdx:
                    node = left_child(node)
                self.allocation[node] = index
            else:
                right_node = node
                left_node = node
                while left_node < self.leaf_startIdx:
                    left_node = left_child(left_node)
                while right_node < self.leaf_startIdx:
                    right_node = right_child(right_node)
                self.allocation[left_node] = index[0]
                self.allocation[right_node] = index[1]
            return

        # 处理递归情况
        cluster_data = self.data[index]
        cluster = KMeans(n_clusters=2, n_init=3)
        cluster.fit(cluster_data)
        labels = cluster.labels_
        km_distances = cluster.transform(cluster_data)

        left_cluster = index[labels == 0]
        right_cluster = index[labels == 1]

        # 根据距离平衡左右簇
        if len(right_cluster) - len(left_cluster) > 1:
            distance = km_distances[labels == 1]
            left_cluster, right_cluster = self.rebalance(left_cluster, right_cluster, distance[:, 0])
        elif len(left_cluster) - len(right_cluster) > 1:
            distance = km_distances[labels == 0]
            left_cluster, right_cluster = self.rebalance(right_cluster, left_cluster, distance[:, 1])

        self.allocation[node] = index
        # 递归
        self._cluster(left_child(node), left_cluster)
        self._cluster(right_child(node), right_cluster)

    def cluster(self):
        index = np.arange(len(self.data))
        self._cluster(0, index)
        return self.get_tree_allocation()

    def rebalance(self, less_index, more_index, distances):
        sorted_index = more_index[np.argsort(distances)]
        all_index = np.concatenate((less_index, sorted_index))
        mid = int(len(all_index) / 2)
        return all_index[:mid], all_index[mid:]


    def get_tree_allocation(self):

        allocation = np.zeros(self.leaf_size, dtype=int) - 1

        for i in range(self.leaf_size):
            nodeId = i + self.leaf_startIdx
            if nodeId not in self.allocation:
                pass
            else:
                allocation[i] = self.allocation[nodeId]
        # allocation = np.zeros(self.leaf_size, dtype=int) - 1
        # allocation[:len(self.data)] = np.arange(len(self.data))
        return torch.from_numpy(allocation)




if __name__ == '__main__':

    data = np.random.rand(4500, 50)
    allocator = BiKMeans(data)
    startTime = time.time()
    allocator_id = allocator.cluster()
    print(time.time() - startTime)

    validId = allocator_id[allocator_id != -1]
    sortedIdx = np.argsort(validId)
    sorted_allocation = validId[sortedIdx]
    print(allocator_id)
    print(validId)
    print(sorted_allocation)
