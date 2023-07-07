import numpy as np
import time
from queue import Queue
from joblib import delayed, Parallel
from collections import deque
import psutil
from sklearn.cluster import KMeans


"""
使用方法：
初始化SplitKMeans类，调用train方法，传入数据，返回ids和codes
ids是Item的原始ID，codes是Item对应的叶子节点ID
"""
class ScikitKMeans:

    def __init__(self):
        self.parallel = psutil.cpu_count(logical=False)
        self.n_cluster = 2

    def train(self, data):
        self.ids = np.arange(data.shape[0])
        self.data = data
        queue = Queue()
        mini_batch_queue = Queue()
        queue.put((0, np.array(range(len(self.ids)))))

        # print('Start to Cluster')

        while not queue.empty():
            pcode, index = queue.get()

            if len(index) <= 128:
                mini_batch_queue.put((pcode, index))
            else:
                left_index, right_index = self._cluster(index)
                if len(left_index) > 1:
                    queue.put((2 * pcode + 1, left_index))
                if len(right_index) > 1:
                    queue.put((2 * pcode + 2, right_index))

        qcodes, indice = [], []
        while mini_batch_queue.qsize() > 0:
            pcode, index = mini_batch_queue.get()
            qcodes.append(pcode)
            indice.append(index)
        make_job = delayed(self._minbatch)
        re = Parallel(n_jobs=self.parallel)(make_job(pcode, index) for pcode, index in zip(qcodes, indice))
        id_code_list = []
        for r in re:
            id_code_list.extend(r)
        ids = np.array([id for (id, _) in id_code_list])
        codes = np.array([code for (_, code) in id_code_list])

        # print('cluster all the nodes done, start to rebalance the tree')
        assert (codes <= 0).sum() <= 0
        assert queue.qsize() == 0
        assert len(ids) == len(data)
        return ids, codes

    def _cluster(self, index):
        data = self.data[index]
        kmeans = KMeans(n_clusters=self.n_cluster, n_init=3)
        kmeans.fit(data)
        labels = kmeans.labels_
        km_distances = kmeans.transform(data)
        # km_distances, labels = kmeans.index.search(data, 1)
        l_i = np.where(labels.reshape(-1) == 0)[0]  # l_i is the index of the first cluster of data
        r_i = np.where(labels.reshape(-1) == 1)[0]  # r_i is the index of the second cluster of data
        left_index = index[l_i]
        right_index = index[r_i]
        if len(right_index) - len(left_index) > 1:
            distances = km_distances[r_i]  # kmeans.transform(data[r_i])
            left_index, right_index = self._rebalance(
                left_index, right_index, distances[:, 0])
        elif len(left_index) - len(right_index) > 1:
            distances = km_distances[l_i]  # kmeans.transform(data[l_i])
            left_index, right_index = self._rebalance(
                right_index, left_index, distances[:, 0])
        return left_index, right_index

    def _minbatch(self, pcode, index):
        dq = deque()
        dq.append((pcode, index))
        id_code_list = []
        while dq:
            pcode, index = dq.popleft()  # pop the tuple which is added into the deque early

            if len(index) == 2:
                id_code_list.append((index[0], 2 * pcode + 1))  # (in,code) pair
                id_code_list.append((index[1], 2 * pcode + 2))
                continue
            left_index, right_index = self._cluster(index)
            if len(left_index) > 1:
                dq.append((2 * pcode + 1, left_index))
            elif len(left_index) == 1:
                # code[left_index] = 2 * pcode + 1
                id_code_list.append((left_index[0], 2 * pcode + 1))

            if len(right_index) > 1:
                dq.append((2 * pcode + 2, right_index))
            elif len(right_index) == 1:
                # code[right_index] = 2 * pcode + 2
                id_code_list.append((right_index[0], 2 * pcode + 2))
        return id_code_list

    def _rebalance(self, lindex, rindex, distances):
        sorted_index = rindex[np.argsort(distances)]
        idx = np.concatenate((lindex, sorted_index))
        mid = int(len(idx) / 2)
        return idx[mid:], idx[:mid]
