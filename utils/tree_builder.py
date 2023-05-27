import math
import numpy as np
class Arguments:

    def __init__(self) -> None:
        
        self.num_users = 16
        self.num_items = 3
        self.num_times = 3
        self.qtypes = ['user']
        self.ktypes = ['item', 'time']
        self.narys = 2


# preserve that a subtree contains all the data points in the leaf nodes
def allocate_leaf_node(args):

    # 1. 获取ktype的维度数量
    num_treenodes = 1
    subtree_size = 0
    for ktype in args.ktypes:
        num_types_instance = getattr(args, f'num_{ktype}s')
        num_nodes_for_type = int(math.log(num_types_instance, args.narys)) + 1

        num_treenodes *= (args.narys ** num_nodes_for_type)
        subtree_size = (args.narys ** num_nodes_for_type) 

    # 2. 树结构数组：有效叶子节点数组、有效树节点数组
    leaf_mask = np.zeros((2 * num_treenodes - 1, ), dtype=np.int32)
    node_mask = np.zeros((2 * num_treenodes - 1, ), dtype=np.int32)
    leaf_mapp = np.zeros((2 * num_treenodes - 1, ), dtype=np.int32)
    leaf_startIdx = num_treenodes - 1

    # 3. 分配叶子节点
    # 如果只有一个被查询维度，则不需要重构叶子节点
    if len(args.ktypes) == 1:
        num_types_instance = getattr(args, f'num_{args.ktypes[0]}s')
        leaf_mask[leaf_startIdx: leaf_startIdx + num_types_instance] = 1
        node_mask[leaf_startIdx: leaf_startIdx + num_types_instance] = 1
        leaf_mapp[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = np.arange(num_types_instance)

    # 如果有两个被查询维度，则需要重构叶子节点，按照最后一个维度去安排叶子节点
    elif len(args.ktypes) == 2:
        num_types_instance_0 = getattr(args, f'num_{args.ktypes[0]}s')
        num_types_instance_1 = getattr(args, f'num_{args.ktypes[1]}s')
        for i in range(num_types_instance_0):
            subtree_offset = i * subtree_size
            leaf_mask[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = 1
            node_mask[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = 1
            leaf_mapp[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = np.arange(i * num_types_instance_1, (i+1) * num_types_instance_1)
    
    else:
        raise NotImplementedError(f'Not support {len(args.ktypes)} x ktypes')

    # 4. 标记树节点有效性
    for i in range(leaf_startIdx - 1, -1, -1):
        valid_flag = 0
        for j in range(args.narys):
            valid_flag |= node_mask[args.narys * i + (j + 1)]
        node_mask[i] = valid_flag

    return

if __name__ == '__main__':
    args = Arguments()
    allocate_leaf_node(args)