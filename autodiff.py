from typing import List, Dict, Tuple
from basic_operator import Op, Value

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    给定一个节点列表，返回以这些节点结束的拓扑排序列表。
    一种简单的算法是对给定的节点进行后序深度优先搜索（DFS）遍历，
    根据输入边向后遍历。由于一个节点是在其所有前驱节点遍历后才被添加到排序中的，
    因此我们得到了一个拓扑排序。
    """
    topo_order = []
    visited = set()
    for node in node_list:
        if id(node) not in visited:
            topo_sort_dfs(node, visited, topo_order)
    return topo_order
    


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    visited.add(id(node))

    if node.is_leaf():
        topo_order.append(node)
    else:
        for node_input in node.inputs:
            if id(node_input) not in visited:
                topo_sort_dfs(node_input, visited, topo_order)
        topo_order.append(node)
    

def compute_gradient_of_variables(output_tensor, out_grad):
    """
    对输出节点相对于 node_list 中的每个节点求梯度。
    将计算结果存储在每个 Variable 的 grad 字段中。
    """
    # map for 从节点到每个输出节点的梯度贡献列表
    node_to_output_grads_list = {}
    # 我们实际上是在对标量 reduce_sum(output_node) 
    # 而非向量 output_node 取导数。
    # 但这是损失函数的常见情况。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 根据我们要对其求梯度的 output_node，以逆拓扑排序遍历图。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        output_grads_list = node_to_output_grads_list[node]
        node.grad = output_grads_list[0]
        for i in range(1,len(output_grads_list)):
            node.grad += output_grads_list[i]
        if not node.is_leaf():
            for node_input, grad in zip(node.inputs,
                                        node.op.gradient_as_tuple(node.grad, node)):
                if node_input not in node_to_output_grads_list:
                    node_to_output_grads_list[node_input] = [grad]
                else:
                    node_to_output_grads_list[node_input].append(grad)
    
