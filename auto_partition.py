from collections import deque
import tensorflow as tf


def is_blue_node(node):
    """
      Determines if the given node can be run on the TPU or not (i.e. if the node is blue or not)

      Args:
          node: A node in the tensorflow graph

      Returns:
          True if the node is compatible with TPU, False otherwise.
      """

    supported_ops = {"Add", "AveragePool2d", "Concatenation", "Conv2d", "DepthwiseConv2d", "ExpandDims",
                     "FullyConnected", "L2Normalization",
                     "Logistic", "LSTM", "Maximum", "MaxPool2d", "Mean", "Minimum", "Mul", "Pack", "Pad", "PReLU",
                     "Quantize", "ReduceMax",
                     "ReduceMin", "ReLU", "ReLU6", "ReLUN1To1", "Reshape", "ResizeBilinear", "ResizeNearestNeighbor",
                     "Rsqrt", "Slice",
                     "Softmax", "SpaceToDepth", "Split", "Squeeze", "StridedSlice", "Sub", "Sum", "Squared-difference",
                     "Tanh", "Transpose", "TransposeConv"}

    op_name = node.name
    for sup_op in supported_ops:
        # Check for compatibility with TPU
        if sup_op in op_name:
            return True

    return False


def find_all_blue_nodes(graph):
    # list of supported operations by TPU
    supported_ops = {"Add", "AveragePool2d", "Concatenation", "Conv2d", "DepthwiseConv2d", "ExpandDims",
                     "FullyConnected", "L2Normalization",
                     "Logistic", "LSTM", "Maximum", "MaxPool2d", "Mean", "Minimum", "Mul", "Pack", "Pad", "PReLU",
                     "Quantize", "ReduceMax",
                     "ReduceMin", "ReLU", "ReLU6", "ReLUN1To1", "Reshape", "ResizeBilinear", "ResizeNearestNeighbor",
                     "Rsqrt", "Slice",
                     "Softmax", "SpaceToDepth", "Split", "Squeeze", "StridedSlice", "Sub", "Sum", "Squared-difference",
                     "Tanh", "Transpose", "TransposeConv"}
    blue_nodes = list()

    # get all operations
    ops = graph.get_operations()


    # iterate through all operations in the graph
    for op in ops:
        op_name = op.name
        for sup_op in supported_ops:
            # Check for compatibility with TPU
            if sup_op in op_name:
                blue_nodes.append(op)

    return blue_nodes



# to do: cache the result from call to is_blue_node for speed up
# to do: try to minimze the edges of the cut (the edges between the two subgraphs)
def find_max_blue_subgraph(graph):
    """
    Finds the partition of a directed acyclic graph (DAG) that maximizes the size of the subgraph containing only blue nodes.

    Args:
      graph: A dictionary representing the graph. Keys are nodes, and values are sets of neighbor nodes.

    Returns:
      A tuple containing two sets: the first set represents the nodes in the subgraph with only blue nodes, and the second set represents the remaining nodes.
    """
    # Find blue nodes with no incoming edges (potential roots for BFS)
    blue_roots = find_all_blue_nodes(graph)

    max_size = 0
    optimal_partition = None

    for last_node in blue_roots:
        # first node of the graph since we know for sure that the first node is blue
        # initialize necessary variables for BFS
        visited = set()
        queue = deque([last_node])

        # conduct a modified BFS starting from the last_node
        while queue:
            node = queue.popleft()
            visited.add(node.name)

            # checking if all the neighbours are blue for the current node
            for neighbor in node.inputs:
                if neighbor.name not in visited and is_blue_node(node) == True:
                    queue.append(graph.get_operation_by_name(neighbor.name.split(":")[0]))

        all_operation_names = set()
        for operation in graph.get_operations():
            all_operation_names.add(operation.name)

        # Update optimal partition if current size is larger
        if len(visited) > max_size:
            max_size = len(visited)
            optimal_partition = (visited, all_operation_names - visited)

    return optimal_partition


def traverse_graph(op):
  """
  Recursive function to traverse the graph node by node.

  Args:
    op: A TensorFlow Operation object.
  """
  # Print information about the current node
  """print(f"Node Name: {op.name}")
  print(f"Node Type: {op.type}")
  print(f"Node inputs: {op.inputs}")"""

  # Iterate through the input tensors of the current operation
  for input_tensor in op.inputs:
    # Get the operation that produces this input tensor
    producer_op = input_tensor.op

    # Recursively call the function on the producer operation
    traverse_graph(producer_op)

def main():
    tfv1 = tf.compat.v1
    gdef = tfv1.GraphDef()
    with tfv1.io.gfile.GFile("/Users/yektakocaogullar/Desktop/DLC_ma_superquadruped_resnet_50_iteration-0_shuffle-1/snapshot-700000.pb","rb") as f:
        gdef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(gdef, name='DLC')

    """for i in range(len(g.get_operations())):
        if "Add" in  g.get_operations()[i].name:
            print(g.get_operations()[i].name)"""

    #find_all_blue_nodes(g)

    #first_op = g.get_operation_by_name(g.get_operations()[len(g.get_operations())-1].name)
    #traverse_graph(first_op)

    #print(find_max_blue_subgraph(g, starting_node))

    print(find_max_blue_subgraph(g))

    #print(g.get_operation_by_name(g.get_operations()[len(g.get_operations())-1].name).inputs)
    #print(g.get_operation_by_name(g.get_operations()[len(g.get_operations())-1].name))



if __name__ == '__main__':
    main()