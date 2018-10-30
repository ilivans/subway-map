import numpy as np

from graph import Graph


def force_directed_layout(graph: Graph,
                          position_initial: dict,
                          fixed: list = None,
                          normalize: bool = True,
                          iterations: int = 50,
                          k: float = None,
                          threshold: float = 0,
                          epsilon: float = 1e-6
                          ) -> dict:
    """
    Position nodes using variation of Fruchterman-Reingold force-directed algorithm.

    Algorithm description:
     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8444&rep=rep1&type=pdf
    :param graph: graph
    :param position_initial: initial nodes positions as mapping {node -> (x, y)}
    :param fixed: nodes to keep at initial positions
    :param normalize: whether to put initial positions into 1x1 square for stability 
    :param iterations: maximum number of iterations taken
    :param k: ideal distance between nodes
    :param threshold: threshold for relative error in node position changes,
                    iteration stops if the error is below the threshold
    :param epsilon: small value to avoid division by zero
    :return nodes positions mapping {node -> (x, y)}
    """
    if fixed is not None:
        nodes_index = dict(zip(graph.nodes, range(len(graph))))
        fixed = np.asarray([nodes_index[v] for v in fixed])
    pos = np.asarray(list(position_initial.values()))  # Position matrix
    if normalize:
        pos = (pos - pos.mean(0)) / (pos.max(0) - pos.min(0))
    domain_size = (pos.max(0) - pos.min(0)).prod()
    num_nodes = len(graph)
    if k is None:
        k = np.sqrt(domain_size / num_nodes)
    adjacency = graph.adjacency_matrix()
    temperature = np.sqrt(domain_size) * 0.1  # Initial temperature from the article
    dt = temperature / float(iterations + 1)  # Linearly step down to 0
    for _ in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, epsilon, None, out=distance)
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 k * k / distance ** 2 - adjacency * distance / k)
        length = np.linalg.norm(displacement, axis=-1)
        np.clip(length, epsilon, None, out=length)
        delta_pos = np.einsum('ij,i->ij', displacement, temperature / length)
        if fixed is not None:
            delta_pos[fixed] = 0
        pos += delta_pos  # Move nodes
        temperature -= dt  # Cool temperature
        if np.linalg.norm(delta_pos) / num_nodes < threshold:
            break
    position = dict(zip(graph.nodes, pos))
    return position
