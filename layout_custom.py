from collections import defaultdict
import numpy as np

from graph import Graph


def force_directed_layout_angles1(graph: Graph,
                                  position_initial: dict,
                                  fixed: list = None,
                                  normalize: bool = True,
                                  iterations: int = 50,
                                  k: float = None,
                                  threshold: float = 0,
                                  epsilon: float = 1e-6,
                                  additional_force_coef: float = 0.5
                                  ) -> dict:
    """
    Position nodes using variation of Fruchterman-Reingold force-directed algorithm.

    Algorithm description:
     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8444&rep=rep1&type=pdf
    Force edges to be placed at specific angles.
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

    anchors = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [-1, 1],
        [1, -1],
        [0, -1],
        [-1, 0],
        [-1, -1]
    ], dtype=np.float32)
    anchors /= np.linalg.norm(anchors, axis=-1)[:, None]
    mid = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])
    max_anchor_dist = np.linalg.norm(np.array([1, 0]) - mid)

    for iteration in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, epsilon, None, out=distance)
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 k * k / distance ** 2 - adjacency * distance / k)

        # Force edges to be placed at specific angles
        mask = (adjacency != 0).sum(-1) == 2
        if iteration > 1000:
            directions = delta / distance[:,:,None]
            deltas = anchors[None, None, :, :] - directions[:, :, None, :]
            dists = np.linalg.norm(deltas, axis=-1)
            dists_argmin = np.argmin(dists, -1)

            n, _, _, c = deltas.shape
            dis = deltas[np.repeat(np.arange(n), n), np.tile(np.arange(n), n), dists_argmin.flatten()].reshape(n, n, c)
            dists = dists[np.repeat(np.arange(n), n), np.tile(np.arange(n), n), dists_argmin.flatten()].reshape(n, n, 1)
            deltas_close = dis * (max_anchor_dist - dists)  # linear force
            deltas_close = deltas_close.sum(1)
            deltas_close[mask] = 0
            displacement += deltas_close * additional_force_coef

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


def force_directed_layout_angles2(graph: Graph,
                                  position_initial: dict,
                                  fixed: list = None,
                                  normalize: bool = True,
                                  iterations: int = 50,
                                  k: float = None,
                                  threshold: float = 0,
                                  epsilon: float = 1e-6,
                                  additional_force_coef: float = 3.
                                  ) -> dict:
    """
    Position nodes using variation of Fruchterman-Reingold force-directed algorithm.

    Algorithm description:
     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8444&rep=rep1&type=pdf
    Force edges to be placed at specific angles.
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

        # Force edges to be placed at specific angles
        mask = (adjacency != 0).sum(-1) == 2
        displacement_angle = np.zeros_like(pos)
        for i in range(adjacency.shape[0]):
            if mask[i]:
                j, jj = np.where(adjacency[i])[0]
                u, v = -delta[i, j], -delta[i, jj]
                bisect = u + v
                bisect /= np.linalg.norm(bisect)
                cos = np.dot(u, v) / distance[i, j] / distance[i, jj]
                angle = np.arccos(cos)
                pi = np.pi
                # if 0 < angle < pi / 2:
                #     displacement_angle[i] = bisect * (pi / 2 - angle)
                # elif pi / 2 < angle < 5 / 8 * pi:
                #     displacement_angle[i] = -bisect * (angle - pi / 2)
                # elif 5 / 8 * pi < angle < 3 / 4 * pi:
                #     displacement_angle[i] = bisect * (3 / 4 * pi - angle)
                # elif 3 / 4 * pi < angle < 7 / 8 * pi:
                #     displacement_angle[i] = -bisect * (angle - 3 / 4 * pi)
                # else:
                #     displacement_angle[i] = bisect * (pi - angle)
                if 0 < angle < pi / 2:
                    displacement_angle[i] = bisect * (angle)
                elif pi / 2 < angle < 5 / 8 * pi:
                    displacement_angle[i] = -bisect * (5 / 8 * pi - angle)
                elif 5 / 8 * pi < angle < 3 / 4 * pi:
                    displacement_angle[i] = bisect * (angle - 5 / 8 * pi)
                elif 3 / 4 * pi < angle < 7 / 8 * pi:
                    displacement_angle[i] = -bisect * (7 / 8 * pi - angle)
                else:
                    displacement_angle[i] = bisect * (angle - 7 / 8 * pi)
        displacement += displacement_angle * additional_force_coef

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


def force_directed_layout_initial(graph: Graph,
                                  position_initial: dict,
                                  fixed: list = None,
                                  normalize: bool = True,
                                  iterations: int = 50,
                                  k: float = None,
                                  threshold: float = 0,
                                  epsilon: float = 1e-6,
                                  additional_force_coef: float = 1.
                                  ) -> dict:
    """
    Position nodes using variation of Fruchterman-Reingold force-directed algorithm.

    Algorithm description:
     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8444&rep=rep1&type=pdf
    Nodes are attracted by their initial positions as well. 
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

    pos0 = pos.copy()  # Initial position
    for _ in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, epsilon, None, out=distance)
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 k * k / distance ** 2 - adjacency * distance / k)

        # Add attractive force directed to initial position
        delta0 = pos0 - pos
        distance0 = np.linalg.norm(delta0, axis=-1)
        displacement += delta0 * distance0[:, None] * additional_force_coef

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


def force_directed_layout_line_direction(graph: Graph,
                                         position_initial: dict,
                                         fixed: list = None,
                                         normalize: bool=True,
                                         iterations: int = 50,
                                         k: float=None,
                                         threshold: float=0,
                                         epsilon: float=1e-6,
                                         additional_force_coef: float = 1.
                                         ) -> dict:
    """
    Position nodes using variation of Fruchterman-Reingold force-directed algorithm.
    
    Algorithm description:
     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8444&rep=rep1&type=pdf
    Edge color attribute is used to force edge direction towards general line direction.
    Line is a chain of nodes with edges of the same color.
    Line direction is a difference between its terminal nodes (two nodes ending chain from two sides). 
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

    # Prepare direction vectors of linees determined by color
    color_to_ends = defaultdict(lambda: [])
    nodeslist = list(graph)
    index = dict(zip(nodeslist, range(num_nodes)))
    for (u, v), edge_info in graph.edges.items():
        c = edge_info['color']
        if sum(graph.adjacency[u][nodeslist[i]]['color'] == c for i in np.nonzero(adjacency[index[u]])[0]) == 1:
            color_to_ends[c].append(index[u])
        if sum(graph.adjacency[v][nodeslist[i]]['color'] == c for i in np.nonzero(adjacency[index[v]])[0]) == 1:
            color_to_ends[c].append(index[v])
    color_to_ends = dict(color_to_ends)
    color_to_direction = {}
    for c, ends in color_to_ends.items():
        i, j = color_to_ends[c]
        color_to_direction[c] = pos[i] - pos[j]
        color_to_direction[c] /= np.linalg.norm(color_to_direction[c])
    adjacency_color = np.full((num_nodes, num_nodes), '', dtype=np.object)
    for (u, v), edge_info in graph.edges.items():
        adjacency_color[index[u], index[v]] = edge_info['color']
        adjacency_color[index[v], index[u]] = edge_info['color']

    for _ in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, epsilon, None, out=distance)
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 k * k / distance ** 2 - adjacency * distance / k)

        # Add attractive force directed to corresponding line vector
        correction = np.zeros_like(displacement)
        for i, j in zip(*np.nonzero(adjacency)):
            # if i < j:
            target_direction = color_to_direction[adjacency_color[i, j]]
            direction = delta[i, j] / distance[i, j]
            cos = np.dot(direction, target_direction)
            correction[i] += target_direction * cos - direction
        displacement += correction * additional_force_coef

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
