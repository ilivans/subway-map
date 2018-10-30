from collections import OrderedDict, Mapping

import numpy as np


class EdgeView(Mapping):
    def __init__(self, graph):
        self._adjacency = graph.adjacency

    def __len__(self):
        return sum(len(neighbors) for node, neighbors in self._adjacency.items())

    def __iter__(self):
        for node, neighbors in self._adjacency.items():
            for neighbor in neighbors:
                yield (node, neighbor)

    def __getitem__(self, e):
        u, v = e
        return self._adjacency[u][v]


class Graph:
    """
    Undirected graph with optional data for nodes and edges
    """
    def __init__(self):
        """
        Create empty graph.
        """
        self.nodes = OrderedDict()  # Nodes attributes dictionary
        self.adjacency = OrderedDict()

    def __len__(self):
        """
        Number of nodes.
        """
        return len(self.nodes)

    def add_node(self, node, node_info: dict=None):
        """
        Add node with attributes.
        
        If node already exists, attributes are updated.
        :param node: node identifier (any hashable Python object) 
        :param node_info: node attributes
        """
        if node_info is None:
            node_info = {}
        if node in self.nodes:
            self.nodes[node].update(node_info)
        else:
            self.nodes[node] = node_info
            self.adjacency[node] = OrderedDict()

    def add_edge(self, node_u, node_v, edge_info: dict=None):
        """
        Add edge between node_u and node_v.

        If edge already exists, attributes are updated.
        :param node_u: node identifier (any hashable Python object) 
        :param node_v: node identifier (any hashable Python object) 
        :param edge_info: edge attributes
        """
        if node_u not in self.nodes:
            self.add_node(node_u)
        if node_v not in self.nodes:
            self.add_node(node_v)
        edge_info_existed = self.adjacency[node_u].get(node_v, {})
        edge_info_existed.update(edge_info)
        self.adjacency[node_u][node_v] = edge_info_existed
        self.adjacency[node_v][node_u] = edge_info_existed

    @property
    def edges(self) -> EdgeView:
        """
        Edges view as a mapping (u, v) -> edge_info.
        """
        return EdgeView(self)

    def adjacency_matrix(self, weight='weight'):
        """
        Graph adjacency matrix as a numpy array.
        """
        nodelist = list(self.nodes)
        nlen = len(nodelist)
        index = dict(zip(nodelist, range(nlen)))
        A = np.zeros((nlen, nlen), dtype=np.float32)
        for (u, v), edge_info in self.edges.items():
            A[index[u], index[v]] = edge_info.get(weight, 1)
        return A
