from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
import numpy as np

from graph import Graph


def draw_graph(graph: Graph,
               pos: dict=None,
               nodes_list: list = None,
               nodes_size: Union[int, list]=50,
               nodes_color: Union[str, list]='r',
               edges_list: list=None,
               edges_color: Union[str, list]='k',
               axes: plt.Axes=None
               ) -> None:
    """
    Draw graph using matplotlib.

    :param graph: graph
    :param pos: node -> position (x,y) dictionary
    :param nodes_list: nodes to draw
    :param nodes_size: nodes size, scalar or sequence
    :param nodes_color: nodes color, scalar or sequence (matplotlib compatible)
    :param edges_list: edges to draw
    :param edges_color: nodes color, scalar or sequence (matplotlib compatible)
    :param axes: axes to draw the graph at
    """
    draw_nodes(graph, pos, nodes_list, nodes_size, nodes_color, axes)
    draw_edges(graph, pos, edges_list, edges_color, axes)
    plt.draw_if_interactive()


def draw_nodes(graph: Graph,
               pos: dict,
               nodes_list: list=None,
               nodes_size: Union[int, list]=50,
               nodes_color: Union[str, list]='r',
               axes: plt.Axes=None
               ) -> None:
    """
    Draw graph nodes.

    :param graph: graph
    :param pos: node -> position (x,y) dictionary
    :param nodes_list: nodes to draw
    :param nodes_size: nodes size, scalar or sequence
    :param nodes_color: nodes color, scalar or sequence (matplotlib compatible)
    :param axes: axes to draw the graph at
    """
    if axes is None:
        axes = plt.gca()

    xy = np.asarray([pos[v] for v in (nodes_list or graph.nodes)])
    node_collection = axes.scatter(xy[:, 0], xy[:, 1],
                                   s=nodes_size,
                                   c=nodes_color)
    node_collection.set_zorder(2)


def draw_edges(graph: Graph,
               pos: dict,
               edges_list: list=None,
               edges_color: Union[str, list]='k',
               axes: plt.Axes=None
               ) -> None:
    """
    Draw graph edges.

    :param graph: graph
    :param pos: node -> position (x,y) dictionary
    :param edges_list: edges to draw
    :param edges_color: nodes color, scalar or sequence (matplotlib compatible)
    :param axes: axes to draw the graph at
    """
    if axes is None:
        axes = plt.gca()

    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in (edges_list or graph.edges)])
    if isinstance(edges_color, str):
        edges_color = (edges_color,)
    edges_color = tuple([colorConverter.to_rgba(c) for c in edges_color])
    edges_collection = LineCollection(edge_pos,
                                      colors=edges_color,
                                      transOffset=axes.transData)
    edges_collection.set_zorder(1)  # Edges go behind nodes.
    axes.add_collection(edges_collection)
