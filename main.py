import matplotlib.pyplot as plt

from drawing import draw_graph
from layout import force_directed_layout
from reader import read_moscow_subway, read_spb_subway


def experiment(graph, city_name, file_name):
    pos_init = {node: (node_info['lng'], node_info['lat']) for node, node_info in graph.nodes.items()}
    nodes_list = list(graph.nodes.keys())
    nodes_color = ['#' + graph.nodes[node]['color'] for node in nodes_list]
    edges_list = list(graph.edges)
    edges_color = ['#' + graph.edges[edge]['color'] for edge in edges_list]

    fixed = []
    for node, neighbors in graph.adjacency.items():
        if len(neighbors) == 1:
            fixed.append(node)

    pos = force_directed_layout(
        graph,
        pos_init,
        k=8e-3,
        iterations=2000
    )
    pos_fixed = force_directed_layout(
        graph,
        pos_init,
        fixed,
        k=8e-3,
        iterations=2000
    )

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('{} subway map'.format(city_name), fontsize=30)
    axs[0].set_title('Original', fontsize=20)
    axs[1].set_title('Force-directed', fontsize=20)
    axs[2].set_title('Force-directed, fixed terminals', fontsize=20)

    draw_graph(graph,
               pos_init,
               nodes_list=nodes_list,
               nodes_color=nodes_color,
               edges_list=edges_list,
               edges_color=edges_color,
               axes=axs[0])
    draw_graph(graph,
               pos,
               nodes_list=nodes_list,
               nodes_color=nodes_color,
               edges_list=edges_list,
               edges_color=edges_color,
               axes=axs[1])
    draw_graph(graph,
               pos_fixed,
               nodes_list=nodes_list,
               nodes_color=nodes_color,
               edges_list=edges_list,
               edges_color=edges_color,
               axes=axs[2])
    plt.savefig('images/{}'.format(file_name))


if __name__ == "__main__":
    graph = read_moscow_subway()
    experiment(graph, 'Moscow', 'moscow.png')
    graph = read_spb_subway()
    experiment(graph, 'Saint Petersburg', 'spb.png')
