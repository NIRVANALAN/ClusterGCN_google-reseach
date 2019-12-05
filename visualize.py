# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
import randomcolor
import numpy as np


def load_graph(id, root='cluster', specify='single', print_label=False):
    import scipy.sparse as sp
    import numpy as np
    part_adj = sp.load_npz(f'{root}/{specify}/cluster_adj_{id}.npz')
    part_label = np.load(f'{root}/{specify}/cluster_y_{id}.npy')
    if print_label:
        print_label_table(part_label)
    part_label = np.argmax(part_label, axis=1)
    print(f'shape: {part_adj.shape}')
    return part_adj, part_label


def print_label_table(label):
    from tabletext import to_text
    from collections import Counter
    labeled_node = np.argwhere(label)[:, 1]
    print(f'all_node: {len(label)}, labeld: {len(labeled_node)}')
    label_stat = Counter(labeled_node)
    label_table = [['label', 'number', 'percent']]
    for _label_tuple in label_stat.most_common():
        label_table.append([_label_tuple[0], _label_tuple[1],
                            f'{_label_tuple[1]/len(labeled_node)*100:.2f}%'])
    print(to_text(label_table))
    pass
def get_colors(label_number=41):
    import randomcolor
    rand_color = randomcolor.RandomColor()
    colors = rand_color.generate(count=label_number)
    return colors

# colors = get_colors(41)

# part_adj, cluster_label = load_graph(773, print_label=True)
# cluster_G = nx.from_scipy_sparse_matrix(part_adj)

def get_node_pos(cluster_label, mode='raw', predict=None):
    node_pos = {}
    if mode == 'raw':
        for label in set(cluster_label):
            node_pos.update(
                {label: np.argwhere(cluster_label == label).flatten().tolist()})
    elif mode == 'diff':
        assert predict is not None
        true = np.argwhere(cluster_label == predict).flatten().tolist()
        false = set(range(len(cluster_label))).difference(true)
        node_pos.update({'true':true, 'false':false})

    else:
        raise NotImplementedError(f'mode: {mode} unrecognized')

    return node_pos

options = {"node_size": 40, "alpha": 0.8}
def plot_cluster(cluster_G, node_pos, options, colors, figsize=(8,6)):
    pos = nx.spring_layout(cluster_G)
    figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
    for node_label in node_pos:
        nx.draw_networkx_nodes(
            cluster_G, pos, nodelist=node_pos[node_label], node_color=colors[node_label], **options)
    nx.draw_networkx_edges(cluster_G, pos, width=0.2, alpha=0.5)
