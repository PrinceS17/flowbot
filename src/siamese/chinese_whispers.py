"""
MIT License

Copyright (c) 2018-2019 NLPub

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This is copied from chinese whispers package.
# Here the class of a flow is determined by its nearest neighbors instead
# of a single closest flow.

import random
from collections import defaultdict
from math import log2
from operator import itemgetter
from random import Random
from typing import Any, Callable, Sequence, Tuple, ItemsView, Union, Dict, DefaultDict, Optional, Set, cast

from networkx.classes import Graph
from networkx.utils import create_py_random_state

from collections import Counter
# from .preprocess import measure_time


# noinspection PyPep8Naming
def top_weighting(G: Graph, node: Any, neighbor: Any) -> float:
    """A weight is the edge weight."""
    return cast(float, G[node][neighbor].get('weight', 1.))


# noinspection PyPep8Naming
def linear_weighting(G: Graph, node: Any, neighbor: Any) -> float:
    """A weight is the edge weight divided to the node degree."""
    return cast(float, G[node][neighbor].get('weight', 1.)) / cast(float, G.degree[neighbor])


# noinspection PyPep8Naming
def log_weighting(G: Graph, node: Any, neighbor: Any) -> float:
    """A weight is the edge weight divided to the log2 of node degree."""
    return cast(float, G[node][neighbor].get('weight', 1.)) / log2(G.degree[neighbor] + 1)

def inv_sq_weighting(G: Graph, node: Any, neighbor: Any) -> float:
    """[Jinhui Song] A weight is the inverse sum of square distances.
    1 / (d1 ^ 2 + d2 ^ 2 + ..) = 1 / ( 1 / norm2 ^ 2 ) = weight ^ 2
    """
    return cast(float, G[node][neighbor].get('weight', 1) ** 2)

"""Shortcuts for the node weighting functions."""
WEIGHTING: Dict[str, Callable[[Graph, Any, Any], float]] = {
    'top': top_weighting,
    'lin': linear_weighting,
    'log': log_weighting,
    'inv_sq': inv_sq_weighting
}


# noinspection PyPep8Naming
# @measure_time()
def chinese_whispers(G: Graph, weighting: Union[str, Callable[[Graph, Any, Any], float]] = 'top', iterations: int = 20,
                     seed: Optional[int] = None, label_key: str = 'label',
                     n_voter: int = 0, th: float = 1.0, closest = {},
                     sorted_weight = {}, voter_weight = {}, r_voter = 0.1) -> Graph:
    """Perform clustering of nodes in a graph G using the 'weighting' method.

    Three weighing schemas are available:

    top
      Just use the edge weights from the input graph.

    lin
      Normalize an edge weight by the degree of the related node.

    log
      Normalize an edge weight by the logarithm of the related node degree.

    It is possible to specify the maximum number of iterations as well as the random seed to use."""

    weighting_func = WEIGHTING[weighting] if isinstance(weighting, str) else weighting

    rng: Random = create_py_random_state(seed)

    for i, node in enumerate(G):
        if label_key not in G.nodes[node]:
            G.nodes[node][label_key] = i + 1

    nodes = list(G)

    # unweighted & weighted vote
    # f_weight = lambda v: 1
    # f_weight = lambda v: 1 / Th - 1 / v
    f_weight = lambda v: v

    neighbor_weight = {}    # {node: [neighbors' weight] }
    closest = {}            # {node: closest neighbor}
    sorted_weight = {}      # {node: sorted neighbors' weight}
    voter_weight = {}       # {node: [voters' weight] }
    for i in range(iterations):
        labels = set([G.nodes[x][label_key] for x in G.nodes])
        # print('iteration: ', i, 'number of clusters: ', len(labels))

        changes = False
        rng.shuffle(nodes)

        for node in nodes:
            previous = G.nodes[node][label_key]

            if G[node]:
                # original nearest neighbors
                # scores = score(G, node, weighting_func, label_key)
                # tmp = random_argmax(scores.items(), choice=rng.choice)
                if i == 0:

                    neighbor_weight[node] = \
                        {k: weighting_func(G, node, k) for k in G[node]}.items()
                    
                    # neighbor_weight[node] = [(k, G[node][k]['weight']) for k in G.neighbors(node)]
                    if n_voter == 1:
                        closest[node] = random_argmax(neighbor_weight[node],
                                        choice=rng.choice)
                    elif n_voter > 1:
                        sorted_weight[node] = sorted(neighbor_weight[node],
                                        key=lambda x: x[1], reverse=True)
                    elif n_voter < 1:
                        voter_weight[node] = \
                            {k: v for k, v in neighbor_weight[node] if v >= th}.items()


                # below wastes O(N) for each node
                # neighbor_weight = {k: weighting_func(G, node, k) for k in G[node]}.items()

                if n_voter == 1:
                    # update only when the closest is not too far
                    if weighting_func(G, node, closest[node]) >= th:
                        G.nodes[node][label_key] = G.nodes[closest[node]][label_key]
                    else:
                        print(' too far: ', node, closest,
                              weighting_func(G, node, closest[node]))
                elif n_voter > 1:
                    # [Modification] use n voters instead of just 1
                    # choose the label with the most total weights from n voters
                    # hopefully this would decompose the closest pair / triplet 
                    # case and thus give a better cluster by using more neighbors data

                    # no weight, then becomes n-nearest neighbors
                    label_count = Counter()
                    n_voter_to_use = min(int(len(G.nodes) * r_voter), n_voter)
                    for k, v in sorted_weight[node][:n_voter_to_use]:
                        if v < th:
                            continue
                        # add f_weight for later count, 1 for unweighted addition,
                        # other function for weighted addition
                        label_count.update({G.nodes[k][label_key]: f_weight(v)})
                    if label_count:
                        label, weight = label_count.most_common(1)[0]
                        G.nodes[node][label_key] = label
                elif n_voter < 1:
                    # [Modification] distance based voting
                    # choose voters by distance but not number, and then vote
                    # k: 1 for unweighted voting, k: v for weighted voting

                    # why weight makes it so worse?
                    # should be something mild but not that hard?

                    if len(voter_weight[node]) == 0:
                        continue
                    label_count = Counter()
                    for k, v in voter_weight[node]:
                        label_count.update({G.nodes[k][label_key]: f_weight(v)})
                    label, weight = label_count.most_common(1)[0]
                    G.nodes[node][label_key] = label


            changes = changes or previous != G.nodes[node][label_key]

        if not changes:
            # print('\niteration: ', i, 'number of clusters: ', len(labels))
            break

    return G


# noinspection PyPep8Naming
def score(G: Graph, node: Any, weighting_func: Callable[[Graph, Any, Any], float],
          label_key: str) -> DefaultDict[int, float]:
    """Compute label scores in the given node neighborhood."""

    scores: DefaultDict[int, float] = defaultdict(float)

    if node not in G:
        return scores

    for neighbor in G[node]:
        scores[G.nodes[neighbor][label_key]] += weighting_func(G, node, neighbor)

    return scores


def random_argmax(items: Union[Sequence[Tuple[Any, float]], ItemsView[Any, float]],
                  choice: Callable[[Sequence[Any]], Any] = random.choice) -> Optional[int]:
    """An argmax function that breaks the ties randomly."""
    if not items:
        # https://github.com/python/mypy/issues/1003
        return None

    _, maximum = max(items, key=itemgetter(1))

    keys = [k for k, v in items if v == maximum]

    return cast('Optional[int]', choice(keys))


# noinspection PyPep8Naming
# @measure_time()
def aggregate_clusters(G: Graph, label_key: str = 'label') -> Dict[int, Set[Any]]:
    """Produce a dictionary with the keys being cluster IDs and the values being sets of cluster elements."""

    clusters: Dict[int, Set[Any]] = {}

    for node in G:
        label = G.nodes[node][label_key]

        if label not in clusters:
            clusters[label] = {node}
        else:
            clusters[label].add(node)

    return clusters
