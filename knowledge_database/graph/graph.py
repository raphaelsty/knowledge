import itertools
import typing

import networkx as nx

__all__ = ["Graph"]


class Graph:
    """Knowledge Graph.

    Example:
    --------

    >>> import json
    >>> from knowledge_database import graph

    >>> with open("database/triples.json", "r") as f:
    ...     triples = json.load(f)

    >>> knowledge_graph = graph.Graph(triples=triples)

    >>> knowledge_graph(tags=["nlp", "llms", "deep-learning"])

    """

    def __init__(self, triples):
        self.graph = nx.Graph()

        nodes = {
            **{node["head"]: True for node in triples},
            **{node["tail"]: True for node in triples},
        }

        self.idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        self.node_to_idx = {node: idx for idx, node in self.idx_to_node.items()}

        for triple in triples:
            head = triple["head"]
            tail = triple["tail"]
            self.graph.add_edge(self.node_to_idx[head], self.node_to_idx[tail])
            self.graph.add_edge(self.node_to_idx[tail], self.node_to_idx[head])

    def __call__(
        self,
        tags: typing.List,
        retrieved_tags: typing.List,
        k_yens: int = 3,
        k_walk: int = 3,
    ):

        nodes, lonely = [], []
        output_nodes = {}

        for list_tag, color in [(tags, "#86E5FF"), (retrieved_tags, "#19bc8e")]:
            for tag in list_tag:
                idx = self.node_to_idx.get(tag, None)
                if idx is None:
                    lonely.append(tag)
                else:
                    nodes.append(idx)
                output_nodes[tag] = {"id": tag, "color": color}

        paths = []

        if len(nodes) >= 2:

            for start, end in itertools.combinations(nodes, 2):

                if start != end:
                    try:
                        paths += self.yens(start=start, end=end, k=k_yens)
                    except:
                        # No path between start and end
                        continue

        if len(nodes) == 1 or len(paths) == 0:

            for start in nodes:

                paths.append(self.walk(start=start, k=k_walk))

        for path in paths:
            for node in path:
                node = self.idx_to_node[node]
                if node not in output_nodes:
                    output_nodes[node] = {
                        "id": node,
                        "color": "#FFFFFF",
                    }

        return list(output_nodes.values()), self.format_triples(paths=paths)

    def yens(self, start: int, end: int, k: int):
        """K-shortest path between start and end node."""
        paths = []

        for idx, path in enumerate(
            nx.shortest_simple_paths(
                self.graph,
                start,
                end,
            )
        ):

            # Avoid too long paths.
            if len(path) <= 3:
                paths.append(path)

            if idx > k:
                break

        return paths

    def walk(self, start: int, k):
        neighbours = [start]
        for n, node in enumerate(nx.all_neighbors(self.graph, start)):
            neighbours.append(node)
            if n > k:
                return neighbours

        return neighbours

    def format_triples(self, paths: typing.List[typing.List[str]]):
        """Convert nodes as triples."""
        triples = {}
        for path in paths:
            for start, end in zip(path[:-1], path[1:]):
                if start != end and f"{end}_{start}" not in triples:
                    triples[f"{start}_{end}"] = True

        links = []
        for triple in triples:
            head, tail = tuple(triple.split("_"))
            head = self.idx_to_node[int(head)]
            tail = self.idx_to_node[int(tail)]
            links.append(
                {
                    "source": head,
                    "relation": "link",
                    "target": tail,
                    "value": 1,
                }
            )

        return links
