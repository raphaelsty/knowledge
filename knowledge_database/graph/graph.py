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
        self._document_counts = {}

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

    @property
    def document_counts(self) -> typing.Dict[str, int]:
        """Get document counts, returning empty dict if not set (for backwards compatibility)."""
        return getattr(self, '_document_counts', {})

    def set_document_counts(self, document_counts: typing.Dict[str, int]):
        """Set document counts for each tag."""
        self._document_counts = document_counts

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
                    degree = 0
                else:
                    nodes.append(idx)
                    degree = self.graph.degree(idx)
                output_nodes[tag] = {
                    "id": tag,
                    "color": color,
                    "degree": degree,
                    "documentCount": self.document_counts.get(tag, 0),
                }

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
            for node_idx in path:
                node_name = self.idx_to_node[node_idx]
                if node_name not in output_nodes:
                    output_nodes[node_name] = {
                        "id": node_name,
                        "color": "#FFFFFF",
                        "degree": self.graph.degree(node_idx),
                        "documentCount": self.document_counts.get(node_name, 0),
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
        """Convert nodes as triples with edge weights."""
        triples = {}
        for path in paths:
            for start, end in zip(path[:-1], path[1:]):
                key = tuple(sorted([start, end]))
                if key not in triples:
                    triples[key] = 1
                else:
                    triples[key] += 1

        max_weight = max(triples.values()) if triples else 1

        links = []
        for (start, end), count in triples.items():
            head = self.idx_to_node[start]
            tail = self.idx_to_node[end]
            links.append(
                {
                    "source": head,
                    "relation": "link",
                    "target": tail,
                    "value": count,
                    "weight": count / max_weight,
                }
            )

        return links

    def expand(self, node_id: str, k: int = 10):
        """Get neighbors of a specific node for progressive graph expansion."""
        idx = self.node_to_idx.get(node_id)
        if idx is None:
            return [], []

        output_nodes = {
            node_id: {
                "id": node_id,
                "color": "#86E5FF",
                "degree": self.graph.degree(idx),
                "documentCount": self.document_counts.get(node_id, 0),
            }
        }

        links = []
        for i, neighbor_idx in enumerate(nx.all_neighbors(self.graph, idx)):
            if i >= k:
                break
            neighbor_name = self.idx_to_node[neighbor_idx]
            output_nodes[neighbor_name] = {
                "id": neighbor_name,
                "color": "#FFFFFF",
                "degree": self.graph.degree(neighbor_idx),
                "documentCount": self.document_counts.get(neighbor_name, 0),
            }
            links.append({
                "source": node_id,
                "target": neighbor_name,
                "value": 1,
                "weight": 0.5,
            })

        return list(output_nodes.values()), links
