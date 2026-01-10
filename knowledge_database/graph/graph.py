"""
Knowledge Graph module for visualizing tag relationships.

This module provides a graph-based representation of tag relationships,
enabling path finding and neighborhood exploration for interactive
knowledge visualization.
"""

import itertools

import networkx as nx

__all__ = ["Graph"]


class Graph:
    """
    A knowledge graph built from tag co-occurrence relationships.

    The graph connects tags that appear together in documents, enabling
    visualization of knowledge domains and discovery of related topics
    through path finding algorithms.

    Parameters
    ----------
    triples : list[dict]
        List of edge dictionaries with 'head' and 'tail' keys representing
        connected tags.

    Attributes
    ----------
    graph : nx.Graph
        The underlying NetworkX graph structure.
    idx_to_node : dict[int, str]
        Mapping from node indices to tag names.
    node_to_idx : dict[str, int]
        Mapping from tag names to node indices.

    Example
    -------
    >>> import json
    >>> from knowledge_database import graph
    >>>
    >>> with open("database/triples.json") as f:
    ...     triples = json.load(f)
    >>>
    >>> kg = graph.Graph(triples=triples)
    >>> nodes, links = kg(tags=["nlp", "transformers"], retrieved_tags=["bert"])
    """

    def __init__(self, triples: list[dict]):
        self.graph = nx.Graph()
        self._document_counts: dict[str, int] = {}

        # Extract unique nodes from all triples
        nodes = {
            **{node["head"]: True for node in triples},
            **{node["tail"]: True for node in triples},
        }

        # Create bidirectional mappings between indices and node names
        self.idx_to_node = dict(enumerate(nodes))
        self.node_to_idx = {node: idx for idx, node in self.idx_to_node.items()}

        # Build the graph edges
        for triple in triples:
            head = triple["head"]
            tail = triple["tail"]
            self.graph.add_edge(self.node_to_idx[head], self.node_to_idx[tail])
            self.graph.add_edge(self.node_to_idx[tail], self.node_to_idx[head])

    @property
    def document_counts(self) -> dict[str, int]:
        """
        Get the number of documents associated with each tag.

        Returns
        -------
        dict[str, int]
            Mapping from tag names to document counts.
        """
        return getattr(self, "_document_counts", {})

    def set_document_counts(self, document_counts: dict[str, int]) -> None:
        """
        Set document counts for each tag.

        Parameters
        ----------
        document_counts : dict[str, int]
            Mapping from tag names to the number of documents containing that tag.
        """
        self._document_counts = document_counts

    def __call__(
        self,
        tags: list[str],
        retrieved_tags: list[str],
        k_yens: int = 3,
        k_walk: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate a subgraph connecting the given tags.

        Finds paths between tags from documents and retrieved tags, creating
        a visualization-ready subgraph with nodes and links.

        Parameters
        ----------
        tags : list[str]
            Tags extracted from matching documents (displayed in blue).
        retrieved_tags : list[str]
            Tags matched directly from the query (displayed in green).
        k_yens : int, default=3
            Maximum number of shortest paths to find between tag pairs.
        k_walk : int, default=3
            Number of neighbors to explore in random walks.

        Returns
        -------
        nodes : list[dict]
            List of node dictionaries with 'id', 'color', 'degree', and 'documentCount'.
        links : list[dict]
            List of edge dictionaries with 'source', 'target', and 'weight'.
        """
        nodes, lonely = [], []
        output_nodes = {}

        # Color coding: blue for document tags, green for query-matched tags
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

        # Find shortest paths between all pairs of known nodes
        if len(nodes) >= 2:
            for start, end in itertools.combinations(nodes, 2):
                if start != end:
                    try:
                        paths += self.yens(start=start, end=end, k=k_yens)
                    except Exception:
                        # No path exists between start and end
                        continue

        # Fall back to neighborhood exploration for single nodes or disconnected graphs
        if len(nodes) == 1 or len(paths) == 0:
            for start in nodes:
                paths.append(self.walk(start=start, k=k_walk))

        # Add intermediate nodes discovered through path finding
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

    def yens(self, start: int, end: int, k: int) -> list[list[int]]:
        """
        Find k-shortest paths between two nodes using Yen's algorithm.

        Parameters
        ----------
        start : int
            Index of the starting node.
        end : int
            Index of the ending node.
        k : int
            Maximum number of paths to return.

        Returns
        -------
        list[list[int]]
            List of paths, where each path is a list of node indices.
            Only paths with 3 or fewer nodes are included.
        """
        paths = []

        for idx, path in enumerate(nx.shortest_simple_paths(self.graph, start, end)):
            # Filter out overly long paths for cleaner visualization
            if len(path) <= 3:
                paths.append(path)

            if idx > k:
                break

        return paths

    def walk(self, start: int, k: int) -> list[int]:
        """
        Perform a breadth-first walk from a starting node.

        Parameters
        ----------
        start : int
            Index of the starting node.
        k : int
            Maximum number of neighbors to collect.

        Returns
        -------
        list[int]
            List of node indices including the start node and its neighbors.
        """
        neighbours = [start]
        for n, node in enumerate(nx.all_neighbors(self.graph, start)):
            neighbours.append(node)
            if n > k:
                return neighbours

        return neighbours

    def format_triples(self, paths: list[list[int]]) -> list[dict]:
        """
        Convert paths to weighted edge dictionaries for visualization.

        Aggregates edge occurrences across all paths to compute edge weights,
        where more frequently traversed edges have higher weights.

        Parameters
        ----------
        paths : list[list[int]]
            List of paths, where each path is a list of node indices.

        Returns
        -------
        list[dict]
            List of edge dictionaries with 'source', 'target', 'value',
            'weight', and 'relation' keys.
        """
        triples: dict[tuple, int] = {}

        # Count edge occurrences across all paths
        for path in paths:
            for start, end in zip(path[:-1], path[1:], strict=False):
                key = tuple(sorted([start, end]))
                if key not in triples:
                    triples[key] = 1
                else:
                    triples[key] += 1

        max_weight = max(triples.values()) if triples else 1

        # Convert to visualization format with normalized weights
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

    def expand(self, node_id: str, k: int = 10) -> tuple[list[dict], list[dict]]:
        """
        Get neighbors of a node for progressive graph expansion.

        Used for interactive exploration where users can click on nodes
        to reveal their connections.

        Parameters
        ----------
        node_id : str
            The tag name of the node to expand.
        k : int, default=10
            Maximum number of neighbors to return.

        Returns
        -------
        nodes : list[dict]
            List of node dictionaries including the expanded node and its neighbors.
        links : list[dict]
            List of edge dictionaries connecting the node to its neighbors.
        """
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
            links.append(
                {
                    "source": node_id,
                    "target": neighbor_name,
                    "value": 1,
                    "weight": 0.5,
                }
            )

        return list(output_nodes.values()), links
