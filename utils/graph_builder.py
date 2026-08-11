# utils/graph_builder.py

class GraphBuilder:
    def __init__(self, roadnet):
        self.roadnet = roadnet
        self.adj_matrix = self.build_graph()

    def build_graph(self):
        # 解析 LibSignal roadnet.json
        return adjacency_matrix

    def get_neighbors(self, node_id):
        return neighbors