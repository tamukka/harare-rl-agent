import osmnx as ox

def save_harare_graph():
    # Download and simplify graph
    G = ox.graph_from_place("Harare, Zimbabwe", network_type="drive")
    # G = ox.simplify_graph(G)

    # Save nodes and edges with all needed metadata (including 'key')
    ox.save_graph_geopackage(G, filepath="../data/harare_graph.gpkg")

if __name__ == "__main__":
    save_harare_graph()