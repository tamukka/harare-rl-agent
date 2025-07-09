import osmnx as ox
import matplotlib.pyplot as plt

def fetch_harare_roads():
    #define place name
    place = "Harare, Zimbabwe"
    graph = ox.graph_from_place(place, network_type='drive')
    # graph = ox.simplify_graph(graph)
    fig, ax = ox.plot_graph(graph, figsize=(10,10), node_size=5, edge_color='black', edge_linewidth=0.5)
    fig.savefig("../data/harare_road_network.png")
    print("Harare road network downlaoded and save to data/harare_road_network.png")
    return graph
if __name__=="__main__":
    fetch_harare_roads()