import osmnx as ox
import geopandas as gpd

def extract_graph_elements():
    place = "Harare, Zimbabwe"

    # Get the road network graph
    graph = ox.graph_from_place(place, network_type='drive')

    # Convert to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)

    # Save as GeoJSON
    nodes.to_file("../data/harare_nodes.geojson", driver='GeoJSON')
    edges.to_file("../data/harare_edges.geojson", driver='GeoJSON')

    print(f"✅ Extracted {len(nodes)} nodes and {len(edges)} edges.")
    return nodes, edges

if __name__ == "__main__":
    extract_graph_elements()