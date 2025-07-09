import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import os
from shapely.geometry import box

def fetch_satellite_tiles(geojson_path="../data/harare_edges.geojson", out_dir="../data/satellite_tiles", tile_size=0.002):
    edges = gpd.read_file(geojson_path)
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in edges.iterrows():
        center = row.geometry.centroid
        minx, miny = center.x - tile_size, center.y - tile_size
        maxx, maxy = center.x + tile_size, center.y + tile_size
        bbox = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

        ax = gdf.to_crs(epsg=3857).plot(edgecolor='none', facecolor='none')
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        ax.set_axis_off()

        plt.savefig(f"{out_dir}/tile_{idx}.png", bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        if idx >= 99:  # limit to 100 tiles
            break

    print(f"✅ Saved 100 satellite tiles to {out_dir}/")

if __name__ == "__main__":
    fetch_satellite_tiles()