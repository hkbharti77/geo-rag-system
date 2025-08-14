from dataclasses import dataclass
import geopandas as gpd


@dataclass
class GeoProcessor:
	output_crs: str = "EPSG:4326"

	def validate_and_normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
		gdf = gdf.dropna(subset=["geometry"]).copy()
		gdf = gdf[gdf.is_valid]
		if gdf.crs is None or str(gdf.crs).upper() != self.output_crs:
			gdf = gdf.to_crs(self.output_crs)
		return gdf
