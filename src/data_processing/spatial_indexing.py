from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import geopandas as gpd


@dataclass
class GridSpatialIndex:
	cell_size_deg: float = 1.0

	def build_index(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
		indexed: List[Dict[str, Any]] = []
		for idx, row in gdf.iterrows():
			minx, miny, maxx, maxy = row.geometry.bounds
			# Compute centroid if possible for better mapping
			centroid_lat = None
			centroid_lon = None
			try:
				cent = row.geometry.centroid
				centroid_lat = float(cent.y)
				centroid_lon = float(cent.x)
			except Exception:
				pass
			indexed.append({
				"id": str(idx),
				"minx": float(minx),
				"miny": float(miny),
				"maxx": float(maxx),
				"maxy": float(maxy),
				"properties": {k: v for k, v in row.drop(labels=["geometry"]).items()},
				"centroid_lat": centroid_lat,
				"centroid_lon": centroid_lon
			})
		return indexed

	@staticmethod
	def to_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"minx": entry["minx"],
			"miny": entry["miny"],
			"maxx": entry["maxx"],
			"maxy": entry["maxy"],
			"centroid_lat": entry.get("centroid_lat"),
			"centroid_lon": entry.get("centroid_lon"),
			**entry.get("properties", {})
		}
