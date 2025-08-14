from typing import Dict, Any
import geopandas as gpd


def to_table(gdf: gpd.GeoDataFrame, limit: int = 20) -> Dict[str, Any]:
	data = gdf.head(limit).drop(columns=["geometry"], errors="ignore").to_dict(orient="records")
	return {"rows": data, "count": int(len(gdf))}


def to_geojson(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
	return gdf.to_json()
