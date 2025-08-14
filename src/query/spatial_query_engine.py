from dataclasses import dataclass
from typing import Tuple
import geopandas as gpd
from shapely.geometry import Point, Polygon

from ..spatial.spatial_operations import features_within_distance, features_within_polygon, features_intersecting


@dataclass
class SpatialQueryEngine:
	gdf: gpd.GeoDataFrame

	def point_query(self, lat: float, lon: float) -> gpd.GeoDataFrame:
		return features_within_distance(self.gdf, Point(lon, lat), distance_meters=50)

	def range_query(self, lat: float, lon: float, radius_km: float) -> gpd.GeoDataFrame:
		return features_within_distance(self.gdf, Point(lon, lat), distance_meters=radius_km * 1000.0)

	def polygon_query(self, polygon: Polygon) -> gpd.GeoDataFrame:
		return features_within_polygon(self.gdf, polygon)
