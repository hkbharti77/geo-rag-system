from dataclasses import dataclass
from typing import Tuple
import geopandas as gpd
from shapely.geometry import Point, Polygon, box

from ..spatial.spatial_operations import features_within_distance, features_within_polygon, features_intersecting


@dataclass
class SpatialQueryEngine:
	gdf: gpd.GeoDataFrame

	def point_query(self, lat: float, lon: float, tolerance_km: float = 1.0) -> gpd.GeoDataFrame:
		return features_within_distance(self.gdf, Point(lon, lat), distance_meters=tolerance_km * 1000.0)

	def range_query(self, lat: float, lon: float, radius_km: float) -> gpd.GeoDataFrame:
		return features_within_distance(self.gdf, Point(lon, lat), distance_meters=radius_km * 1000.0)

	def polygon_query(self, polygon: Polygon) -> gpd.GeoDataFrame:
		return features_within_polygon(self.gdf, polygon)

	def bbox_query(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> gpd.GeoDataFrame:
		bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
		return features_intersecting(self.gdf, bbox_geom)
