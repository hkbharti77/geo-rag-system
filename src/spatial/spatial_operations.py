from typing import List
import geopandas as gpd
from shapely.geometry import Point


def features_within_distance(gdf: gpd.GeoDataFrame, point: Point, distance_meters: float) -> gpd.GeoDataFrame:
	if gdf.crs is None or gdf.crs.to_epsg() != 4326:
		gdf = gdf.to_crs(epsg=4326)
	sphere_buffer_deg = distance_meters / 111_000.0
	buffer = Point(point.x, point.y).buffer(sphere_buffer_deg)
	return gdf[gdf.geometry.intersects(buffer)].copy()


def features_within_polygon(gdf: gpd.GeoDataFrame, polygon) -> gpd.GeoDataFrame:
	return gdf[gdf.geometry.within(polygon)].copy()


def features_intersecting(gdf: gpd.GeoDataFrame, geometry) -> gpd.GeoDataFrame:
	return gdf[gdf.geometry.intersects(geometry)].copy()
