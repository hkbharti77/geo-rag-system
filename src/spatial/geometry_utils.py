from typing import Iterable
from shapely.geometry import shape, mapping, Polygon, Point, LineString


def bounds_polygon(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
	return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def centroid_point(geom) -> Point:
	return geom.centroid


def length_of_line(line: LineString) -> float:
	return float(line.length)


def area_of_polygon(poly: Polygon) -> float:
	return float(poly.area)


def to_geojson_dict(geom) -> dict:
	return mapping(geom)


def from_geojson_dict(geojson_obj: dict):
	return shape(geojson_obj)
