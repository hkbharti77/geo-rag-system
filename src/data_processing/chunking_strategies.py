from typing import List, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import box

try:
	from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
	KMeans = None


def grid_chunks(gdf: gpd.GeoDataFrame, cell_size_deg: float = 1.0) -> List[Tuple[float, float, float, float]]:
	if gdf.empty:
		return []
	minx, miny, maxx, maxy = gdf.total_bounds
	bboxes = []
	x = minx
	while x < maxx:
		y = miny
		while y < maxy:
			bboxes.append((x, y, min(x + cell_size_deg, maxx), min(y + cell_size_deg, maxy)))
			y += cell_size_deg
		x += cell_size_deg
	return bboxes


def kmeans_point_clusters(gdf: gpd.GeoDataFrame, n_clusters: int = 10) -> List[Tuple[float, float, float, float]]:
	if gdf.empty:
		return []
	points = np.array([[geom.x, geom.y] for geom in gdf.geometry if geom.geom_type == "Point"])
	if len(points) == 0 or KMeans is None:
		return []
	model = KMeans(n_clusters=min(n_clusters, len(points)), n_init=5, random_state=42)
	labels = model.fit_predict(points)
	bboxes: List[Tuple[float, float, float, float]] = []
	for label in np.unique(labels):
		cluster_pts = points[labels == label]
		minx, miny = cluster_pts.min(axis=0)
		maxx, maxy = cluster_pts.max(axis=0)
		bboxes.append((float(minx), float(miny), float(maxx), float(maxy)))
	return bboxes
