from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
import pandas as pd
import geopandas as gpd
import os

try:
	import rasterio
except Exception:  # pragma: no cover
	rasterio = None


def read_vector_file(path: Union[str, Path]) -> gpd.GeoDataFrame:
	path = Path(path)
	
	# Check if file exists
	if not path.exists():
		raise FileNotFoundError(f"File not found: {path}")
	
	# Try to read based on file extension
	try:
		if path.suffix.lower() in {".geojson", ".json"}:
			# First try to read as GeoJSON
			try:
				return gpd.read_file(path)
			except Exception as e:
				# If it fails, check if it's a regular JSON that might be convertible
				with open(path, 'r', encoding='utf-8') as f:
					data = json.load(f)
				# If it's a regular JSON, try to convert to GeoJSON format
				if isinstance(data, dict) and 'features' in data:
					return gpd.read_file(path)
				else:
					raise ValueError(f"File {path.name} is not in valid GeoJSON format. Expected GeoJSON with 'features' array.")
					
		elif path.suffix.lower() == ".shp":
			# Check for required Shapefile components
			base_path = path.with_suffix('')
			required_files = ['.shp', '.shx', '.dbf']
			missing_files = [f for f in required_files if not (base_path.with_suffix(f)).exists()]
			
			if missing_files:
				raise ValueError(f"Shapefile {path.name} is missing required files: {missing_files}. Shapefiles need .shp, .shx, and .dbf files.")
			
			return gpd.read_file(path)
			
		elif path.suffix.lower() in {".kml", ".gpkg"}:
			return gpd.read_file(path)
			
		elif path.suffix.lower() == ".csv":
			df = pd.read_csv(path)
			# Check for required columns
			if {"latitude", "longitude"}.issubset(df.columns):
				gdf = gpd.GeoDataFrame(
					df,
					geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
					crs="EPSG:4326",
				)
				return gdf
			elif {"lat", "lon"}.issubset(df.columns):
				gdf = gpd.GeoDataFrame(
					df,
					geometry=gpd.points_from_xy(df["lon"], df["lat"]),
					crs="EPSG:4326",
				)
				return gdf
			else:
				raise ValueError(f"CSV file {path.name} missing required coordinate columns. Need 'latitude'/'longitude' or 'lat'/'lon' columns.")
		else:
			raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: .geojson, .json, .shp, .kml, .gpkg, .csv")
			
	except Exception as e:
		raise ValueError(f"Error reading file {path.name}: {str(e)}")


def write_geojson(gdf: gpd.GeoDataFrame, path: Union[str, Path]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	gdf.to_file(path, driver="GeoJSON")


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(content: Dict[str, Any], path: Union[str, Path]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(content, f, ensure_ascii=False, indent=2)


def read_raster(path: Union[str, Path]):
	if rasterio is None:
		raise RuntimeError("rasterio not available")
	return rasterio.open(path)
