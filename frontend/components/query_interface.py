import streamlit as st
import geopandas as gpd
import json
from pathlib import Path

from src.data_processing.chunking_strategies import grid_chunks, kmeans_point_clusters
from src.data_processing.spatial_indexing import GridSpatialIndex
from config.settings import CHUNKS_DIR, PROCESSED_DIR


def chunking_controls(gdf: gpd.GeoDataFrame):
	st.subheader("🗺️ Geographic Chunking & Indexing")
	
	# Show data info
	st.info(f"📊 **Loaded Data**: {len(gdf)} features")
	
	mode = st.selectbox(
		"Chunking Strategy", 
		["grid", "kmeans"], 
		index=0,
		help="Grid: Divide area into regular squares | K-means: Group points by proximity"
	)
	indexed = False
	
	if mode == "grid":
		st.write("**Grid Chunking**: Divides the map into regular square cells")
		size = st.number_input(
			"Cell size (degrees)", 
			value=1.0, 
			min_value=0.01, 
			step=0.1,
			help="1.0° ≈ 111km at equator. Smaller values = more detailed chunks"
		)
		if st.button("🗺️ Compute Grid Chunks", type="primary"):
			bboxes = grid_chunks(gdf, cell_size_deg=size)
			if bboxes:
				# Save chunks as GeoJSON
				chunk_features = []
				for i, (minx, miny, maxx, maxy) in enumerate(bboxes):
					chunk_features.append({
						"type": "Feature",
						"properties": {"chunk_id": f"grid_{i}", "cell_size": size},
						"geometry": {
							"type": "Polygon",
							"coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
						}
					})
				
				chunk_geojson = {
					"type": "FeatureCollection",
					"features": chunk_features
				}
				
				# Save to chunks directory
				chunk_file = CHUNKS_DIR / f"grid_chunks_{size}deg.geojson"
				with open(chunk_file, 'w') as f:
					json.dump(chunk_geojson, f, indent=2)
				
				st.success(f"✅ Computed {len(bboxes)} grid chunks")
				st.info(f"💾 Saved to: {chunk_file}")
				indexed = True
			else:
				st.warning("No chunks generated - check your data")
				
	elif mode == "kmeans":
		st.write("**K-means Clustering**: Groups nearby points into natural clusters")
		k = st.number_input(
			"Number of clusters", 
			value=5, 
			min_value=1, 
			step=1,
			help="Recommended: 3-10 clusters for most datasets"
		)
		if st.button("🎯 Compute K-means Clusters", type="primary"):
			bboxes = kmeans_point_clusters(gdf, n_clusters=int(k))
			if bboxes:
				# Save clusters as GeoJSON
				cluster_features = []
				for i, (minx, miny, maxx, maxy) in enumerate(bboxes):
					cluster_features.append({
						"type": "Feature",
						"properties": {"cluster_id": f"cluster_{i}", "n_clusters": k},
						"geometry": {
							"type": "Polygon",
							"coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
						}
					})
				
				cluster_geojson = {
					"type": "FeatureCollection",
					"features": cluster_features
				}
				
				# Save to chunks directory
				cluster_file = CHUNKS_DIR / f"kmeans_clusters_{k}.geojson"
				with open(cluster_file, 'w') as f:
					json.dump(cluster_geojson, f, indent=2)
				
				st.success(f"✅ Computed {len(bboxes)} cluster chunks")
				st.info(f"💾 Saved to: {cluster_file}")
				indexed = True
			else:
				st.warning("No clusters generated - check if you have point data")
	
	# Show existing chunk files with better UX
	chunk_files = list(CHUNKS_DIR.glob("*.geojson"))
	if chunk_files:
		st.subheader("📁 Existing Chunk Files")
		col1, col2 = st.columns([3, 1])
		with col1:
			for file in chunk_files:
				# Extract info from filename
				if "grid_chunks" in file.name:
					size = file.name.split("_")[2].replace("deg.geojson", "")
					st.text(f"🗺️ Grid chunks ({size}° cells)")
				elif "kmeans_clusters" in file.name:
					clusters = file.name.split("_")[2].replace(".geojson", "")
					st.text(f"🎯 K-means clusters ({clusters} clusters)")
				else:
					st.text(f"📄 {file.name}")
		with col2:
			st.text(f"Total: {len(chunk_files)} files")
	else:
		# Only show this if no files exist and user hasn't created any yet
		if not indexed:
			st.info("💡 **Tip**: Run chunking above to create geographic regions for analysis")
	
	return indexed
