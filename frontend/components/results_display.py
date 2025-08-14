import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def display_search_results(results: Dict[str, Any], query: str):
	"""Display search results in a user-friendly format"""
	
	st.subheader(f"🔍 Search Results for: '{query}'")
	
	if not results or not results.get("ids"):
		st.info("No results found. Try a different query or check if data is indexed.")
		return
	
	# Extract data from results
	ids = results.get("ids", [[]])[0]
	metadatas = results.get("metadatas", [[]])[0]
	documents = results.get("documents", [[]])[0]
	distances = results.get("distances", [[]])[0]
	
	# Create a clean results table
	results_data = []
	for i, (id_, meta, doc, dist) in enumerate(zip(ids, metadatas, documents, distances)):
		name = meta.get("name", "Unknown")
		lat = meta.get("latitude") or meta.get("miny")
		lon = meta.get("longitude") or meta.get("minx")
		
		# Calculate similarity score (1 - distance, since lower distance = higher similarity)
		similarity = round((1 - dist) * 100, 1) if dist is not None else 0
		
		results_data.append({
			"Rank": i + 1,
			"Name": name,
			"Latitude": f"{lat:.4f}" if lat else "N/A",
			"Longitude": f"{lon:.4f}" if lon else "N/A",
			"Similarity": f"{similarity}%",
			"ID": id_
		})
	
	# Display results table
	if results_data:
		df = pd.DataFrame(results_data)
		st.dataframe(df, use_container_width=True, hide_index=True)
		
		# Show top result prominently
		if results_data:
			top_result = results_data[0]
			st.success(f"🎯 **Top Match**: {top_result['Name']} ({top_result['Similarity']} similar)")
			
			# Show coordinates if available
			if top_result['Latitude'] != "N/A":
				st.info(f"📍 **Location**: {top_result['Latitude']}, {top_result['Longitude']}")
	
	# Show raw data in expander for debugging
	with st.expander("🔧 Raw Results (for developers)"):
		st.json(results)


def display_spatial_results(gdf_results, query_type: str, lat: float = None, lon: float = None, radius: float = None):
	"""Display spatial query results in a user-friendly format"""
	
	st.subheader(f"🗺️ {query_type} Results")
	
	if gdf_results.empty:
		st.info(f"No features found within the specified {query_type.lower()}.")
		return
	
	# Show query parameters
	if lat and lon and radius:
		st.info(f"📍 **Query Center**: {lat:.4f}, {lon:.4f} | **Radius**: {radius} km")
	
	# Create results table
	results_data = []
	for idx, row in gdf_results.iterrows():
		# Extract properties, excluding geometry
		props = {k: v for k, v in row.items() if k != 'geometry'}
		name = props.get('name', f'Feature {idx}')
		
		# Get coordinates from geometry if available
		if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
			lat_coord = row.geometry.y
			lon_coord = row.geometry.x
		else:
			lat_coord = lon_coord = "N/A"
		
		results_data.append({
			"Name": name,
			"Latitude": f"{lat_coord:.4f}" if lat_coord != "N/A" else "N/A",
			"Longitude": f"{lon_coord:.4f}" if lon_coord != "N/A" else "N/A",
			**{k: str(v) for k, v in props.items() if k != 'name'}
		})
	
	if results_data:
		df = pd.DataFrame(results_data)
		st.dataframe(df, use_container_width=True, hide_index=True)
		st.success(f"✅ Found {len(results_data)} features")
	
	# Show raw data in expander
	with st.expander("🔧 Raw GeoDataFrame (for developers)"):
		st.write(gdf_results)
