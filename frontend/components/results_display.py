import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Add custom CSS for better styling
st.markdown("""
<style>
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .similarity-high {
        color: #28a745;
        font-weight: bold;
    }
    .similarity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .similarity-low {
        color: #dc3545;
        font-weight: bold;
    }
    .map-link {
        color: #007bff;
        text-decoration: none;
    }
    .map-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


def format_coordinates(lat: float, lon: float) -> str:
	"""Format coordinates in a user-friendly way"""
	if lat is None or lon is None:
		return "Location not available"
	
	# Determine hemisphere
	lat_dir = "N" if lat >= 0 else "S"
	lon_dir = "E" if lon >= 0 else "W"
	
	# Format with direction
	return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"


def get_similarity_color(similarity: float) -> str:
	"""Get color class based on similarity score"""
	if similarity >= 80:
		return "similarity-high"
	elif similarity >= 60:
		return "similarity-medium"
	else:
		return "similarity-low"


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
	
	# Display results in a user-friendly format
	for i, (id_, meta, doc, dist) in enumerate(zip(ids, metadatas, documents, distances)):
		# Calculate similarity score
		similarity = round((1 - dist) * 100, 1) if dist is not None else 0
		
		# Extract location information
		name = meta.get("name", "Unknown Location")
		lat = meta.get("latitude") or meta.get("miny")
		lon = meta.get("longitude") or meta.get("minx")
		
		# Create a card for each result
		with st.container():
			st.markdown("---")
			
			# Header with rank and similarity
			col1, col2 = st.columns([3, 1])
			with col1:
				st.markdown(f"### #{i+1} - {name}")
			with col2:
				# Color code similarity
				if similarity >= 80:
					st.success(f"**{similarity}%** Match")
				elif similarity >= 60:
					st.warning(f"**{similarity}%** Match")
				else:
					st.error(f"**{similarity}%** Match")
			
			# Location information
			if lat and lon:
				# Format coordinates with direction
				lat_dir = "N" if lat >= 0 else "S"
				lon_dir = "E" if lon >= 0 else "W"
				coord_str = f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"
				st.markdown(f"📍 **Location**: {coord_str}")
				
				# Add a map link
				map_url = f"https://www.google.com/maps?q={lat},{lon}"
				st.markdown(f"🗺️ [View on Google Maps]({map_url})")
			
			# Additional metadata in a clean format
			if meta:
				st.markdown("**Details:**")
				meta_display = {}
				
				# Filter and format metadata
				for key, value in meta.items():
					if key not in ['name', 'latitude', 'longitude', 'minx', 'miny', 'maxx', 'maxy']:
						# Format the key to be more readable
						formatted_key = key.replace('_', ' ').title()
						meta_display[formatted_key] = value
				
				# Display additional metadata in columns
				if meta_display:
					cols = st.columns(min(3, len(meta_display)))
					for idx, (key, value) in enumerate(meta_display.items()):
						with cols[idx % len(cols)]:
							st.markdown(f"**{key}:** {value}")
	
	# Summary
	st.markdown("---")
	st.success(f"✅ Found **{len(ids)}** result(s) for your search")
	
	# Show raw data in expander for developers
	with st.expander("🔧 Raw Data (for developers)"):
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
	
	# Display results in a user-friendly format
	for idx, row in gdf_results.iterrows():
		# Extract properties, excluding geometry
		props = {k: v for k, v in row.items() if k != 'geometry'}
		name = props.get('name', f'Feature {idx}')
		
		# Get coordinates from geometry if available
		if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
			lat_coord = row.geometry.y
			lon_coord = row.geometry.x
		else:
			lat_coord = lon_coord = None
		
		# Create a card for each result
		with st.container():
			st.markdown("---")
			
			# Header
			st.markdown(f"### #{idx+1} - {name}")
			
			# Location information
			if lat_coord and lon_coord:
				# Format coordinates with direction
				lat_dir = "N" if lat_coord >= 0 else "S"
				lon_dir = "E" if lon_coord >= 0 else "W"
				coord_str = f"{abs(lat_coord):.4f}°{lat_dir}, {abs(lon_coord):.4f}°{lon_dir}"
				st.markdown(f"📍 **Location**: {coord_str}")
				
				# Add a map link
				map_url = f"https://www.google.com/maps?q={lat_coord},{lon_coord}"
				st.markdown(f"🗺️ [View on Google Maps]({map_url})")
			
			# Additional properties in a clean format
			if props:
				st.markdown("**Details:**")
				prop_display = {}
				
				# Filter and format properties
				for key, value in props.items():
					if key not in ['name']:
						# Format the key to be more readable
						formatted_key = key.replace('_', ' ').title()
						prop_display[formatted_key] = value
				
				# Display additional properties in columns
				if prop_display:
					cols = st.columns(min(3, len(prop_display)))
					for col_idx, (key, value) in enumerate(prop_display.items()):
						with cols[col_idx % len(cols)]:
							st.markdown(f"**{key}:** {value}")
	
	# Summary
	st.markdown("---")
	st.success(f"✅ Found **{len(gdf_results)}** feature(s) in the {query_type.lower()}")
	
	# Show raw data in expander for developers
	with st.expander("🔧 Raw GeoDataFrame (for developers)"):
		st.write(gdf_results)
