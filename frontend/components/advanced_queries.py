import streamlit as st
import geopandas as gpd
from typing import Dict, Any, Optional, Tuple
import folium

# Import streamlit_folium with fallback
try:
    from streamlit_folium import st_folium
except Exception:
    def st_folium(*args, **kwargs):
        st.warning("streamlit-folium is not available; interactive map rendering is disabled.")
        return None


def advanced_search_interface(retrieval_engine, spatial_engine):
    """Enhanced search interface with multiple query types"""
    
    st.header("🔍 Advanced Geographic Search")
    
    # Query type selector
    query_type = st.selectbox(
        "Search Type",
        ["Semantic Search", "Spatial Query", "Hybrid Search", "Query Examples"],
        help="Choose your search strategy"
    )
    
    if query_type == "Semantic Search":
        semantic_search_panel(retrieval_engine)
    elif query_type == "Spatial Query":
        spatial_query_panel(spatial_engine)
    elif query_type == "Hybrid Search":
        hybrid_search_panel(retrieval_engine, spatial_engine)
    elif query_type == "Query Examples":
        query_examples_panel(retrieval_engine, spatial_engine)


def semantic_search_panel(retrieval_engine):
    """Enhanced semantic search with filters"""
    
    st.subheader("🎯 Semantic Search")
    
    # Query input with examples
    query_examples = [
        "Find major cities",
        "Locations near water",
        "Urban areas",
        "Coastal cities",
        "Capital cities"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your search query", value="Find major cities")
    with col2:
        if st.button("💡 Examples"):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)
    
    if st.session_state.get('show_examples', False):
        st.write("**Try these examples:**")
        for example in query_examples:
            if st.button(example, key=f"ex_{example}"):
                st.session_state.query = example
                st.rerun()
    
    # Search options
    col1, col2, col3 = st.columns(3)
    with col1:
        n_results = st.number_input("Max results", value=5, min_value=1, max_value=20)
    with col2:
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.1)
    with col3:
        use_spatial_filter = st.checkbox("Use spatial filter", value=False)
    
    # Spatial filter options
    bbox = None
    if use_spatial_filter:
        st.write("**Spatial Filter:**")
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Min Latitude", value=-90.0, max_value=90.0)
            max_lat = st.number_input("Max Latitude", value=90.0, max_value=90.0)
        with col2:
            min_lon = st.number_input("Min Longitude", value=-180.0, max_value=180.0)
            max_lon = st.number_input("Max Longitude", value=180.0, max_value=180.0)
        bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Execute search
    if st.button("🔍 Search", type="primary"):
        if query.strip():
            with st.spinner("Searching..."):
                try:
                    results = retrieval_engine.semantic_search(
                        query, 
                        n_results=n_results, 
                        bbox=bbox
                    )
                    
                    # Display results with filtering
                    display_filtered_results(results, query, similarity_threshold)
                    
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        else:
            st.warning("Please enter a search query")


def spatial_query_panel(spatial_engine):
    """Enhanced spatial query interface"""
    
    st.subheader("📍 Spatial Queries")
    
    # Query type
    spatial_type = st.selectbox(
        "Spatial Query Type",
        ["Range Query", "Point Query", "Bounding Box Query"],
        help="Choose your spatial query method"
    )
    
    if spatial_type == "Range Query":
        range_query_interface(spatial_engine)
    elif spatial_type == "Point Query":
        point_query_interface(spatial_engine)
    elif spatial_type == "Bounding Box Query":
        bbox_query_interface(spatial_engine)


def range_query_interface(spatial_engine):
    """Range query with map selection"""
    
    st.write("**Range Query:** Find features within a radius")
    
    # Map for point selection
    st.write("Click on the map to select a center point:")
    
    # Create a simple map for point selection
    m = folium.Map(location=[40.0, -100.0], zoom_start=4)
    
    # Add click handler
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(m, height=300, use_container_width=True)
    
    # Manual input as fallback
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input("Latitude", value=40.0, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=-100.0, format="%.6f")
    with col3:
        radius_km = st.number_input("Radius (km)", value=100.0, min_value=0.1)
    
    # Get point from map if available
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
    
    if st.button("🔍 Find Nearby Features", type="primary"):
        with st.spinner("Searching..."):
            try:
                results = spatial_engine.range_query(lat=lat, lon=lon, radius_km=radius_km)
                display_spatial_results(results, f"Features within {radius_km}km of ({lat:.3f}, {lon:.3f})")
            except Exception as e:
                st.error(f"Query error: {str(e)}")


def point_query_interface(spatial_engine):
    """Point query interface"""
    
    st.write("**Point Query:** Find features at a specific location")
    
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=40.7128, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=-74.0060, format="%.6f")
    
    tolerance = st.slider("Tolerance (km)", 0.1, 10.0, 1.0, 0.1)
    
    if st.button("🔍 Find Features at Point", type="primary"):
        with st.spinner("Searching..."):
            try:
                results = spatial_engine.point_query(lat=lat, lon=lon, tolerance_km=tolerance)
                display_spatial_results(results, f"Features at ({lat:.3f}, {lon:.3f}) ±{tolerance}km")
            except Exception as e:
                st.error(f"Query error: {str(e)}")


def bbox_query_interface(spatial_engine):
    """Bounding box query interface"""
    
    st.write("**Bounding Box Query:** Find features within a rectangular area")
    
    col1, col2 = st.columns(2)
    with col1:
        min_lat = st.number_input("Min Latitude", value=35.0, format="%.6f")
        max_lat = st.number_input("Max Latitude", value=45.0, format="%.6f")
    with col2:
        min_lon = st.number_input("Min Longitude", value=-80.0, format="%.6f")
        max_lon = st.number_input("Max Longitude", value=-70.0, format="%.6f")
    
    if st.button("🔍 Find Features in Bounding Box", type="primary"):
        with st.spinner("Searching..."):
            try:
                results = spatial_engine.bbox_query(
                    min_lat=min_lat,
                    min_lon=min_lon,
                    max_lat=max_lat,
                    max_lon=max_lon
                )
                display_spatial_results(results, "Bounding Box Query")
            except Exception as e:
                st.error(f"Query error: {str(e)}")


def hybrid_search_panel(retrieval_engine, spatial_engine):
    """Combined semantic and spatial search"""
    
    st.subheader("🔄 Hybrid Search")
    st.write("Combine semantic search with spatial constraints")
    
    # Semantic query
    query = st.text_input("Semantic query", value="Find major cities")
    
    # Spatial constraints
    st.write("**Spatial Constraints:**")
    constraint_type = st.selectbox("Constraint Type", ["None", "Bounding Box", "Radius"])
    
    bbox = None
    if constraint_type == "Bounding Box":
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Min Lat", value=35.0)
            max_lat = st.number_input("Max Lat", value=45.0)
        with col2:
            min_lon = st.number_input("Min Lon", value=-80.0)
            max_lon = st.number_input("Max Lon", value=-70.0)
        bbox = (min_lon, min_lat, max_lon, max_lat)
    
    elif constraint_type == "Radius":
        col1, col2, col3 = st.columns(3)
        with col1:
            center_lat = st.number_input("Center Lat", value=40.0)
        with col2:
            center_lon = st.number_input("Center Lon", value=-75.0)
        with col3:
            radius_km = st.number_input("Radius (km)", value=100.0)
    
    if st.button("🔄 Hybrid Search", type="primary"):
        with st.spinner("Performing hybrid search..."):
            try:
                # Perform semantic search with spatial filter
                results = retrieval_engine.semantic_search(query, n_results=10, bbox=bbox)
                display_filtered_results(results, f"Hybrid: {query}")
            except Exception as e:
                st.error(f"Hybrid search error: {str(e)}")


def query_examples_panel(retrieval_engine, spatial_engine):
    """Pre-built query examples"""
    
    st.subheader("📚 Query Examples")
    
    examples = {
        "Semantic Examples": [
            ("Find major cities", "semantic"),
            ("Locations near water", "semantic"),
            ("Urban areas", "semantic"),
            ("Capital cities", "semantic")
        ],
        "Spatial Examples": [
            ("Cities near New York", "spatial", {"lat": 40.7128, "lon": -74.0060, "radius": 200}),
            ("West Coast cities", "spatial", {"lat": 37.7749, "lon": -122.4194, "radius": 500}),
            ("East Coast cities", "spatial", {"lat": 38.9072, "lon": -77.0369, "radius": 500})
        ]
    }
    
    for category, example_list in examples.items():
        st.write(f"**{category}:**")
        for example in example_list:
            if len(example) == 2:
                query, query_type = example
                if st.button(f"🔍 {query}", key=f"ex_{query}"):
                    if query_type == "semantic":
                        results = retrieval_engine.semantic_search(query, n_results=5)
                        display_filtered_results(results, query)
            elif len(example) == 3:
                query, query_type, params = example
                if st.button(f"📍 {query}", key=f"ex_{query}"):
                    if query_type == "spatial":
                        results = spatial_engine.range_query(**params)
                        display_spatial_results(results, query)


def display_filtered_results(results: Dict[str, Any], query: str, threshold: float = 0.5):
    """Display search results with similarity filtering"""
    
    if not results or 'ids' not in results or not results['ids']:
        st.warning("No results found")
        return
    
    # Store results in session state for map display
    st.session_state.last_search_results = results
    
    # Filter by similarity threshold
    filtered_results = []
    for i, (doc_id, distance, metadata, document) in enumerate(zip(
        results['ids'][0], 
        results['distances'][0], 
        results['metadatas'][0], 
        results['documents'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity
        if similarity >= threshold:
            filtered_results.append({
                'id': doc_id,
                'similarity': similarity,
                'metadata': metadata,
                'document': document
            })
    
    if not filtered_results:
        st.warning(f"No results above similarity threshold ({threshold})")
        return
    
    # Display results in user-friendly format
    st.success(f"Found {len(filtered_results)} results for '{query}'")
    
    for i, result in enumerate(filtered_results):
        # Calculate similarity percentage
        similarity_pct = round(result['similarity'] * 100, 1)
        
        # Extract location information
        metadata = result['metadata']
        name = metadata.get("name", "Unknown Location")
        lat = metadata.get("latitude") or metadata.get("miny")
        lon = metadata.get("longitude") or metadata.get("minx")
        
        # Create a card for each result
        with st.container():
            st.markdown("---")
            
            # Header with rank and similarity
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### #{i+1} - {name}")
            with col2:
                # Color code similarity
                if similarity_pct >= 80:
                    st.success(f"**{similarity_pct}%** Match")
                elif similarity_pct >= 60:
                    st.warning(f"**{similarity_pct}%** Match")
                else:
                    st.error(f"**{similarity_pct}%** Match")
            
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
            if metadata:
                st.markdown("**Details:**")
                meta_display = {}
                
                # Filter and format metadata
                for key, value in metadata.items():
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
    st.success(f"✅ Found **{len(filtered_results)}** result(s) for your search")
    
    # Show raw data in expander for developers
    with st.expander("🔧 Raw Data (for developers)"):
        st.json(results)


def display_spatial_results(results: gpd.GeoDataFrame, query_description: str):
    """Display spatial query results in a user-friendly format"""
    
    if results is None or len(results) == 0:
        st.warning("No spatial results found")
        return
    
    # Store results in session state for map display
    st.session_state.last_spatial_results = results
    
    st.success(f"Found {len(results)} features: {query_description}")
    
    # Display results in a user-friendly format
    for idx, row in results.iterrows():
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
    st.success(f"✅ Found **{len(results)}** feature(s) in the spatial query")
    
    # Show raw data in expander for developers
    with st.expander("🔧 Raw GeoDataFrame (for developers)"):
        st.write(results)


# Import for map functionality
try:
    from streamlit_folium import st_folium
except ImportError:
    def st_folium(m, **kwargs):
        st.warning("streamlit-folium not available")
        return None
