import streamlit as st
import geopandas as gpd
from typing import Dict, Any, Optional, Tuple
import folium


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
                # This would need to be implemented in spatial_engine
                st.info("Bounding box query not yet implemented")
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
    
    # Display results
    st.success(f"Found {len(filtered_results)} results for '{query}'")
    
    for i, result in enumerate(filtered_results):
        with st.expander(f"#{i+1} - {result['document']} (Similarity: {result['similarity']:.3f})"):
            st.write(f"**Document:** {result['document']}")
            st.write(f"**Similarity:** {result['similarity']:.3f}")
            if result['metadata']:
                st.write("**Metadata:**")
                st.json(result['metadata'])


def display_spatial_results(results: gpd.GeoDataFrame, query_description: str):
    """Display spatial query results"""
    
    if results is None or len(results) == 0:
        st.warning("No spatial results found")
        return
    
    # Store results in session state for map display
    st.session_state.last_spatial_results = results
    
    st.success(f"Found {len(results)} features: {query_description}")
    
    # Show results table
    if 'name' in results.columns:
        display_df = results[['name']].copy()
        if 'geometry' in results.columns:
            # Add centroid coordinates
            centroids = results.geometry.centroid
            display_df['latitude'] = centroids.y
            display_df['longitude'] = centroids.x
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(results.drop(columns=['geometry']), use_container_width=True)


# Import for map functionality
try:
    from streamlit_folium import st_folium
except ImportError:
    def st_folium(m, **kwargs):
        st.warning("streamlit-folium not available")
        return None
