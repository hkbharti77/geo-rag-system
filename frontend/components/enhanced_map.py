import streamlit as st
import folium
import geopandas as gpd
import json
from typing import Dict, Any, List, Optional
import pandas as pd


def create_enhanced_map(gdf: gpd.GeoDataFrame, 
                       center: Optional[List[float]] = None,
                       zoom_start: int = 4,
                       height: int = 600) -> folium.Map:
    """Create an enhanced interactive map"""
    
    # Default center if not provided
    if center is None:
        if len(gdf) > 0:
            # Calculate center from data
            cent = gdf.geometry.union_all().centroid if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union.centroid
            center = [cent.y, cent.x]
        else:
            center = [20.0, 0.0]
    
    # Create base map with better styling
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add multiple tile layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Topographic',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False
    ).add_to(m)
    
    return m


def add_data_to_map(m: folium.Map, 
                   gdf: gpd.GeoDataFrame, 
                   layer_name: str = "Data",
                   color: str = "red",
                   radius: int = 8) -> folium.Map:
    """Add GeoDataFrame data to map with styling"""
    
    if len(gdf) == 0:
        return m
    
    # Convert to GeoJSON
    geojson_data = json.loads(gdf.to_json())
    
    # Create feature group for this layer
    fg = folium.FeatureGroup(name=layer_name)
    
    # Add features with popups
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        name = props.get('name', 'Unknown')
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>{name}</h4>
            <p><strong>Type:</strong> {feature['geometry']['type']}</p>
        """
        
        # Add other properties to popup
        for key, value in props.items():
            if key != 'name':
                popup_content += f"<p><strong>{key}:</strong> {value}</p>"
        
        popup_content += "</div>"
        
        # Add to map based on geometry type
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=radius,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(fg)
        else:
            # For other geometry types, use GeoJson
            folium.GeoJson(
                feature,
                popup=folium.Popup(popup_content, max_width=300),
                style_function=lambda x: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.3
                }
            ).add_to(fg)
    
    fg.add_to(m)
    return m


def add_search_results_to_map(m: folium.Map, 
                             results: Dict[str, Any], 
                             layer_name: str = "Search Results",
                             color: str = "blue") -> folium.Map:
    """Add search results to map"""
    
    if not results or 'metadatas' not in results or not results['metadatas']:
        return m
    
    fg = folium.FeatureGroup(name=layer_name)
    
    for i, (metadata, distance, document) in enumerate(zip(
        results['metadatas'][0], 
        results['distances'][0], 
        results['documents'][0]
    )):
        if metadata and 'centroid_lat' in metadata and 'centroid_lon' in metadata:
            lat = metadata['centroid_lat']
            lon = metadata['centroid_lon']
            similarity = 1 - distance
            
            popup_content = f"""
            <div style="width: 200px;">
                <h4>Search Result #{i+1}</h4>
                <p><strong>Document:</strong> {document}</p>
                <p><strong>Similarity:</strong> {similarity:.3f}</p>
                <p><strong>Location:</strong> {lat:.3f}, {lon:.3f}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=3
            ).add_to(fg)
    
    fg.add_to(m)
    return m


def add_spatial_results_to_map(m: folium.Map, 
                              results: gpd.GeoDataFrame, 
                              layer_name: str = "Spatial Results",
                              color: str = "green") -> folium.Map:
    """Add spatial query results to map"""
    
    if results is None or len(results) == 0:
        return m
    
    fg = folium.FeatureGroup(name=layer_name)
    
    # Convert to GeoJSON and add with styling
    geojson_data = json.loads(results.to_json())
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        name = props.get('name', 'Unknown')
        
        popup_content = f"""
        <div style="width: 200px;">
            <h4>Spatial Result</h4>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Type:</strong> {feature['geometry']['type']}</p>
        </div>
        """
        
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=12,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=3
            ).add_to(fg)
        else:
            folium.GeoJson(
                feature,
                popup=folium.Popup(popup_content, max_width=300),
                style_function=lambda x: {
                    'fillColor': color,
                    'color': color,
                    'weight': 3,
                    'fillOpacity': 0.4
                }
            ).add_to(fg)
    
    fg.add_to(m)
    return m


def add_chunks_to_map(m: folium.Map, 
                     chunks_file: str, 
                     layer_name: str = "Chunks",
                     color: str = "orange") -> folium.Map:
    """Add chunk boundaries to map"""
    
    try:
        from pathlib import Path
        from config.settings import CHUNKS_DIR
        
        chunk_path = CHUNKS_DIR / chunks_file
        if chunk_path.exists():
            chunks_gdf = gpd.read_file(chunk_path)
            
            fg = folium.FeatureGroup(name=layer_name)
            
            geojson_data = json.loads(chunks_gdf.to_json())
            
            for feature in geojson_data['features']:
                props = feature.get('properties', {})
                chunk_id = props.get('chunk_id', props.get('cluster_id', 'Unknown'))
                
                popup_content = f"""
                <div style="width: 200px;">
                    <h4>Chunk: {chunk_id}</h4>
                    <p><strong>Type:</strong> {feature['geometry']['type']}</p>
                </div>
                """
                
                folium.GeoJson(
                    feature,
                    popup=folium.Popup(popup_content, max_width=300),
                    style_function=lambda x: {
                        'fillColor': color,
                        'color': color,
                        'weight': 2,
                        'fillOpacity': 0.1,
                        'dashArray': '5, 5'
                    }
                ).add_to(fg)
            
            fg.add_to(m)
            
    except Exception as e:
        st.warning(f"Could not load chunks: {str(e)}")
    
    return m


def create_map_interface(gdf: gpd.GeoDataFrame,
                        search_results: Optional[Dict[str, Any]] = None,
                        spatial_results: Optional[gpd.GeoDataFrame] = None,
                        show_chunks: bool = False) -> folium.Map:
    """Create a comprehensive map interface"""
    
    st.subheader("🗺️ Interactive Map")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    with col1:
        map_height = st.slider("Map Height", 400, 800, 600, 50)
    with col2:
        show_data = st.checkbox("Show Data", value=True)
    with col3:
        show_results = st.checkbox("Show Results", value=True)
    
    # Calculate center
    center = None
    if len(gdf) > 0:
        cent = gdf.geometry.union_all().centroid if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union.centroid
        center = [cent.y, cent.x]
    
    # Create map
    m = create_enhanced_map(gdf, center=center, height=map_height)
    
    # Add data layer
    if show_data and len(gdf) > 0:
        m = add_data_to_map(m, gdf, "Loaded Data", "red", 8)
    
    # Add search results
    if show_results and search_results:
        m = add_search_results_to_map(m, search_results, "Search Results", "blue")
    
    # Add spatial results
    if show_results and spatial_results is not None:
        m = add_spatial_results_to_map(m, spatial_results, "Spatial Results", "green")
    
    # Add chunks if requested
    if show_chunks:
        # Get available chunk files
        from pathlib import Path
        from config.settings import CHUNKS_DIR
        
        chunk_files = list(CHUNKS_DIR.glob("*.geojson"))
        if chunk_files:
            selected_chunk = st.selectbox(
                "Select chunk file to display:",
                [f.name for f in chunk_files]
            )
            m = add_chunks_to_map(m, selected_chunk, "Chunks", "orange")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def display_map_with_controls(m: folium.Map, height: int = 600):
    """Display map with additional controls"""
    
    # Map display
    map_data = st_folium(m, height=height, use_container_width=True)
    
    # Map interaction info
    with st.expander("🗺️ Map Controls"):
        st.markdown("""
        **Map Features:**
        - **Layers**: Toggle different data layers on/off
        - **Zoom**: Use mouse wheel or +/- buttons
        - **Pan**: Click and drag to move around
        - **Popups**: Click on features for details
        - **Fullscreen**: Click the fullscreen button
        
        **Tile Layers:**
        - **Street Map**: Standard OpenStreetMap
        - **Satellite**: High-resolution satellite imagery
        - **Topographic**: Terrain and elevation data
        """)
    
    return map_data


# Import for map functionality
try:
    from streamlit_folium import st_folium
except ImportError:
    def st_folium(m, **kwargs):
        st.warning("streamlit-folium not available")
        return None
