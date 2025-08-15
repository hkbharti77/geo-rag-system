import json
from pathlib import Path
import streamlit as st
# Ensure project root is on sys.path for absolute imports like `config.settings`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import st_folium with a safe fallback to avoid hard dependency at app startup
try:
    from streamlit_folium import st_folium
except Exception:
    def st_folium(*args, **kwargs):
        st.warning("streamlit-folium is not available; interactive map rendering is disabled.")
        return None
import folium
import geopandas as gpd
import pandas as pd

from config.settings import PROJECT_ROOT, SAMPLE_DATA_DIR, CHROMA_DIR, RAW_DIR
from src.utils.file_handlers import read_vector_file
from src.rag.embedding_manager import EmbeddingManager
from src.rag.vector_store import VectorStore
from src.rag.retrieval_engine import RetrievalEngine
from src.query.spatial_query_engine import SpatialQueryEngine
from src.query.result_formatter import to_table
from src.data_processing.spatial_indexing import GridSpatialIndex

# Import new components
from frontend.components.data_manager import data_upload_section, data_validation_panel, show_workflow_guide
from frontend.components.query_interface import chunking_controls
from frontend.components.advanced_queries import advanced_search_interface
from frontend.components.enhanced_map import create_map_interface, display_map_with_controls
from frontend.components.results_display import display_search_results, display_spatial_results

# Import geographic analysis functionality
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from analysis import AnalysisManager

# Helper functions for geographic analysis (defined early to avoid NameError)
def _load_uploaded_data_for_analysis(uploaded_file):
    """Load uploaded geographic data for analysis."""
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            # Try to create geometry from lat/lon columns
            if 'latitude' in data.columns and 'longitude' in data.columns:
                data['geometry'] = gpd.points_from_xy(data['longitude'], data['latitude'])
                data = gpd.GeoDataFrame(data, crs="EPSG:4326")
        elif uploaded_file.name.endswith('.geojson'):
            data = gpd.read_file(uploaded_file)
        elif uploaded_file.name.endswith('.shp'):
            data = gpd.read_file(uploaded_file)
        else:
            st.error("❌ Unsupported file format")
            return None
        
        return data
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def _create_sample_data_for_analysis():
    """Create sample geographic data for analysis demonstration."""
    import numpy as np
    import pandas as pd
    
    # Create sample point data
    np.random.seed(42)
    n_points = 100
    
    # Generate random coordinates
    lats = np.random.uniform(40.0, 42.0, n_points)
    lons = np.random.uniform(-74.0, -72.0, n_points)
    
    # Create sample attributes
    data = {
        'id': range(n_points),
        'latitude': lats,
        'longitude': lons,
        'population': np.random.poisson(1000, n_points),
        'elevation': np.random.normal(100, 50, n_points),
        'temperature': np.random.normal(20, 5, n_points),
        'precipitation': np.random.exponential(10, n_points),
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='D')
    }
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs="EPSG:4326"
    )
    
    return gdf

def _display_quick_stats_for_analysis(data):
    """Display quick statistics about the data for analysis."""
    stats = {
        "Total Records": len(data),
        "Geometry Type": str(data.geometry.geom_type.iloc[0]) if hasattr(data, 'geometry') else "N/A",
        "CRS": str(data.crs) if hasattr(data, 'crs') else "N/A",
        "Columns": len(data.columns)
    }
    
    for key, value in stats.items():
        st.metric(key, value)
    
    # Display data bounds
    if hasattr(data, 'total_bounds'):
        bounds = data.total_bounds
        st.markdown("**Spatial Extent:**")
        st.write(f"Min: ({bounds[0]:.4f}, {bounds[1]:.4f})")
        st.write(f"Max: ({bounds[2]:.4f}, {bounds[3]:.4f})")

def _run_analysis(data, analysis_types):
    """Run the selected analysis types."""
    # Convert analysis type names to internal names
    type_mapping = {
        "Spatial Statistics": "spatial",
        "Temporal Analysis": "temporal",
        "Elevation Processing": "elevation",
        "Weather Integration": "weather",
        "Population Density": "population",
        "Transportation Networks": "transportation"
    }
    
    internal_types = [type_mapping[atype] for atype in analysis_types if atype in type_mapping]
    
    # Run analysis using the analysis manager
    results = st.session_state.analysis_manager.run_comprehensive_analysis(data, internal_types)
    
    return results

def _export_analysis_results(results):
    """Export analysis results."""
    # Create download button for JSON results
    results_json = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Results (JSON)",
        data=results_json,
        file_name="geographic_analysis_results.json",
        mime="application/json"
    )

def _display_analysis_results(results):
    """Display analysis results in an organized way."""
    # Create tabs for different analysis types
    if results:
        tab_names = list(results.keys())
        tabs = st.tabs([name.replace('_', ' ').title() for name in tab_names])
        
        for i, (analysis_name, analysis_results) in enumerate(results.items()):
            with tabs[i]:
                _display_specific_analysis(analysis_name, analysis_results)

def _display_specific_analysis(analysis_name, analysis_results):
    """Display results for a specific analysis type."""
    if isinstance(analysis_results, dict):
        if 'error' in analysis_results:
            st.error(f"❌ {analysis_results['error']}")
            return
        
        # Display key metrics
        st.subheader("📊 Key Metrics")
        
        # Create columns for metrics
        cols = st.columns(3)
        col_idx = 0
        
        for key, value in analysis_results.items():
            if isinstance(value, dict) and key not in ['error']:
                # Display nested metrics
                with cols[col_idx % 3]:
                    st.metric(key.replace('_', ' ').title(), f"{len(value)} sub-metrics")
                col_idx += 1
            elif isinstance(value, (int, float)):
                with cols[col_idx % 3]:
                    st.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else value)
                col_idx += 1
        
        # Display detailed results
        st.subheader("📋 Detailed Results")
        
        for key, value in analysis_results.items():
            if isinstance(value, dict) and key not in ['error']:
                with st.expander(f"📁 {key.replace('_', ' ').title()}"):
                    _display_nested_results(value)
            elif isinstance(value, (list, tuple)):
                with st.expander(f"📁 {key.replace('_', ' ').title()}"):
                    st.write(f"Number of items: {len(value)}")
                    if len(value) > 0 and isinstance(value[0], dict):
                        try:
                            st.dataframe(pd.DataFrame(value))
                        except Exception as e:
                            st.write("Data preview (table format not available):")
                            st.json(value[:5])  # Show first 5 items as JSON

def _display_nested_results(nested_dict):
    """Display nested dictionary results."""
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            st.markdown(f"**{key.replace('_', ' ').title()}:**")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    st.write(f"  {sub_key}: {sub_value:.4f}" if isinstance(sub_value, float) else f"  {sub_key}: {sub_value}")
                else:
                    st.write(f"  {sub_key}: {sub_value}")
        elif isinstance(value, (int, float)):
            st.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else value)
        else:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")


# Page configuration
st.set_page_config(
    page_title="Geo RAG System - Intelligent Geographic Analysis",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🗺️ Geographic Information RAG System</h1>
    <p>Intelligent Spatial Analysis with Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'spatial_results' not in st.session_state:
    st.session_state.spatial_results = None
if 'data_indexed' not in st.session_state:
    st.session_state.data_indexed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_manager' not in st.session_state:
    st.session_state.analysis_manager = AnalysisManager()

# Sidebar with enhanced data management
with st.sidebar:
    # Data upload and management
    gdf = data_upload_section()
    
    if gdf is not None:
        st.session_state.current_data = gdf
        
        # Data validation
        validation = data_validation_panel(gdf)
        
        if validation['warnings']:
            with st.expander("⚠️ Data Warnings", expanded=True):
                for warning in validation['warnings']:
                    st.warning(warning)
        
        if validation['errors']:
            with st.expander("❌ Data Errors", expanded=True):
                for error in validation['errors']:
                    st.error(error)
    
    # Indexing section
    st.sidebar.subheader("🔍 Indexing")
    
    if st.session_state.current_data is not None and len(st.session_state.current_data) > 0:
        if st.sidebar.button("📊 Index Current Data", type="primary"):
            with st.spinner("Indexing features..."):
                try:
                    # Initialize engines with error handling
                    try:
                        emb_manager = EmbeddingManager()
                    except Exception as emb_error:
                        error_msg = str(emb_error)
                        if "Network error" in error_msg or "MaxRetryError" in error_msg or "getaddrinfo failed" in error_msg:
                            st.sidebar.error("❌ Network connectivity issue")
                            st.sidebar.error(f"Details: {error_msg}")
                            st.sidebar.info("💡 Solutions:")
                            st.sidebar.info("1. Check your internet connection")
                            st.sidebar.info("2. Try using a VPN if you're behind a firewall")
                            st.sidebar.info("3. The app will use a simple fallback mode")
                            st.sidebar.info("4. Restart the app to try again")
                        else:
                            st.sidebar.error(f"❌ Failed to initialize embedding model: {error_msg}")
                            st.sidebar.info("💡 Try restarting the app or check your PyTorch installation")
                        st.stop()
                    
                    text_store = VectorStore(persist_directory=CHROMA_DIR, collection_name="geographic_features")
                    
                    # Prepare data for indexing
                    gdf = st.session_state.current_data.reset_index(drop=True)
                    idx = GridSpatialIndex()
                    entries = idx.build_index(gdf)
                    
                    texts = []
                    metas = []
                    docs = []
                    ids = []
                    
                    for i, entry in enumerate(entries):
                        props = entry.get("properties", {})
                        name = str(props.get("name", f"feature_{i}"))
                        text = name
                        texts.append(text)
                        docs.append(f"Feature: {name}")
                        metas.append(GridSpatialIndex.to_metadata(entry))
                        ids.append(f"feat-{i}")
                    
                    # Create embeddings and add to store
                    embs = emb_manager.embed_texts(texts)
                    text_store.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
                    
                    st.session_state.data_indexed = True
                    st.sidebar.success(f"✅ Indexed {len(ids)} features successfully!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Indexing failed: {str(e)}")
        
        # Show indexing status
        if st.session_state.data_indexed:
            st.sidebar.success("✅ Data is indexed and ready for search")
        else:
            st.sidebar.info("ℹ️ Click 'Index Current Data' to enable search")
    
    # Workflow guide
    show_workflow_guide()

# Main content area
if st.session_state.current_data is not None and len(st.session_state.current_data) > 0:
    
    # Data overview card
    with st.container():
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Features", len(st.session_state.current_data))
        with col2:
            geom_type = st.session_state.current_data.geometry.geom_type.iloc[0] if len(st.session_state.current_data) > 0 else "None"
            st.metric("Geometry Type", geom_type)
        with col3:
            crs = str(st.session_state.current_data.crs) if st.session_state.current_data.crs else "None"
            st.metric("CRS", crs)
        with col4:
            status = "✅ Indexed" if st.session_state.data_indexed else "⏳ Not Indexed"
            st.metric("Search Status", status)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "🔍 Search & Query", "📈 Analysis"])
    
    with tab1:
        # Enhanced map interface
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Interactive Geographic Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Map controls
        col1, col2 = st.columns(2)
        with col1:
            show_search_results = st.checkbox("Show Search Results", value=True)
        with col2:
            show_spatial_results = st.checkbox("Show Spatial Results", value=True)
        
        # Create and display map
        m = create_map_interface(
            gdf=st.session_state.current_data,
            search_results=st.session_state.search_results if show_search_results else None,
            spatial_results=st.session_state.spatial_results if show_spatial_results else None
        )
        
        # Display map with controls
        map_data = display_map_with_controls(m, height=600)
        
        # Map interaction info
        with st.expander("🗺️ Map Features"):
            st.markdown("""
            **Interactive Features:**
            - **Multiple Tile Layers**: Street, Satellite, and Topographic views
            - **Layer Controls**: Toggle different data layers on/off
            - **Interactive Popups**: Click on features for detailed information
            - **Zoom & Pan**: Navigate the map with mouse controls
            - **Search Results**: Visualize search results with different colors
            - **Chunk Boundaries**: View geographic chunking regions
            """)
    
    with tab2:
        # Advanced search interface
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Advanced Geographic Search</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.data_indexed:
            # Initialize engines for search with error handling
            try:
                emb_manager = EmbeddingManager()
                text_store = VectorStore(persist_directory=CHROMA_DIR, collection_name="geographic_features")
                retrieval = RetrievalEngine(text_store=text_store, image_store=None, embeddings=emb_manager)
                spatial_engine = SpatialQueryEngine(gdf=st.session_state.current_data)
            except Exception as e:
                error_msg = str(e)
                if "Network error" in error_msg or "MaxRetryError" in error_msg or "getaddrinfo failed" in error_msg:
                    st.error("❌ Network connectivity issue")
                    st.error(f"Details: {error_msg}")
                    st.info("💡 Solutions:")
                    st.info("1. Check your internet connection")
                    st.info("2. Try using a VPN if you're behind a firewall")
                    st.info("3. The app will use a simple fallback mode")
                    st.info("4. Restart the app to try again")
                else:
                    st.error(f"❌ Failed to initialize search engines: {error_msg}")
                    st.info("💡 Try restarting the app or check your PyTorch installation")
                st.stop()
            
            # Advanced search interface
            advanced_search_interface(retrieval, spatial_engine)
            
            # Store results in session state for map display
            if 'last_search_results' in st.session_state:
                st.session_state.search_results = st.session_state.last_search_results
            if 'last_spatial_results' in st.session_state:
                st.session_state.spatial_results = st.session_state.last_spatial_results
                
        else:
            st.warning("⚠️ Please index your data first to enable search functionality")
            st.info("Go to the sidebar and click 'Index Current Data'")
    
    with tab3:
        # Geographic Analysis Interface
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Geographic Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis configuration sidebar
        with st.sidebar:
            st.header("🔧 Analysis Configuration")
            
            # Analysis type selection
            analysis_types = st.multiselect(
                "Select Analysis Types",
                options=[
                    "Spatial Statistics",
                    "Temporal Analysis", 
                    "Elevation Processing",
                    "Weather Integration",
                    "Population Density",
                    "Transportation Networks"
                ],
                default=["Spatial Statistics"]
            )
            
            # Analysis parameters
            st.subheader("📊 Analysis Parameters")
            
            # Spatial analysis parameters
            if "Spatial Statistics" in analysis_types:
                st.markdown("**Spatial Statistics Parameters:**")
                cell_size = st.slider("Cell Size (meters)", 100, 5000, 1000)
                clustering_method = st.selectbox("Clustering Method", ["kmeans", "dbscan"])
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            
            # Temporal analysis parameters
            if "Temporal Analysis" in analysis_types:
                st.markdown("**Temporal Analysis Parameters:**")
                analysis_period = st.selectbox("Analysis Period", ["daily", "weekly", "monthly", "seasonal"])
                max_lag = st.slider("Maximum Lag", 5, 20, 10)
            
            # Elevation analysis parameters
            if "Elevation Processing" in analysis_types:
                st.markdown("**Elevation Processing Parameters:**")
                dem_cell_size = st.slider("DEM Cell Size (meters)", 10, 100, 30)
                roughness_window = st.slider("Roughness Window Size", 3, 11, 5)
            
            # Weather analysis parameters
            if "Weather Integration" in analysis_types:
                st.markdown("**Weather Integration Parameters:**")
                weather_distance = st.slider("Weather Distance Threshold (meters)", 500, 5000, 1000)
            
            # Population analysis parameters
            if "Population Density" in analysis_types:
                st.markdown("**Population Analysis Parameters:**")
                pop_cell_size = st.slider("Population Cell Size (meters)", 500, 2000, 1000)
            
            # Transportation analysis parameters
            if "Transportation Networks" in analysis_types:
                st.markdown("**Transportation Analysis Parameters:**")
                service_radius = st.slider("Service Radius (meters)", 500, 5000, 1000)
        
        # Main analysis content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Data Input")
            
            # Use current data or allow new upload
            if st.session_state.current_data is not None:
                st.success(f"✅ Using current data! Shape: {st.session_state.current_data.shape}")
                data = st.session_state.current_data
                # Convert GeoDataFrame to DataFrame for display (excluding geometry column)
                if hasattr(data, 'geometry'):
                    display_data = data.drop(columns=['geometry'])
                    st.dataframe(display_data.head())
                else:
                    st.dataframe(data.head())
            else:
                # File upload for analysis
                uploaded_file = st.file_uploader(
                    "Upload Geographic Data for Analysis (GeoJSON, Shapefile, CSV)",
                    type=['geojson', 'shp', 'csv'],
                    help="Upload your geographic data file for analysis"
                )
                
                # Sample data option
                use_sample_data = st.checkbox("Use Sample Data for Demonstration")
                
                if uploaded_file is not None:
                    data = _load_uploaded_data_for_analysis(uploaded_file)
                    if data is not None:
                        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                        if hasattr(data, 'geometry'):
                            display_data = data.drop(columns=['geometry'])
                            st.dataframe(display_data.head())
                        else:
                            st.dataframe(data.head())
                elif use_sample_data:
                    data = _create_sample_data_for_analysis()
                    st.success("✅ Sample data created for demonstration!")
                    if hasattr(data, 'geometry'):
                        display_data = data.drop(columns=['geometry'])
                        st.dataframe(display_data.head())
                    else:
                        st.dataframe(data.head())
                else:
                    data = None
                    st.info("📁 Please upload data or select sample data to begin analysis")
        
        with col2:
            st.subheader("📈 Quick Stats")
            if data is not None:
                _display_quick_stats_for_analysis(data)
        
        # Analysis execution
        if data is not None:
            st.markdown("---")
            st.subheader("🚀 Run Analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔍 Run Selected Analysis", type="primary"):
                    with st.spinner("Running analysis..."):
                        results = _run_analysis(data, analysis_types)
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis completed!")
            
            with col2:
                if st.button("📊 Generate Report"):
                    if st.session_state.analysis_results:
                        report = st.session_state.analysis_manager.get_analysis_report()
                        st.text_area("📋 Analysis Report", report, height=400)
                    else:
                        st.warning("⚠️ No analysis results available. Run analysis first.")
            
            with col3:
                if st.button("💾 Export Results"):
                    if st.session_state.analysis_results:
                        _export_analysis_results(st.session_state.analysis_results)
                    else:
                        st.warning("⚠️ No analysis results available. Run analysis first.")
        
        # Results display
        if st.session_state.analysis_results:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            _display_analysis_results(st.session_state.analysis_results)

else:
    # Welcome screen when no data is loaded
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Welcome to the Geographic RAG System</h3>
        <p>Get started by uploading your geographic data or using our sample datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Upload Your Data
        **Supported Formats:**
        - **GeoJSON** (.geojson, .json)
        - **Shapefile** (.shp with .shx, .dbf)
        - **CSV** with latitude/longitude columns
        - **KML** (.kml)
        - **GeoPackage** (.gpkg)
        
        **Data Requirements:**
        - Geographic coordinates (WGS84 recommended)
        - Feature names or descriptions
        - Valid geometries
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. **Upload Data**: Use the sidebar to upload your file
        2. **Index Data**: Click "Index Current Data" to make it searchable
        3. **Search & Explore**: Use semantic and spatial queries
        4. **Visualize**: View results on the interactive map
        
        **Sample Data Available:**
        - US Cities (20 locations)
        - World Cities (3 locations)
        - US Cities CSV format
        """)
    
    # Show system capabilities
    st.markdown("""
    <div class="feature-card">
        <h3>🔧 System Capabilities</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🗺️ Spatial Operations**
        - Point-in-polygon queries
        - Range searches (radius-based)
        - Bounding box operations
        - Distance calculations
        - Spatial indexing (Grid-based)
        """)
    
    with col2:
        st.markdown("""
        **🔍 RAG Features**
        - Semantic search with embeddings
        - Natural language queries
        - Hybrid spatial-semantic search
        - Vector similarity matching
        - Context-aware retrieval
        """)
    
    with col3:
        st.markdown("""
        **📊 Data Processing**
        - Multi-format support
        - Automatic CRS handling
        - Data validation
        - Chunking strategies
        - Metadata extraction
        """)



# Footer
st.markdown("---")
st.caption("""
**Geo RAG System** - Intelligent Geographic Information System with Retrieval-Augmented Generation
Built with Streamlit, GeoPandas, ChromaDB, and Sentence Transformers
""")
