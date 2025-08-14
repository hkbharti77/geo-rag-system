import json
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd

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
                    # Initialize engines
                    emb_manager = EmbeddingManager()
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
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "🔍 Search & Query", "🗂️ Chunking", "📈 Analysis"])
    
    with tab1:
        # Enhanced map interface
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Interactive Geographic Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Map controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_chunks = st.checkbox("Show Chunks", value=False)
        with col2:
            show_search_results = st.checkbox("Show Search Results", value=True)
        with col3:
            show_spatial_results = st.checkbox("Show Spatial Results", value=True)
        
        # Create and display map
        m = create_map_interface(
            gdf=st.session_state.current_data,
            search_results=st.session_state.search_results if show_search_results else None,
            spatial_results=st.session_state.spatial_results if show_spatial_results else None,
            show_chunks=show_chunks
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
            # Initialize engines for search
            emb_manager = EmbeddingManager()
            text_store = VectorStore(persist_directory=CHROMA_DIR, collection_name="geographic_features")
            retrieval = RetrievalEngine(text_store=text_store, image_store=None, embeddings=emb_manager)
            spatial_engine = SpatialQueryEngine(gdf=st.session_state.current_data)
            
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
        # Chunking interface
        st.markdown("""
        <div class="feature-card">
            <h3>🗂️ Geographic Chunking Strategies</h3>
        </div>
        """, unsafe_allow_html=True)
        
        chunking_controls(st.session_state.current_data)
    
    with tab4:
        # Analysis tab (placeholder for future features)
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Geographic Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("🚧 Analysis features coming soon!")
        st.markdown("""
        **Planned Features:**
        - **Spatial Statistics**: Density analysis, clustering metrics
        - **Temporal Analysis**: Time-series geographic data
        - **Elevation Processing**: Terrain and elevation analysis
        - **Weather Integration**: Climate and weather data overlay
        - **Population Density**: Demographic analysis
        - **Transportation Networks**: Route and accessibility analysis
        """)

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
        3. **Create Chunks**: Divide your area into manageable regions
        4. **Search & Explore**: Use semantic and spatial queries
        5. **Visualize**: View results on the interactive map
        
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
