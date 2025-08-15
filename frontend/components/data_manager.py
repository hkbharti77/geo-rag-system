import streamlit as st
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import json

from config.settings import SAMPLE_DATA_DIR, RAW_DIR
from src.utils.file_handlers import read_vector_file


def data_upload_section() -> Optional[gpd.GeoDataFrame]:
    """Enhanced data upload with preview and validation"""
    
    st.sidebar.header("📁 Data Management")
    
    # File upload with better UX
    uploaded_file = st.sidebar.file_uploader(
        "Upload Geographic Data", 
        type=["geojson", "json", "shp", "csv", "kml", "gpkg"],
        help="Supported: GeoJSON, Shapefile, CSV (with lat/lon), KML, GeoPackage"
    )
    
    gdf = None
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            upload_path = RAW_DIR / uploaded_file.name
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load and validate
            gdf = read_vector_file(upload_path)
            
            # Show success with details
            st.sidebar.success(f"✅ **{uploaded_file.name}** loaded successfully!")
            
            # Data preview
            with st.sidebar.expander("📊 Data Preview", expanded=True):
                st.write(f"**Features:** {len(gdf)}")
                st.write(f"**Geometry Type:** {gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else 'None'}")
                st.write(f"**CRS:** {gdf.crs}")
                
                # Show sample data
                if len(gdf) > 0:
                    sample_df = gdf.head(3).drop(columns=['geometry'])
                    st.dataframe(sample_df, use_container_width=True)
                    
                    # Show bounding box
                    bounds = gdf.total_bounds
                    st.write(f"**Bounds:** {bounds[0]:.3f}, {bounds[1]:.3f} to {bounds[2]:.3f}, {bounds[3]:.3f}")
            
            return gdf
            
        except Exception as e:
            st.sidebar.error(f"❌ **Error loading file:** {str(e)}")
            st.sidebar.info("💡 **Need help?** Try our sample data below")
            return None
    
    # Sample data section
    st.sidebar.subheader("🎯 Quick Start")
    
    sample_files = {
        "US Cities (20)": "locations.geojson",
        "World Cities (3)": "cities.geojson", 
        "US Cities CSV": "locations.csv"
    }
    
    selected_sample = st.sidebar.selectbox(
        "Try sample data:",
        ["None"] + list(sample_files.keys())
    )
    
    if selected_sample != "None":
        sample_file = sample_files[selected_sample]
        sample_path = SAMPLE_DATA_DIR / sample_file
        
        if sample_path.exists():
            try:
                gdf = read_vector_file(sample_path)
                st.sidebar.success(f"✅ Loaded {len(gdf)} features from {sample_file}")
                
                # Show sample info
                with st.sidebar.expander("📊 Sample Data Info"):
                    st.write(f"**Features:** {len(gdf)}")
                    if 'name' in gdf.columns:
                        names = gdf['name'].tolist()[:5]
                        st.write(f"**Sample names:** {', '.join(names)}")
                
                return gdf
                
            except Exception as e:
                st.sidebar.error(f"Error loading sample: {str(e)}")
    
    return None


def data_validation_panel(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Data validation and quality checks"""
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    if gdf is None or len(gdf) == 0:
        validation_results["valid"] = False
        validation_results["errors"].append("No data loaded")
        return validation_results
    
    # Check for required columns
    if 'name' not in gdf.columns:
        validation_results["warnings"].append("No 'name' column found - using generic names")
    
    # Check for valid geometries
    invalid_geoms = gdf[~gdf.geometry.is_valid]
    if len(invalid_geoms) > 0:
        validation_results["warnings"].append(f"{len(invalid_geoms)} invalid geometries found")
    
    # Check CRS
    if gdf.crs is None:
        validation_results["warnings"].append("No CRS specified - assuming WGS84")
    elif str(gdf.crs) != "EPSG:4326":
        validation_results["warnings"].append(f"CRS is {gdf.crs} - consider reprojecting to WGS84")
    
    # Check for duplicates
    if len(gdf) != len(gdf.drop_duplicates()):
        validation_results["warnings"].append("Duplicate features detected")
    
    return validation_results


def show_workflow_guide():
    """Show step-by-step workflow guide"""
    
    st.sidebar.subheader("🛠️ Workflow Guide")
    
    with st.sidebar.expander("📋 How to use this system"):
        st.markdown("""
        **1. 📁 Upload Data**
        - Upload your geographic file
        - Or use sample data to get started
        
        **2. 🔍 Index Data**
        - Click "Index Current Data" to make it searchable
        
        **3. 🔎 Query & Explore**
        - Use semantic search for natural language queries
        - Use spatial queries for location-based searches
        - View results on the interactive map
        """)
    

