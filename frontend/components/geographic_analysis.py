"""
Geographic Analysis Frontend Component

Provides a comprehensive interface for geographic analysis including:
- Spatial Statistics: Density analysis, clustering metrics
- Temporal Analysis: Time-series geographic data
- Elevation Processing: Terrain and elevation analysis
- Weather Integration: Climate and weather data overlay
- Population Density: Demographic analysis
- Transportation Networks: Route and accessibility analysis
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import folium
from streamlit_folium import folium_static
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis import AnalysisManager


class GeographicAnalysisInterface:
    """Frontend interface for geographic analysis."""
    
    def __init__(self):
        self.analysis_manager = AnalysisManager()
        
    def render_analysis_interface(self):
        """Render the main geographic analysis interface."""
        st.title("📈 Geographic Analysis")
        st.markdown("---")
        
        # Sidebar for analysis configuration
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
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Data Input")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Geographic Data (GeoJSON, Shapefile, CSV)",
                type=['geojson', 'shp', 'csv'],
                help="Upload your geographic data file"
            )
            
            # Sample data option
            use_sample_data = st.checkbox("Use Sample Data for Demonstration")
            
            if uploaded_file is not None:
                data = self._load_uploaded_data(uploaded_file)
                if data is not None:
                    st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                    # Convert GeoDataFrame to DataFrame for display (excluding geometry column)
                    if hasattr(data, 'geometry'):
                        display_data = data.drop(columns=['geometry'])
                        st.dataframe(display_data.head())
                    else:
                        st.dataframe(data.head())
            elif use_sample_data:
                data = self._create_sample_data()
                st.success("✅ Sample data created for demonstration!")
                # Convert GeoDataFrame to DataFrame for display (excluding geometry column)
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
                self._display_quick_stats(data)
        
        # Analysis execution
        if data is not None:
            st.markdown("---")
            st.subheader("🚀 Run Analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔍 Run Selected Analysis", type="primary"):
                    with st.spinner("Running analysis..."):
                        results = self._run_analysis(data, analysis_types)
                        st.session_state['analysis_results'] = results
                        st.success("✅ Analysis completed!")
            
            with col2:
                if st.button("📊 Generate Report"):
                    if 'analysis_results' in st.session_state:
                        report = self._generate_analysis_report(st.session_state['analysis_results'])
                        st.text_area("📋 Analysis Report", report, height=400)
                    else:
                        st.warning("⚠️ No analysis results available. Run analysis first.")
            
            with col3:
                if st.button("💾 Export Results"):
                    if 'analysis_results' in st.session_state:
                        self._export_results(st.session_state['analysis_results'])
                    else:
                        st.warning("⚠️ No analysis results available. Run analysis first.")
        
        # Results display
        if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            self._display_analysis_results(st.session_state['analysis_results'])
    
    def _load_uploaded_data(self, uploaded_file):
        """Load uploaded geographic data."""
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
    
    def _create_sample_data(self):
        """Create sample geographic data for demonstration."""
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
    
    def _display_quick_stats(self, data):
        """Display quick statistics about the data."""
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
    
    def _run_analysis(self, data, analysis_types):
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
        
        # Run analysis
        results = self.analysis_manager.run_comprehensive_analysis(data, internal_types)
        
        return results
    
    def _generate_analysis_report(self, results):
        """Generate a comprehensive analysis report."""
        return self.analysis_manager.get_analysis_report()
    
    def _export_results(self, results):
        """Export analysis results."""
        # Create download button for JSON results
        import json
        results_json = json.dumps(results, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name="geographic_analysis_results.json",
            mime="application/json"
        )
    
    def _display_analysis_results(self, results):
        """Display analysis results in an organized way."""
        # Create tabs for different analysis types
        if results:
            tab_names = list(results.keys())
            tabs = st.tabs([name.replace('_', ' ').title() for name in tab_names])
            
            for i, (analysis_name, analysis_results) in enumerate(results.items()):
                with tabs[i]:
                    self._display_specific_analysis(analysis_name, analysis_results)
    
    def _display_specific_analysis(self, analysis_name, analysis_results):
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
                        self._display_nested_results(value)
                elif isinstance(value, (list, tuple)):
                    with st.expander(f"📁 {key.replace('_', ' ').title()}"):
                        st.write(f"Number of items: {len(value)}")
                        if len(value) > 0 and isinstance(value[0], dict):
                            try:
                                st.dataframe(pd.DataFrame(value))
                            except Exception as e:
                                st.write("Data preview (table format not available):")
                                st.json(value[:5])  # Show first 5 items as JSON
    
    def _display_nested_results(self, nested_dict):
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


def main():
    """Main function to run the geographic analysis interface."""
    st.set_page_config(
        page_title="Geographic Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and render the interface
    analysis_interface = GeographicAnalysisInterface()
    analysis_interface.render_analysis_interface()


if __name__ == "__main__":
    main()
