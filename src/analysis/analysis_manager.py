"""
Analysis Manager Module

Coordinates and manages all geographic analysis components including:
- Spatial Statistics
- Temporal Analysis
- Elevation Processing
- Weather Integration
- Population Analysis
- Transportation Networks
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from .spatial_statistics import SpatialStatistics
from .temporal_analysis import TemporalAnalysis
from .elevation_processing import ElevationProcessor
from .weather_integration import WeatherIntegration
from .population_analysis import PopulationAnalysis


class AnalysisManager:
    """Main analysis manager for coordinating all geographic analysis components."""
    
    def __init__(self):
        """Initialize analysis manager with all analysis components."""
        self.spatial_stats = SpatialStatistics()
        self.temporal_analysis = TemporalAnalysis()
        self.elevation_processor = ElevationProcessor()
        self.weather_integration = WeatherIntegration()
        self.population_analysis = PopulationAnalysis()
        
        self.analysis_results = {}
        self.analysis_history = []
        
    def run_comprehensive_analysis(self, 
                                 gdf: gpd.GeoDataFrame,
                                 analysis_types: List[str] = None,
                                 **kwargs) -> Dict:
        """
        Run comprehensive geographic analysis.
        
        Args:
            gdf: GeoDataFrame with geographic data
            analysis_types: List of analysis types to run
            **kwargs: Additional parameters for specific analyses
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if analysis_types is None:
            analysis_types = ['spatial', 'temporal', 'elevation', 'weather', 'population']
        
        comprehensive_results = {}
        
        # Run spatial statistics analysis
        if 'spatial' in analysis_types:
            spatial_results = self._run_spatial_analysis(gdf, **kwargs)
            comprehensive_results['spatial_analysis'] = spatial_results
        
        # Run temporal analysis
        if 'temporal' in analysis_types:
            temporal_results = self._run_temporal_analysis(gdf, **kwargs)
            comprehensive_results['temporal_analysis'] = temporal_results
        
        # Run elevation analysis
        if 'elevation' in analysis_types:
            elevation_results = self._run_elevation_analysis(gdf, **kwargs)
            comprehensive_results['elevation_analysis'] = elevation_results
        
        # Run weather analysis
        if 'weather' in analysis_types:
            weather_results = self._run_weather_analysis(gdf, **kwargs)
            comprehensive_results['weather_analysis'] = weather_results
        
        # Run population analysis
        if 'population' in analysis_types:
            population_results = self._run_population_analysis(gdf, **kwargs)
            comprehensive_results['population_analysis'] = population_results
        
        # Store results
        self.analysis_results = comprehensive_results
        self.analysis_history.append({
            'timestamp': pd.Timestamp.now(),
            'analysis_types': analysis_types,
            'data_shape': gdf.shape,
            'results_summary': self._create_results_summary(comprehensive_results)
        })
        
        return comprehensive_results
    
    def _run_spatial_analysis(self, gdf: gpd.GeoDataFrame, **kwargs) -> Dict:
        """Run spatial statistics analysis."""
        spatial_results = {}
        
        try:
            # Point density analysis
            if all(isinstance(geom, Point) for geom in gdf.geometry):
                density_results = self.spatial_stats.point_density_analysis(gdf, **kwargs)
                spatial_results['density_analysis'] = density_results
            
            # Spatial clustering
            clustering_results = self.spatial_stats.spatial_clustering(gdf, **kwargs)
            spatial_results['clustering'] = clustering_results
            
            # Nearest neighbor analysis
            nn_results = self.spatial_stats.nearest_neighbor_analysis(gdf)
            spatial_results['nearest_neighbor'] = nn_results
            
            # Spatial autocorrelation (if numeric column provided)
            numeric_cols = gdf.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                autocorr_results = self.spatial_stats.spatial_autocorrelation(gdf, numeric_cols[0], **kwargs)
                spatial_results['spatial_autocorrelation'] = autocorr_results
            
        except Exception as e:
            spatial_results['error'] = f"Spatial analysis failed: {str(e)}"
        
        return spatial_results
    
    def _run_temporal_analysis(self, gdf: gpd.GeoDataFrame, **kwargs) -> Dict:
        """Run temporal analysis."""
        temporal_results = {}
        
        try:
            # Check for temporal columns
            temporal_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                             for keyword in ['date', 'time', 'year', 'month', 'day'])]
            
            if temporal_cols and len(temporal_cols) > 0:
                time_col = temporal_cols[0]
                numeric_cols = gdf.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    value_col = numeric_cols[0]
                    
                    # Time series analysis
                    ts_results = self.temporal_analysis.time_series_analysis(gdf, time_col, value_col, **kwargs)
                    temporal_results['time_series'] = ts_results
                    
                    # Temporal clustering
                    tc_results = self.temporal_analysis.temporal_clustering(gdf, time_col, value_col, **kwargs)
                    temporal_results['temporal_clustering'] = tc_results
                    
                    # Temporal autocorrelation
                    ta_results = self.temporal_analysis.temporal_autocorrelation(gdf, time_col, value_col, **kwargs)
                    temporal_results['temporal_autocorrelation'] = ta_results
            else:
                temporal_results['error'] = "No temporal columns found for analysis"
                
        except Exception as e:
            temporal_results['error'] = f"Temporal analysis failed: {str(e)}"
        
        return temporal_results
    
    def _run_elevation_analysis(self, gdf: gpd.GeoDataFrame, **kwargs) -> Dict:
        """Run elevation analysis."""
        elevation_results = {}
        
        try:
            # Check if we have elevation data
            elevation_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                              for keyword in ['elevation', 'height', 'altitude', 'dem'])]
            
            if elevation_cols:
                elevation_col = elevation_cols[0]
                elevation_data = gdf[elevation_col].values
                
                # Process DEM
                dem_results = self.elevation_processor.process_dem(elevation_data, **kwargs)
                elevation_results['dem_processing'] = dem_results
                
                # Calculate slope and aspect
                slope_results = self.elevation_processor.calculate_slope_aspect(elevation_data, **kwargs)
                elevation_results['slope_aspect'] = slope_results
                
                # Terrain roughness
                roughness_results = self.elevation_processor.calculate_terrain_roughness(elevation_data, **kwargs)
                elevation_results['terrain_roughness'] = roughness_results
            else:
                elevation_results['error'] = "No elevation data found for analysis"
                
        except Exception as e:
            elevation_results['error'] = f"Elevation analysis failed: {str(e)}"
        
        return elevation_results
    
    def _run_weather_analysis(self, gdf: gpd.GeoDataFrame, **kwargs) -> Dict:
        """Run weather analysis."""
        weather_results = {}
        
        try:
            # Check for weather data
            weather_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                            for keyword in ['temperature', 'precipitation', 'wind', 'humidity', 'pressure'])]
            
            if weather_cols:
                # Create sample weather data for demonstration
                weather_data = self._create_sample_weather_data(gdf)
                
                # Climate pattern analysis
                climate_results = self.weather_integration.analyze_climate_patterns(weather_data, **kwargs)
                weather_results['climate_patterns'] = climate_results
                
                # Temperature analysis
                temp_results = self.weather_integration.analyze_temperature(weather_data, **kwargs)
                weather_results['temperature_analysis'] = temp_results
                
                # Precipitation analysis
                precip_results = self.weather_integration.analyze_precipitation(weather_data, **kwargs)
                weather_results['precipitation_analysis'] = precip_results
            else:
                weather_results['error'] = "No weather data found for analysis"
                
        except Exception as e:
            weather_results['error'] = f"Weather analysis failed: {str(e)}"
        
        return weather_results
    
    def _run_population_analysis(self, gdf: gpd.GeoDataFrame, **kwargs) -> Dict:
        """Run population analysis."""
        population_results = {}
        
        try:
            # Check for population data
            population_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                               for keyword in ['population', 'people', 'residents', 'inhabitants'])]
            
            if population_cols:
                population_col = population_cols[0]
                
                # Population density analysis
                density_results = self.population_analysis.analyze_population_density(gdf, population_col, **kwargs)
                population_results['population_density'] = density_results
                
                # Demographic composition analysis
                demo_results = self.population_analysis.analyze_demographic_composition(gdf, **kwargs)
                population_results['demographic_composition'] = demo_results
                
                # Age structure analysis
                age_cols = [col for col in gdf.columns if 'age' in col.lower()]
                if age_cols:
                    age_results = self.population_analysis.analyze_age_structure(gdf, age_cols[0], population_col, **kwargs)
                    population_results['age_structure'] = age_results
            else:
                population_results['error'] = "No population data found for analysis"
                
        except Exception as e:
            population_results['error'] = f"Population analysis failed: {str(e)}"
        
        return population_results
    
    def _create_sample_weather_data(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create sample weather data for demonstration."""
        # Generate sample weather data based on geographic extent
        bounds = gdf.total_bounds
        
        # Create grid of weather stations
        n_stations = 10
        lats = np.linspace(bounds[1], bounds[3], n_stations)
        lons = np.linspace(bounds[0], bounds[2], n_stations)
        
        weather_data = []
        for lat in lats:
            for lon in lons:
                for day in range(1, 32):  # 31 days
                    weather_data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': pd.Timestamp(2024, 1, day),
                        'temperature': 20 + 10 * np.sin(2 * np.pi * day / 31) + np.random.normal(0, 2),
                        'precipitation': np.random.exponential(5),
                        'wind_speed': np.random.exponential(3),
                        'wind_direction': np.random.uniform(0, 360)
                    })
        
        return pd.DataFrame(weather_data)
    
    def _create_results_summary(self, results: Dict) -> Dict:
        """Create a summary of analysis results."""
        summary = {
            'total_analyses': len(results),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'analysis_types': list(results.keys())
        }
        
        for analysis_type, analysis_results in results.items():
            if isinstance(analysis_results, dict) and 'error' not in analysis_results:
                summary['successful_analyses'] += 1
            else:
                summary['failed_analyses'] += 1
        
        return summary
    
    def get_analysis_report(self, analysis_type: str = None) -> str:
        """Generate analysis report."""
        if analysis_type is None:
            # Generate comprehensive report
            report = "=== COMPREHENSIVE GEOGRAPHIC ANALYSIS REPORT ===\n\n"
            
            for analysis_name, results in self.analysis_results.items():
                report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key != 'error':
                            if isinstance(value, dict):
                                report += f"{key}:\n"
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, float):
                                        report += f"  {sub_key}: {sub_value:.4f}\n"
                                    else:
                                        report += f"  {sub_key}: {sub_value}\n"
                            else:
                                report += f"{key}: {value}\n"
                        else:
                            report += f"ERROR: {value}\n"
                report += "\n"
            
            # Add analysis history
            if self.analysis_history:
                report += "--- ANALYSIS HISTORY ---\n"
                for history in self.analysis_history[-5:]:  # Last 5 analyses
                    report += f"Timestamp: {history['timestamp']}\n"
                    report += f"Analysis Types: {', '.join(history['analysis_types'])}\n"
                    report += f"Data Shape: {history['data_shape']}\n"
                    report += f"Results Summary: {history['results_summary']}\n\n"
        else:
            # Generate specific analysis report
            if analysis_type in self.analysis_results:
                if analysis_type == 'spatial_analysis':
                    report = self.spatial_stats.generate_report()
                elif analysis_type == 'temporal_analysis':
                    report = self.temporal_analysis.generate_report()
                elif analysis_type == 'elevation_analysis':
                    report = self.elevation_processor.generate_report()
                elif analysis_type == 'weather_analysis':
                    report = self.weather_integration.generate_report()
                elif analysis_type == 'population_analysis':
                    report = self.population_analysis.generate_report()
                else:
                    report = f"No specific report generator for {analysis_type}"
            else:
                report = f"No results found for {analysis_type}"
        
        return report
    
    def export_results(self, filepath: str, format: str = 'json') -> bool:
        """Export analysis results to file."""
        try:
            if format.lower() == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.analysis_results, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Export summary statistics to CSV
                summary_data = []
                for analysis_type, results in self.analysis_results.items():
                    if isinstance(results, dict):
                        for key, value in results.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, (int, float, str)):
                                        summary_data.append({
                                            'analysis_type': analysis_type,
                                            'metric': f"{key}_{sub_key}",
                                            'value': sub_value
                                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(filepath, index=False)
            else:
                return False
            
            return True
        except Exception as e:
            print(f"Export failed: {str(e)}")
            return False
    
    def get_analysis_status(self) -> Dict:
        """Get current analysis status."""
        return {
            'total_analyses_run': len(self.analysis_history),
            'last_analysis': self.analysis_history[-1] if self.analysis_history else None,
            'current_results': len(self.analysis_results),
            'available_analyses': [
                'spatial_analysis',
                'temporal_analysis', 
                'elevation_analysis',
                'weather_analysis',
                'population_analysis'
            ]
        }
    
    def clear_results(self):
        """Clear all analysis results."""
        self.analysis_results = {}
        self.analysis_history = []
        
        # Clear individual component results
        self.spatial_stats.results = {}
        self.temporal_analysis.results = {}
        self.elevation_processor.results = {}
        self.weather_integration.results = {}
        self.population_analysis.results = {}
