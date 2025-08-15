#!/usr/bin/env python3
"""
Geographic Analysis Demonstration Script

This script demonstrates the comprehensive geographic analysis capabilities
including spatial statistics, temporal analysis, elevation processing,
weather integration, population analysis, and transportation networks.
"""

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis import AnalysisManager


def create_demo_data():
    """Create comprehensive demo data for analysis."""
    print("📊 Creating demo data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample point data
    n_points = 200
    
    # Generate coordinates in a realistic area (New York City region)
    lats = np.random.uniform(40.5, 41.0, n_points)
    lons = np.random.uniform(-74.2, -73.7, n_points)
    
    # Create temporal data
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    
    # Create comprehensive attributes
    data = {
        'id': range(n_points),
        'latitude': lats,
        'longitude': lons,
        'population': np.random.poisson(2000, n_points),
        'elevation': np.random.normal(50, 30, n_points),
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 365) + np.random.normal(0, 3, n_points),
        'precipitation': np.random.exponential(8, n_points),
        'wind_speed': np.random.exponential(4, n_points),
        'wind_direction': np.random.uniform(0, 360, n_points),
        'timestamp': dates,
        'age_group': np.random.choice(['0-14', '15-24', '25-64', '65+'], n_points),
        'income_level': np.random.choice(['low', 'medium', 'high'], n_points),
        'education_level': np.random.choice(['primary', 'secondary', 'tertiary'], n_points)
    }
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs="EPSG:4326"
    )
    
    print(f"✅ Created demo data with {len(gdf)} points")
    return gdf


def demonstrate_spatial_analysis(analysis_manager, gdf):
    """Demonstrate spatial analysis capabilities."""
    print("\n🔍 Demonstrating Spatial Analysis...")
    
    # Run spatial analysis
    spatial_results = analysis_manager.spatial_stats.point_density_analysis(gdf, cell_size=1000)
    print(f"   📍 Point density analysis completed")
    
    clustering_results = analysis_manager.spatial_stats.spatial_clustering(gdf, method='kmeans', n_clusters=5)
    print(f"   🎯 Spatial clustering completed ({clustering_results['n_clusters']} clusters)")
    
    nn_results = analysis_manager.spatial_stats.nearest_neighbor_analysis(gdf)
    print(f"   📏 Nearest neighbor analysis completed")
    print(f"      Distribution: {nn_results['interpretation']}")
    
    # Spatial autocorrelation
    autocorr_results = analysis_manager.spatial_stats.spatial_autocorrelation(gdf, 'population')
    print(f"   🔗 Spatial autocorrelation completed")
    print(f"      Moran's I: {autocorr_results['morans_i']:.4f}")
    print(f"      Interpretation: {autocorr_results['interpretation']}")


def demonstrate_temporal_analysis(analysis_manager, gdf):
    """Demonstrate temporal analysis capabilities."""
    print("\n⏰ Demonstrating Temporal Analysis...")
    
    # Run temporal analysis
    ts_results = analysis_manager.temporal_analysis.time_series_analysis(
        gdf, 'timestamp', 'temperature'
    )
    print(f"   📈 Time series analysis completed")
    print(f"      Duration: {ts_results['time_stats']['duration_days']} days")
    print(f"      Trend: {ts_results['time_trends']['trend_interpretation']}")
    
    # Temporal clustering
    tc_results = analysis_manager.temporal_analysis.temporal_clustering(
        gdf, 'timestamp', 'temperature', n_clusters=3
    )
    print(f"   🕒 Temporal clustering completed ({tc_results['n_clusters']} clusters)")
    
    # Temporal autocorrelation
    ta_results = analysis_manager.temporal_analysis.temporal_autocorrelation(
        gdf, 'timestamp', 'temperature', max_lag=10
    )
    print(f"   🔄 Temporal autocorrelation completed")
    print(f"      Significant lags: {len(ta_results['significant_lags'])}")


def demonstrate_elevation_analysis(analysis_manager, gdf):
    """Demonstrate elevation analysis capabilities."""
    print("\n🏔️ Demonstrating Elevation Analysis...")
    
    # Create elevation data (2D array simulation)
    elevation_data = np.random.normal(100, 50, (20, 20))
    
    # Process DEM
    dem_results = analysis_manager.elevation_processor.process_dem(elevation_data)
    print(f"   📊 DEM processing completed")
    print(f"      Elevation range: {dem_results['min_elevation']:.1f} - {dem_results['max_elevation']:.1f} m")
    
    # Calculate slope and aspect
    slope_results = analysis_manager.elevation_processor.calculate_slope_aspect(elevation_data)
    print(f"   📐 Slope and aspect calculation completed")
    print(f"      Mean slope: {slope_results['slope_statistics']['mean_slope']:.2f}°")
    
    # Terrain roughness
    roughness_results = analysis_manager.elevation_processor.calculate_terrain_roughness(elevation_data)
    print(f"   🌄 Terrain roughness analysis completed")
    print(f"      Mean TRI: {roughness_results['roughness_statistics']['tri']['mean']:.4f}")


def demonstrate_weather_analysis(analysis_manager, gdf):
    """Demonstrate weather analysis capabilities."""
    print("\n🌤️ Demonstrating Weather Analysis...")
    
    # Create weather data
    weather_data = analysis_manager._create_sample_weather_data(gdf)
    
    # Climate pattern analysis
    climate_results = analysis_manager.weather_integration.analyze_climate_patterns(weather_data)
    print(f"   🌍 Climate pattern analysis completed")
    print(f"      Total observations: {climate_results['data_summary']['total_observations']}")
    
    # Temperature analysis
    temp_results = analysis_manager.weather_integration.analyze_temperature(weather_data)
    print(f"   🌡️ Temperature analysis completed")
    print(f"      Mean temperature: {temp_results['basic_statistics']['mean_temperature']:.2f}°C")
    
    # Precipitation analysis
    precip_results = analysis_manager.weather_integration.analyze_precipitation(weather_data)
    print(f"   🌧️ Precipitation analysis completed")
    print(f"      Total precipitation: {precip_results['basic_statistics']['total_precipitation']:.2f} mm")


def demonstrate_population_analysis(analysis_manager, gdf):
    """Demonstrate population analysis capabilities."""
    print("\n👥 Demonstrating Population Analysis...")
    
    # Population density analysis
    density_results = analysis_manager.population_analysis.analyze_population_density(gdf, 'population')
    print(f"   📊 Population density analysis completed")
    print(f"      Total population: {density_results['basic_statistics']['total_population']:,}")
    
    # Demographic composition analysis
    demo_results = analysis_manager.population_analysis.analyze_demographic_composition(gdf)
    print(f"   👨‍👩‍👧‍👦 Demographic composition analysis completed")
    print(f"      Demographic variables: {len(demo_results['demographic_analysis'])}")
    
    # Age structure analysis
    age_results = analysis_manager.population_analysis.analyze_age_structure(gdf, 'age_group', 'population')
    print(f"   📈 Age structure analysis completed")
    print(f"      Mean age: {age_results['age_statistics']['mean_age']:.1f} years")


def demonstrate_comprehensive_analysis(analysis_manager, gdf):
    """Demonstrate comprehensive analysis capabilities."""
    print("\n🚀 Demonstrating Comprehensive Analysis...")
    
    # Run comprehensive analysis
    comprehensive_results = analysis_manager.run_comprehensive_analysis(
        gdf, 
        analysis_types=['spatial', 'temporal', 'elevation', 'weather', 'population']
    )
    
    print(f"   ✅ Comprehensive analysis completed")
    print(f"      Analysis types: {len(comprehensive_results)}")
    
    # Generate report
    report = analysis_manager.get_analysis_report()
    print(f"   📋 Generated comprehensive report ({len(report)} characters)")
    
    return comprehensive_results


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("📈 GEOGRAPHIC ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create demo data
    gdf = create_demo_data()
    
    # Initialize analysis manager
    analysis_manager = AnalysisManager()
    
    # Demonstrate individual analysis types
    demonstrate_spatial_analysis(analysis_manager, gdf)
    demonstrate_temporal_analysis(analysis_manager, gdf)
    demonstrate_elevation_analysis(analysis_manager, gdf)
    demonstrate_weather_analysis(analysis_manager, gdf)
    demonstrate_population_analysis(analysis_manager, gdf)
    
    # Demonstrate comprehensive analysis
    comprehensive_results = demonstrate_comprehensive_analysis(analysis_manager, gdf)
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    status = analysis_manager.get_analysis_status()
    print(f"Total analyses run: {status['total_analyses_run']}")
    print(f"Current results: {status['current_results']}")
    print(f"Available analyses: {', '.join(status['available_analyses'])}")
    
    # Export results
    print("\n💾 Exporting results...")
    success = analysis_manager.export_results("demo_analysis_results.json", "json")
    if success:
        print("   ✅ Results exported to demo_analysis_results.json")
    else:
        print("   ❌ Export failed")
    
    print("\n🎉 Demonstration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
