# 📈 Geographic Analysis Features

The Geo-RAG System now includes comprehensive geographic analysis capabilities that provide advanced spatial, temporal, and demographic insights. This module offers a complete suite of analysis tools for geographic data processing and interpretation.

## 🚀 Features Overview

### 1. Spatial Statistics
- **Point Density Analysis**: Kernel and grid-based density estimation
- **Spatial Clustering**: K-means and DBSCAN clustering algorithms
- **Hot Spot Analysis**: Getis-Ord Gi* statistic for identifying clusters
- **Spatial Autocorrelation**: Moran's I analysis for spatial patterns
- **Nearest Neighbor Analysis**: Point pattern analysis and distribution testing

### 2. Temporal Analysis
- **Time Series Analysis**: Comprehensive temporal pattern recognition
- **Seasonal Decomposition**: Trend, seasonal, and residual component analysis
- **Change Detection**: Automatic identification of temporal change points
- **Temporal Clustering**: Grouping similar temporal patterns
- **Temporal Autocorrelation**: Lag-based correlation analysis

### 3. Elevation Processing
- **DEM Processing**: Digital Elevation Model analysis and statistics
- **Slope & Aspect Calculation**: Terrain gradient and orientation analysis
- **Terrain Roughness**: Multiple roughness indices (TRI, VRM)
- **Viewshed Analysis**: Line-of-sight and visibility calculations
- **Watershed Delineation**: Catchment area and flow path analysis

### 4. Weather Integration
- **Weather Data Overlay**: Spatial integration of climate data
- **Climate Pattern Analysis**: Seasonal and long-term climate trends
- **Precipitation Analysis**: Rainfall patterns and extreme event detection
- **Temperature Analysis**: Thermal patterns and heat wave identification
- **Wind Pattern Analysis**: Wind speed and direction analysis

### 5. Population Analysis
- **Population Density**: Spatial distribution and density mapping
- **Demographic Composition**: Age, gender, and socioeconomic analysis
- **Population Growth**: Temporal population change analysis
- **Age Structure Analysis**: Population pyramid and dependency ratios
- **Migration Patterns**: Origin-destination flow analysis

### 6. Transportation Networks
- **Network Connectivity**: Graph theory-based network analysis
- **Route Optimization**: Shortest path and accessibility calculations
- **Service Area Analysis**: Coverage and accessibility mapping
- **Travel Time Analysis**: Network performance and congestion analysis
- **Network Centrality**: Key node and vulnerability identification

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Additional Dependencies
The geographic analysis module requires these additional packages:
- `scipy>=1.11.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `seaborn>=0.12.0` - Statistical visualization
- `networkx>=3.0` - Network analysis

## 📖 Usage Examples

### Basic Usage

```python
from src.analysis import AnalysisManager
import geopandas as gpd

# Initialize analysis manager
analysis_manager = AnalysisManager()

# Load your geographic data
gdf = gpd.read_file("your_data.geojson")

# Run comprehensive analysis
results = analysis_manager.run_comprehensive_analysis(
    gdf, 
    analysis_types=['spatial', 'temporal', 'elevation', 'weather', 'population']
)

# Generate report
report = analysis_manager.get_analysis_report()
print(report)
```

### Individual Analysis Types

```python
# Spatial Statistics
spatial_results = analysis_manager.spatial_stats.point_density_analysis(gdf, cell_size=1000)
clustering_results = analysis_manager.spatial_stats.spatial_clustering(gdf, method='kmeans', n_clusters=5)

# Temporal Analysis
temporal_results = analysis_manager.temporal_analysis.time_series_analysis(gdf, 'timestamp', 'value')

# Elevation Processing
elevation_data = np.random.normal(100, 50, (20, 20))  # Your DEM data
dem_results = analysis_manager.elevation_processor.process_dem(elevation_data)

# Weather Integration
weather_data = pd.DataFrame(...)  # Your weather data
weather_results = analysis_manager.weather_integration.analyze_climate_patterns(weather_data)

# Population Analysis
population_results = analysis_manager.population_analysis.analyze_population_density(gdf, 'population')

# Transportation Networks
network_results = analysis_manager.transportation_networks.analyze_network_connectivity(network_gdf)
```

## 🎯 Analysis Parameters

### Spatial Statistics Parameters
- `cell_size`: Grid cell size in meters (default: 1000)
- `clustering_method`: 'kmeans' or 'dbscan' (default: 'kmeans')
- `n_clusters`: Number of clusters for K-means (default: 5)
- `distance_threshold`: Distance for spatial autocorrelation (default: 1000)

### Temporal Analysis Parameters
- `analysis_period`: 'daily', 'weekly', 'monthly', 'seasonal' (default: 'monthly')
- `max_lag`: Maximum lag for autocorrelation (default: 10)
- `n_clusters`: Number of temporal clusters (default: 3)

### Elevation Processing Parameters
- `cell_size`: DEM cell size in meters (default: 30)
- `window_size`: Analysis window size (default: 3)
- `max_distance`: Viewshed analysis distance (default: 5000)

### Weather Integration Parameters
- `distance_threshold`: Weather data interpolation distance (default: 1000)
- `analysis_period`: Climate analysis period (default: 'monthly')

### Population Analysis Parameters
- `cell_size`: Population density cell size (default: 1000)
- `demographic_cols`: List of demographic column names

### Transportation Networks Parameters
- `service_radius`: Service area radius in meters (default: 1000)
- `optimization_criteria`: 'shortest_distance', 'fastest_time', 'least_cost'

## 📊 Output Formats

### Analysis Results Structure
```python
{
    'spatial_analysis': {
        'density_analysis': {...},
        'clustering': {...},
        'nearest_neighbor': {...},
        'spatial_autocorrelation': {...}
    },
    'temporal_analysis': {
        'time_series': {...},
        'temporal_clustering': {...},
        'temporal_autocorrelation': {...}
    },
    'elevation_analysis': {
        'dem_processing': {...},
        'slope_aspect': {...},
        'terrain_roughness': {...}
    },
    'weather_analysis': {
        'climate_patterns': {...},
        'temperature_analysis': {...},
        'precipitation_analysis': {...}
    },
    'population_analysis': {
        'population_density': {...},
        'demographic_composition': {...},
        'age_structure': {...}
    }
}
```

### Report Generation
```python
# Generate comprehensive report
report = analysis_manager.get_analysis_report()

# Generate specific analysis report
spatial_report = analysis_manager.spatial_stats.generate_report()
temporal_report = analysis_manager.temporal_analysis.generate_report()
```

### Export Options
```python
# Export to JSON
analysis_manager.export_results("results.json", format="json")

# Export to CSV
analysis_manager.export_results("results.csv", format="csv")
```

## 🎨 Frontend Interface

The geographic analysis module includes a comprehensive Streamlit frontend interface:

```bash
# Run the frontend
streamlit run frontend/components/geographic_analysis.py
```

### Frontend Features
- **Interactive Analysis Selection**: Choose which analyses to run
- **Parameter Configuration**: Adjust analysis parameters via sliders and dropdowns
- **Data Upload**: Support for GeoJSON, Shapefile, and CSV formats
- **Real-time Results**: Live display of analysis results
- **Export Functionality**: Download results in multiple formats
- **Visualization**: Interactive maps and charts

## 🧪 Demonstration

Run the demonstration script to see all features in action:

```bash
python demo_geographic_analysis.py
```

This script will:
1. Create sample geographic data
2. Run all analysis types
3. Display results and statistics
4. Export results to JSON format

## 📈 Performance Considerations

### Memory Usage
- Large datasets (>100,000 points) may require significant memory
- Consider chunking data for very large analyses
- Use appropriate cell sizes for density analysis

### Processing Time
- Spatial clustering: O(n²) for DBSCAN, O(nk) for K-means
- Temporal analysis: O(n log n) for sorting
- Elevation processing: O(n²) for viewshed analysis
- Weather integration: O(nm) where n=points, m=weather stations

### Optimization Tips
- Use appropriate spatial indexing for large datasets
- Consider parallel processing for independent analyses
- Cache intermediate results for repeated analyses

## 🔧 Configuration

### Analysis Manager Configuration
```python
# Initialize with custom settings
analysis_manager = AnalysisManager()

# Configure analysis parameters
analysis_manager.spatial_stats.results = {}  # Clear previous results
analysis_manager.temporal_analysis.results = {}

# Get analysis status
status = analysis_manager.get_analysis_status()
print(f"Available analyses: {status['available_analyses']}")
```

### Error Handling
```python
try:
    results = analysis_manager.run_comprehensive_analysis(gdf)
except Exception as e:
    print(f"Analysis failed: {str(e)}")
    # Check individual component results for specific errors
```

## 📚 Advanced Usage

### Custom Analysis Workflows
```python
# Custom spatial analysis workflow
spatial_stats = analysis_manager.spatial_stats

# Step 1: Density analysis
density_results = spatial_stats.point_density_analysis(gdf, cell_size=500)

# Step 2: Clustering based on density results
clustering_results = spatial_stats.spatial_clustering(gdf, method='dbscan', eps=1000)

# Step 3: Hot spot analysis
hotspot_results = spatial_stats.hot_spot_analysis(gdf, 'population', distance_threshold=2000)

# Step 4: Generate custom report
custom_report = f"""
Spatial Analysis Report:
- Density: {density_results['mean_density']:.2f}
- Clusters: {clustering_results['n_clusters']}
- Hot spots: {hotspot_results['n_hot_spots']}
"""
```

### Integration with External Data
```python
# Integrate with external weather APIs
import requests

def get_weather_data(lat, lon):
    # Example weather API call
    url = f"https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={lat},{lon}"
    response = requests.get(url)
    return response.json()

# Use in analysis
weather_data = []
for idx, row in gdf.iterrows():
    weather = get_weather_data(row.geometry.y, row.geometry.x)
    weather_data.append(weather)

weather_df = pd.DataFrame(weather_data)
weather_results = analysis_manager.weather_integration.analyze_climate_patterns(weather_df)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce cell size for density analysis
   - Use smaller datasets for testing
   - Enable garbage collection

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify package versions

3. **Analysis Failures**
   - Check data quality and completeness
   - Verify coordinate reference systems
   - Ensure sufficient data points for analysis

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
results = analysis_manager.run_comprehensive_analysis(gdf)
```

## 🤝 Contributing

To contribute to the geographic analysis module:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation
5. Ensure backward compatibility

## 📄 License

This geographic analysis module is part of the Geo-RAG System and follows the same licensing terms.

## 📞 Support

For support and questions about the geographic analysis features:

1. Check the documentation
2. Review the demonstration script
3. Examine the test cases
4. Open an issue on the project repository

---

**Note**: This geographic analysis module is designed to be extensible and can be easily integrated with additional analysis types and external data sources as needed.
