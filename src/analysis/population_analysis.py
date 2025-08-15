"""
Population Analysis Module

Provides population and demographic analysis capabilities including:
- Population density analysis
- Demographic composition analysis
- Population growth analysis
- Age structure analysis
- Migration pattern analysis
- Socioeconomic indicators
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class PopulationAnalysis:
    """Population and demographic analysis tools."""
    
    def __init__(self):
        self.results = {}
        
    def analyze_population_density(self, 
                                 gdf: gpd.GeoDataFrame,
                                 population_col: str = 'population',
                                 area_col: str = 'area',
                                 cell_size: float = 1000) -> Dict:
        """
        Analyze population density patterns.
        
        Args:
            gdf: GeoDataFrame with population data
            population_col: Column name for population values
            area_col: Column name for area values (optional)
            cell_size: Size of grid cells in meters for density calculation
            
        Returns:
            Dictionary with population density analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if population_col not in gdf.columns:
            return {"error": f"Population column '{population_col}' not found"}
            
        # Basic population statistics
        population_values = gdf[population_col].dropna()
        if len(population_values) == 0:
            return {"error": "No valid population data found"}
            
        basic_stats = {
            'total_population': population_values.sum(),
            'mean_population': population_values.mean(),
            'std_population': population_values.std(),
            'min_population': population_values.min(),
            'max_population': population_values.max(),
            'median_population': population_values.median(),
            'n_areas': len(population_values)
        }
        
        # Calculate population density if area column is available
        density_stats = {}
        if area_col in gdf.columns:
            area_values = gdf[area_col].dropna()
            if len(area_values) > 0:
                # Calculate density for each area
                valid_mask = gdf[population_col].notna() & gdf[area_col].notna()
                valid_data = gdf[valid_mask]
                
                if len(valid_data) > 0:
                    densities = valid_data[population_col] / valid_data[area_col]
                    
                    density_stats = {
                        'mean_density': densities.mean(),
                        'std_density': densities.std(),
                        'min_density': densities.min(),
                        'max_density': densities.max(),
                        'median_density': densities.median(),
                        'density_percentiles': densities.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
                    }
        
        # Create population density grid
        density_grid = self._create_population_density_grid(gdf, population_col, cell_size)
        
        # Population distribution analysis
        distribution_analysis = self._analyze_population_distribution(population_values)
        
        # Spatial clustering of population
        spatial_clustering = self._analyze_population_clustering(gdf, population_col)
        
        density_results = {
            'basic_statistics': basic_stats,
            'density_statistics': density_stats,
            'density_grid': density_grid,
            'distribution_analysis': distribution_analysis,
            'spatial_clustering': spatial_clustering
        }
        
        self.results['population_density'] = density_results
        return density_results
    
    def analyze_demographic_composition(self, 
                                      gdf: gpd.GeoDataFrame,
                                      demographic_cols: List[str] = None) -> Dict:
        """
        Analyze demographic composition.
        
        Args:
            gdf: GeoDataFrame with demographic data
            demographic_cols: List of demographic column names
            
        Returns:
            Dictionary with demographic composition analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if demographic_cols is None:
            # Try to identify demographic columns automatically
            demographic_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                                for keyword in ['age', 'gender', 'race', 'ethnicity', 'income', 'education'])]
        
        if not demographic_cols:
            return {"error": "No demographic columns found"}
            
        demographic_analysis = {}
        
        for col in demographic_cols:
            if col in gdf.columns:
                values = gdf[col].dropna()
                if len(values) > 0:
                    # Basic statistics
                    stats_dict = {
                        'count': len(values),
                        'mean': values.mean() if values.dtype in ['float64', 'int64'] else None,
                        'std': values.std() if values.dtype in ['float64', 'int64'] else None,
                        'min': values.min() if values.dtype in ['float64', 'int64'] else None,
                        'max': values.max() if values.dtype in ['float64', 'int64'] else None,
                        'unique_values': values.nunique(),
                        'most_common': values.mode().iloc[0] if len(values.mode()) > 0 else None
                    }
                    
                    # Value distribution for categorical data
                    if values.dtype == 'object' or values.nunique() < 20:
                        value_counts = values.value_counts()
                        stats_dict['value_distribution'] = value_counts.to_dict()
                        stats_dict['value_percentages'] = (value_counts / len(values) * 100).to_dict()
                    
                    demographic_analysis[col] = stats_dict
        
        # Cross-demographic analysis
        cross_analysis = self._analyze_cross_demographics(gdf, demographic_cols)
        
        composition_results = {
            'demographic_analysis': demographic_analysis,
            'cross_demographic_analysis': cross_analysis,
            'demographic_columns': demographic_cols
        }
        
        self.results['demographic_composition'] = composition_results
        return composition_results
    
    def analyze_population_growth(self, 
                                gdf: gpd.GeoDataFrame,
                                population_col: str = 'population',
                                time_col: str = 'year',
                                location_col: str = None) -> Dict:
        """
        Analyze population growth patterns.
        
        Args:
            gdf: GeoDataFrame with population data over time
            population_col: Column name for population values
            time_col: Column name for time period
            location_col: Column name for location identifier (optional)
            
        Returns:
            Dictionary with population growth analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if population_col not in gdf.columns or time_col not in gdf.columns:
            return {"error": "Population or time column not found"}
            
        # Ensure time column is sorted
        gdf = gdf.copy()
        gdf = gdf.sort_values(time_col)
        
        # Calculate growth rates
        growth_analysis = self._calculate_growth_rates(gdf, population_col, time_col)
        
        # Analyze growth trends
        trend_analysis = self._analyze_growth_trends(gdf, population_col, time_col)
        
        # Location-based growth analysis
        location_growth = {}
        if location_col and location_col in gdf.columns:
            location_growth = self._analyze_location_growth(gdf, population_col, time_col, location_col)
        
        # Growth projections (simple linear projection)
        projections = self._calculate_growth_projections(gdf, population_col, time_col)
        
        growth_results = {
            'growth_analysis': growth_analysis,
            'trend_analysis': trend_analysis,
            'location_growth': location_growth,
            'growth_projections': projections
        }
        
        self.results['population_growth'] = growth_results
        return growth_results
    
    def analyze_age_structure(self, 
                            gdf: gpd.GeoDataFrame,
                            age_col: str = 'age',
                            population_col: str = 'population') -> Dict:
        """
        Analyze age structure and demographics.
        
        Args:
            gdf: GeoDataFrame with age and population data
            age_col: Column name for age values
            population_col: Column name for population values
            
        Returns:
            Dictionary with age structure analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if age_col not in gdf.columns:
            return {"error": f"Age column '{age_col}' not found"}
            
        # Basic age statistics
        age_values = gdf[age_col].dropna()
        if len(age_values) == 0:
            return {"error": "No valid age data found"}
            
        age_stats = {
            'mean_age': age_values.mean(),
            'median_age': age_values.median(),
            'std_age': age_values.std(),
            'min_age': age_values.min(),
            'max_age': age_values.max(),
            'age_range': age_values.max() - age_values.min()
        }
        
        # Age group classification
        age_groups = self._classify_age_groups(age_values)
        
        # Age pyramid analysis (if population data available)
        age_pyramid = {}
        if population_col in gdf.columns:
            age_pyramid = self._create_age_pyramid(gdf, age_col, population_col)
        
        # Dependency ratios
        dependency_ratios = self._calculate_dependency_ratios(age_values)
        
        # Age distribution analysis
        distribution_analysis = self._analyze_age_distribution(age_values)
        
        age_results = {
            'age_statistics': age_stats,
            'age_groups': age_groups,
            'age_pyramid': age_pyramid,
            'dependency_ratios': dependency_ratios,
            'distribution_analysis': distribution_analysis
        }
        
        self.results['age_structure'] = age_results
        return age_results
    
    def analyze_migration_patterns(self, 
                                 gdf: gpd.GeoDataFrame,
                                 origin_col: str = 'origin',
                                 destination_col: str = 'destination',
                                 migration_col: str = 'migrants',
                                 time_col: str = 'year') -> Dict:
        """
        Analyze migration patterns.
        
        Args:
            gdf: GeoDataFrame with migration data
            origin_col: Column name for origin location
            destination_col: Column name for destination location
            migration_col: Column name for number of migrants
            time_col: Column name for time period
            
        Returns:
            Dictionary with migration pattern analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        required_cols = [origin_col, destination_col, migration_col]
        if not all(col in gdf.columns for col in required_cols):
            return {"error": "Missing required migration columns"}
            
        # Basic migration statistics
        migration_values = gdf[migration_col].dropna()
        if len(migration_values) == 0:
            return {"error": "No valid migration data found"}
            
        migration_stats = {
            'total_migrants': migration_values.sum(),
            'mean_migrants': migration_values.mean(),
            'std_migrants': migration_values.std(),
            'max_migrants': migration_values.max(),
            'min_migrants': migration_values.min(),
            'n_migration_flows': len(migration_values)
        }
        
        # Origin-destination analysis
        od_analysis = self._analyze_origin_destination(gdf, origin_col, destination_col, migration_col)
        
        # Migration network analysis
        network_analysis = self._analyze_migration_network(gdf, origin_col, destination_col, migration_col)
        
        # Temporal migration patterns
        temporal_patterns = {}
        if time_col in gdf.columns:
            temporal_patterns = self._analyze_temporal_migration(gdf, migration_col, time_col)
        
        # Migration intensity analysis
        intensity_analysis = self._analyze_migration_intensity(gdf, origin_col, destination_col, migration_col)
        
        migration_results = {
            'migration_statistics': migration_stats,
            'origin_destination_analysis': od_analysis,
            'network_analysis': network_analysis,
            'temporal_patterns': temporal_patterns,
            'intensity_analysis': intensity_analysis
        }
        
        self.results['migration_patterns'] = migration_results
        return migration_results
    
    def analyze_socioeconomic_indicators(self, 
                                       gdf: gpd.GeoDataFrame,
                                       indicator_cols: List[str] = None) -> Dict:
        """
        Analyze socioeconomic indicators.
        
        Args:
            gdf: GeoDataFrame with socioeconomic data
            indicator_cols: List of socioeconomic indicator column names
            
        Returns:
            Dictionary with socioeconomic analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        if indicator_cols is None:
            # Try to identify socioeconomic columns automatically
            indicator_cols = [col for col in gdf.columns if any(keyword in col.lower() 
                              for keyword in ['income', 'education', 'employment', 'poverty', 'housing', 'health'])]
        
        if not indicator_cols:
            return {"error": "No socioeconomic indicator columns found"}
            
        socioeconomic_analysis = {}
        
        for col in indicator_cols:
            if col in gdf.columns:
                values = gdf[col].dropna()
                if len(values) > 0:
                    # Basic statistics
                    stats_dict = {
                        'count': len(values),
                        'mean': values.mean() if values.dtype in ['float64', 'int64'] else None,
                        'std': values.std() if values.dtype in ['float64', 'int64'] else None,
                        'min': values.min() if values.dtype in ['float64', 'int64'] else None,
                        'max': values.max() if values.dtype in ['float64', 'int64'] else None,
                        'median': values.median() if values.dtype in ['float64', 'int64'] else None
                    }
                    
                    # Inequality measures
                    if values.dtype in ['float64', 'int64'] and len(values) > 1:
                        inequality_measures = self._calculate_inequality_measures(values)
                        stats_dict['inequality_measures'] = inequality_measures
                    
                    socioeconomic_analysis[col] = stats_dict
        
        # Correlation analysis between indicators
        correlation_analysis = self._analyze_indicator_correlations(gdf, indicator_cols)
        
        # Composite socioeconomic index
        composite_index = self._calculate_composite_socioeconomic_index(gdf, indicator_cols)
        
        socioeconomic_results = {
            'indicator_analysis': socioeconomic_analysis,
            'correlation_analysis': correlation_analysis,
            'composite_index': composite_index,
            'indicator_columns': indicator_cols
        }
        
        self.results['socioeconomic_indicators'] = socioeconomic_results
        return socioeconomic_results
    
    def _create_population_density_grid(self, gdf: gpd.GeoDataFrame, population_col: str, cell_size: float) -> Dict:
        """Create population density grid."""
        bounds = gdf.total_bounds
        x_coords = np.arange(bounds[0], bounds[2], cell_size)
        y_coords = np.arange(bounds[1], bounds[3], cell_size)
        
        density_grid = np.zeros((len(y_coords)-1, len(x_coords)-1))
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                cell = Polygon([
                    (x, y), (x + cell_size, y), 
                    (x + cell_size, y + cell_size), (x, y + cell_size)
                ])
                
                # Find features intersecting with this cell
                intersecting_features = gdf[gdf.geometry.intersects(cell)]
                
                if not intersecting_features.empty:
                    # Calculate population in this cell
                    cell_population = 0
                    for idx, feature in intersecting_features.iterrows():
                        if feature.geometry.intersects(cell):
                            # Simple area-weighted population
                            intersection = feature.geometry.intersection(cell)
                            if intersection.area > 0:
                                cell_population += feature[population_col] * (intersection.area / feature.geometry.area)
                    
                    density_grid[j, i] = cell_population / (cell_size ** 2)
        
        return {
            'density_grid': density_grid.tolist(),
            'bounds': bounds.tolist(),
            'cell_size': cell_size,
            'mean_density': np.mean(density_grid),
            'max_density': np.max(density_grid)
        }
    
    def _analyze_population_distribution(self, population_values: pd.Series) -> Dict:
        """Analyze population distribution."""
        # Calculate percentiles
        percentiles = population_values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        
        # Classify population sizes
        classifications = {
            'Small (< 1000)': (population_values < 1000).sum(),
            'Medium (1000-10000)': ((population_values >= 1000) & (population_values < 10000)).sum(),
            'Large (10000-100000)': ((population_values >= 10000) & (population_values < 100000)).sum(),
            'Very Large (> 100000)': (population_values >= 100000).sum()
        }
        
        return {
            'percentiles': percentiles,
            'classifications': classifications
        }
    
    def _analyze_population_clustering(self, gdf: gpd.GeoDataFrame, population_col: str) -> Dict:
        """Analyze spatial clustering of population."""
        # Extract coordinates and population values
        coords = np.array([[point.x, point.y] for point in gdf.geometry.centroid])
        populations = gdf[population_col].values
        
        # Calculate spatial autocorrelation (simplified)
        if len(coords) > 1:
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, np.inf)
            
            # Find nearest neighbors
            nearest_distances = np.min(distances, axis=1)
            mean_nearest_distance = np.mean(nearest_distances)
            
            # Calculate population-weighted mean distance
            weighted_distance = np.average(nearest_distances, weights=populations)
        else:
            mean_nearest_distance = weighted_distance = 0
        
        return {
            'mean_nearest_distance': mean_nearest_distance,
            'weighted_mean_distance': weighted_distance,
            'spatial_concentration': 'high' if mean_nearest_distance < 1000 else 'low'
        }
    
    def _analyze_cross_demographics(self, gdf: gpd.GeoDataFrame, demographic_cols: List[str]) -> Dict:
        """Analyze relationships between demographic variables."""
        cross_analysis = {}
        
        # Calculate correlations between numeric demographic variables
        numeric_cols = [col for col in demographic_cols if gdf[col].dtype in ['float64', 'int64']]
        
        if len(numeric_cols) > 1:
            correlation_matrix = gdf[numeric_cols].corr()
            cross_analysis['correlations'] = correlation_matrix.to_dict()
        
        return cross_analysis
    
    def _calculate_growth_rates(self, gdf: gpd.GeoDataFrame, population_col: str, time_col: str) -> Dict:
        """Calculate population growth rates."""
        growth_rates = []
        
        for i in range(1, len(gdf)):
            current_pop = gdf.iloc[i][population_col]
            previous_pop = gdf.iloc[i-1][population_col]
            
            if previous_pop > 0:
                growth_rate = (current_pop - previous_pop) / previous_pop * 100
                growth_rates.append(growth_rate)
        
        if growth_rates:
            return {
                'mean_growth_rate': np.mean(growth_rates),
                'std_growth_rate': np.std(growth_rates),
                'max_growth_rate': np.max(growth_rates),
                'min_growth_rate': np.min(growth_rates),
                'growth_rates': growth_rates
            }
        else:
            return {'error': 'Insufficient data for growth rate calculation'}
    
    def _analyze_growth_trends(self, gdf: gpd.GeoDataFrame, population_col: str, time_col: str) -> Dict:
        """Analyze population growth trends."""
        # Calculate linear trend
        x = np.arange(len(gdf))
        y = gdf[population_col].values
        
        if len(y) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'growth_trend_slope': slope,
                'growth_trend_r_squared': r_value ** 2,
                'growth_trend_p_value': p_value,
                'trend_significance': 'significant' if p_value < 0.05 else 'not_significant'
            }
        else:
            return {'error': 'Insufficient data for trend analysis'}
    
    def _analyze_location_growth(self, gdf: gpd.GeoDataFrame, population_col: str, time_col: str, location_col: str) -> Dict:
        """Analyze growth by location."""
        location_growth = {}
        
        for location in gdf[location_col].unique():
            location_data = gdf[gdf[location_col] == location].sort_values(time_col)
            
            if len(location_data) > 1:
                growth_rates = []
                for i in range(1, len(location_data)):
                    current_pop = location_data.iloc[i][population_col]
                    previous_pop = location_data.iloc[i-1][population_col]
                    
                    if previous_pop > 0:
                        growth_rate = (current_pop - previous_pop) / previous_pop * 100
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    location_growth[location] = {
                        'mean_growth_rate': np.mean(growth_rates),
                        'total_growth': location_data.iloc[-1][population_col] - location_data.iloc[0][population_col]
                    }
        
        return location_growth
    
    def _calculate_growth_projections(self, gdf: gpd.GeoDataFrame, population_col: str, time_col: str) -> Dict:
        """Calculate simple population growth projections."""
        if len(gdf) < 2:
            return {'error': 'Insufficient data for projections'}
        
        # Calculate average annual growth rate
        total_growth = gdf.iloc[-1][population_col] - gdf.iloc[0][population_col]
        time_periods = len(gdf) - 1
        avg_growth_rate = total_growth / time_periods
        
        # Simple linear projection for next 5 periods
        current_population = gdf.iloc[-1][population_col]
        projections = []
        
        for i in range(1, 6):
            projected_population = current_population + (avg_growth_rate * i)
            projections.append({
                'period': i,
                'projected_population': projected_population,
                'growth': avg_growth_rate * i
            })
        
        return {
            'average_growth_rate': avg_growth_rate,
            'projections': projections
        }
    
    def _classify_age_groups(self, age_values: pd.Series) -> Dict:
        """Classify ages into groups."""
        age_groups = {
            'Children (0-14)': ((age_values >= 0) & (age_values < 15)).sum(),
            'Youth (15-24)': ((age_values >= 15) & (age_values < 25)).sum(),
            'Working Age (25-64)': ((age_values >= 25) & (age_values < 65)).sum(),
            'Elderly (65+)': (age_values >= 65).sum()
        }
        
        total = len(age_values)
        if total > 0:
            age_groups = {k: (v, v/total*100) for k, v in age_groups.items()}
        
        return age_groups
    
    def _create_age_pyramid(self, gdf: gpd.GeoDataFrame, age_col: str, population_col: str) -> Dict:
        """Create age pyramid data."""
        # Group ages into 5-year intervals
        gdf = gdf.copy()
        gdf['age_group'] = pd.cut(gdf[age_col], bins=np.arange(0, 105, 5), labels=False)
        
        age_pyramid_data = gdf.groupby('age_group')[population_col].sum().to_dict()
        
        return {
            'age_pyramid_data': age_pyramid_data,
            'total_population': gdf[population_col].sum()
        }
    
    def _calculate_dependency_ratios(self, age_values: pd.Series) -> Dict:
        """Calculate dependency ratios."""
        children = ((age_values >= 0) & (age_values < 15)).sum()
        elderly = (age_values >= 65).sum()
        working_age = ((age_values >= 15) & (age_values < 65)).sum()
        
        total = len(age_values)
        
        if working_age > 0:
            youth_dependency = (children / working_age) * 100
            elderly_dependency = (elderly / working_age) * 100
            total_dependency = ((children + elderly) / working_age) * 100
        else:
            youth_dependency = elderly_dependency = total_dependency = 0
        
        return {
            'youth_dependency_ratio': youth_dependency,
            'elderly_dependency_ratio': elderly_dependency,
            'total_dependency_ratio': total_dependency
        }
    
    def _analyze_age_distribution(self, age_values: pd.Series) -> Dict:
        """Analyze age distribution."""
        return {
            'age_percentiles': age_values.quantile([0.25, 0.5, 0.75]).to_dict(),
            'age_skewness': age_values.skew(),
            'age_kurtosis': age_values.kurtosis()
        }
    
    def _analyze_origin_destination(self, gdf: gpd.GeoDataFrame, origin_col: str, destination_col: str, migration_col: str) -> Dict:
        """Analyze origin-destination migration patterns."""
        # Top origins and destinations
        top_origins = gdf.groupby(origin_col)[migration_col].sum().nlargest(10).to_dict()
        top_destinations = gdf.groupby(destination_col)[migration_col].sum().nlargest(10).to_dict()
        
        # Net migration by location
        out_migration = gdf.groupby(origin_col)[migration_col].sum()
        in_migration = gdf.groupby(destination_col)[migration_col].sum()
        
        net_migration = {}
        all_locations = set(out_migration.index) | set(in_migration.index)
        
        for location in all_locations:
            out_flow = out_migration.get(location, 0)
            in_flow = in_migration.get(location, 0)
            net_migration[location] = in_flow - out_flow
        
        return {
            'top_origins': top_origins,
            'top_destinations': top_destinations,
            'net_migration': net_migration
        }
    
    def _analyze_migration_network(self, gdf: gpd.GeoDataFrame, origin_col: str, destination_col: str, migration_col: str) -> Dict:
        """Analyze migration network characteristics."""
        # Calculate network statistics
        unique_origins = gdf[origin_col].nunique()
        unique_destinations = gdf[destination_col].nunique()
        total_flows = len(gdf)
        
        # Average flow size
        avg_flow_size = gdf[migration_col].mean()
        
        return {
            'unique_origins': unique_origins,
            'unique_destinations': unique_destinations,
            'total_flows': total_flows,
            'average_flow_size': avg_flow_size,
            'network_density': total_flows / (unique_origins * unique_destinations) if unique_origins * unique_destinations > 0 else 0
        }
    
    def _analyze_temporal_migration(self, gdf: gpd.GeoDataFrame, migration_col: str, time_col: str) -> Dict:
        """Analyze temporal migration patterns."""
        temporal_stats = gdf.groupby(time_col)[migration_col].agg(['sum', 'mean', 'count']).to_dict()
        
        return {
            'temporal_statistics': temporal_stats
        }
    
    def _analyze_migration_intensity(self, gdf: gpd.GeoDataFrame, origin_col: str, destination_col: str, migration_col: str) -> Dict:
        """Analyze migration intensity patterns."""
        # Calculate migration intensity (migrants per origin-destination pair)
        intensity_stats = {
            'mean_intensity': gdf[migration_col].mean(),
            'max_intensity': gdf[migration_col].max(),
            'intensity_percentiles': gdf[migration_col].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        }
        
        return intensity_stats
    
    def _calculate_inequality_measures(self, values: pd.Series) -> Dict:
        """Calculate inequality measures."""
        # Gini coefficient (simplified)
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Coefficient of variation
        cv = values.std() / values.mean() if values.mean() > 0 else 0
        
        return {
            'gini_coefficient': gini,
            'coefficient_of_variation': cv,
            'p90_p10_ratio': values.quantile(0.9) / values.quantile(0.1) if values.quantile(0.1) > 0 else 0
        }
    
    def _analyze_indicator_correlations(self, gdf: gpd.GeoDataFrame, indicator_cols: List[str]) -> Dict:
        """Analyze correlations between socioeconomic indicators."""
        numeric_cols = [col for col in indicator_cols if gdf[col].dtype in ['float64', 'int64']]
        
        if len(numeric_cols) > 1:
            correlation_matrix = gdf[numeric_cols].corr()
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(correlation_matrix)
            }
        else:
            return {'error': 'Insufficient numeric indicators for correlation analysis'}
    
    def _find_strong_correlations(self, correlation_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations between indicators."""
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'indicator1': correlation_matrix.columns[i],
                        'indicator2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return strong_correlations
    
    def _calculate_composite_socioeconomic_index(self, gdf: gpd.GeoDataFrame, indicator_cols: List[str]) -> Dict:
        """Calculate composite socioeconomic index."""
        numeric_cols = [col for col in indicator_cols if gdf[col].dtype in ['float64', 'int64']]
        
        if len(numeric_cols) > 0:
            # Standardize indicators
            standardized_data = gdf[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
            
            # Calculate composite index (simple average)
            composite_index = standardized_data.mean(axis=1)
            
            return {
                'composite_index_values': composite_index.tolist(),
                'mean_composite_index': composite_index.mean(),
                'std_composite_index': composite_index.std(),
                'index_percentiles': composite_index.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        else:
            return {'error': 'No numeric indicators available for composite index'}
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all population analyses."""
        report = "=== POPULATION ANALYSIS REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['density_grid', 'correlation_matrix']:
                        if isinstance(value, float):
                            report += f"{key}: {value:.4f}\n"
                        elif isinstance(value, dict):
                            report += f"{key}:\n"
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, tuple):
                                    count, percentage = sub_value
                                    report += f"  {sub_key}: {count} ({percentage:.1f}%)\n"
                                elif isinstance(sub_value, float):
                                    report += f"  {sub_key}: {sub_value:.4f}\n"
                                else:
                                    report += f"  {sub_key}: {sub_value}\n"
                        else:
                            report += f"{key}: {value}\n"
            report += "\n"
        
        return report
