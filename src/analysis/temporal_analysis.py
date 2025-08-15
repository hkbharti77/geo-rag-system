"""
Temporal Analysis Module

Provides temporal analysis capabilities for geographic data including:
- Time-series analysis
- Seasonal decomposition
- Trend analysis
- Temporal clustering
- Change detection
- Temporal autocorrelation
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import detrend
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalysis:
    """Temporal analysis tools for geographic time-series data."""
    
    def __init__(self):
        self.results = {}
        
    def time_series_analysis(self, 
                           gdf: gpd.GeoDataFrame,
                           time_column: str,
                           value_column: str,
                           location_column: str = None) -> Dict:
        """
        Perform comprehensive time series analysis.
        
        Args:
            gdf: GeoDataFrame with temporal data
            time_column: Column containing time information
            value_column: Column containing values to analyze
            location_column: Column identifying locations (optional)
            
        Returns:
            Dictionary with time series analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        # Ensure time column is datetime
        gdf = gdf.copy()
        gdf[time_column] = pd.to_datetime(gdf[time_column])
        
        # Sort by time
        gdf = gdf.sort_values(time_column)
        
        # Basic statistics
        time_stats = {
            'start_date': gdf[time_column].min(),
            'end_date': gdf[time_column].max(),
            'duration_days': (gdf[time_column].max() - gdf[time_column].min()).days,
            'total_observations': len(gdf),
            'unique_dates': gdf[time_column].nunique(),
            'mean_value': gdf[value_column].mean(),
            'std_value': gdf[value_column].std(),
            'min_value': gdf[value_column].min(),
            'max_value': gdf[value_column].max()
        }
        
        # Temporal trends
        time_trends = self._calculate_temporal_trends(gdf, time_column, value_column)
        
        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(gdf, time_column, value_column)
        
        # Change detection
        change_points = self._detect_change_points(gdf, time_column, value_column)
        
        # Location-based analysis if location column provided
        location_analysis = {}
        if location_column:
            location_analysis = self._analyze_by_location(gdf, time_column, value_column, location_column)
        
        analysis_results = {
            'time_stats': time_stats,
            'time_trends': time_trends,
            'seasonal_patterns': seasonal_patterns,
            'change_points': change_points,
            'location_analysis': location_analysis
        }
        
        self.results['time_series_analysis'] = analysis_results
        return analysis_results
    
    def _calculate_temporal_trends(self, gdf: gpd.GeoDataFrame, time_column: str, value_column: str) -> Dict:
        """Calculate temporal trends in the data."""
        # Convert time to numeric for trend analysis
        time_numeric = (gdf[time_column] - gdf[time_column].min()).dt.total_seconds()
        values = gdf[value_column].values
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        
        # Mann-Kendall trend test
        mk_result = self._mann_kendall_test(values)
        
        # Detrended data
        detrended_values = detrend(values)
        
        trend_results = {
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_r_squared': r_value ** 2,
            'linear_p_value': p_value,
            'linear_std_error': std_err,
            'mann_kendall_statistic': mk_result['statistic'],
            'mann_kendall_p_value': mk_result['p_value'],
            'mann_kendall_trend': mk_result['trend'],
            'detrended_std': np.std(detrended_values),
            'trend_interpretation': self._interpret_trend(slope, p_value, mk_result['trend'])
        }
        
        return trend_results
    
    def _analyze_seasonal_patterns(self, gdf: gpd.GeoDataFrame, time_column: str, value_column: str) -> Dict:
        """Analyze seasonal patterns in the data."""
        # Extract seasonal components
        gdf = gdf.copy()
        gdf['year'] = gdf[time_column].dt.year
        gdf['month'] = gdf[time_column].dt.month
        gdf['day_of_year'] = gdf[time_column].dt.dayofyear
        gdf['week_of_year'] = gdf[time_column].dt.isocalendar().week
        
        # Monthly patterns
        monthly_stats = gdf.groupby('month')[value_column].agg(['mean', 'std', 'count']).to_dict()
        
        # Weekly patterns
        weekly_stats = gdf.groupby('week_of_year')[value_column].agg(['mean', 'std', 'count']).to_dict()
        
        # Seasonal decomposition (simplified)
        seasonal_components = self._simple_seasonal_decomposition(gdf, value_column, 'month')
        
        seasonal_results = {
            'monthly_patterns': monthly_stats,
            'weekly_patterns': weekly_stats,
            'seasonal_components': seasonal_components,
            'seasonality_strength': self._calculate_seasonality_strength(seasonal_components)
        }
        
        return seasonal_results
    
    def _simple_seasonal_decomposition(self, gdf: gpd.GeoDataFrame, value_column: str, period_column: str) -> Dict:
        """Perform simple seasonal decomposition."""
        # Calculate trend (moving average)
        window_size = min(12, len(gdf) // 4)  # Adaptive window size
        if window_size % 2 == 0:
            window_size += 1
            
        trend = gdf[value_column].rolling(window=window_size, center=True).mean()
        
        # Calculate seasonal component
        seasonal = gdf.groupby(period_column)[value_column].mean() - gdf[value_column].mean()
        
        # Calculate residual
        residual = gdf[value_column] - trend - seasonal.reindex(gdf[period_column]).values
        
        decomposition = {
            'trend': trend.tolist(),
            'seasonal': seasonal.to_dict(),
            'residual': residual.tolist(),
            'original': gdf[value_column].tolist()
        }
        
        return decomposition
    
    def _calculate_seasonality_strength(self, seasonal_components: Dict) -> float:
        """Calculate the strength of seasonality."""
        seasonal_values = list(seasonal_components['seasonal'].values())
        residual_values = seasonal_components['residual']
        
        if len(residual_values) > 0:
            seasonal_variance = np.var(seasonal_values)
            residual_variance = np.var(residual_values)
            
            if residual_variance > 0:
                strength = seasonal_variance / (seasonal_variance + residual_variance)
                return min(strength, 1.0)  # Cap at 1.0
        
        return 0.0
    
    def _detect_change_points(self, gdf: gpd.GeoDataFrame, time_column: str, value_column: str) -> Dict:
        """Detect change points in the time series."""
        values = gdf[value_column].values
        times = gdf[time_column].values
        
        # Simple change point detection using rolling statistics
        window_size = min(10, len(values) // 4)
        if window_size < 3:
            return {"error": "Insufficient data for change point detection"}
        
        # Calculate rolling mean and std
        rolling_mean = pd.Series(values).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(values).rolling(window=window_size, center=True).std()
        
        # Detect significant changes
        change_points = []
        threshold = 2.0  # Standard deviations
        
        for i in range(window_size, len(values) - window_size):
            if not np.isnan(rolling_mean.iloc[i]):
                # Check if current value is significantly different from rolling mean
                z_score = abs(values[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                if z_score > threshold and not np.isnan(z_score):
                    change_points.append({
                        'index': i,
                        'time': times[i],
                        'value': values[i],
                        'z_score': z_score,
                        'rolling_mean': rolling_mean.iloc[i]
                    })
        
        # Remove duplicate change points (within window)
        filtered_changes = []
        for cp in change_points:
            is_duplicate = False
            for existing_cp in filtered_changes:
                if abs(cp['index'] - existing_cp['index']) < window_size:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_changes.append(cp)
        
        change_point_results = {
            'change_points': filtered_changes,
            'n_change_points': len(filtered_changes),
            'window_size': window_size,
            'threshold': threshold
        }
        
        return change_point_results
    
    def _analyze_by_location(self, gdf: gpd.GeoDataFrame, time_column: str, value_column: str, location_column: str) -> Dict:
        """Analyze temporal patterns by location."""
        location_analysis = {}
        
        for location in gdf[location_column].unique():
            location_data = gdf[gdf[location_column] == location]
            
            if len(location_data) > 1:
                # Calculate location-specific trends
                time_numeric = (location_data[time_column] - location_data[time_column].min()).dt.total_seconds()
                values = location_data[value_column].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
                
                location_analysis[location] = {
                    'n_observations': len(location_data),
                    'mean_value': values.mean(),
                    'std_value': values.std(),
                    'trend_slope': slope,
                    'trend_r_squared': r_value ** 2,
                    'trend_p_value': p_value,
                    'start_date': location_data[time_column].min(),
                    'end_date': location_data[time_column].max()
                }
        
        return location_analysis
    
    def _mann_kendall_test(self, values: np.ndarray) -> Dict:
        """Perform Mann-Kendall trend test."""
        n = len(values)
        if n < 3:
            return {'statistic': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if values[j] > values[i]:
                    s += 1
                elif values[j] < values[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend
        if p_value < 0.05:
            if s > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no_trend'
        
        return {
            'statistic': s,
            'p_value': p_value,
            'trend': trend
        }
    
    def _interpret_trend(self, slope: float, p_value: float, mk_trend: str) -> str:
        """Interpret trend results."""
        if p_value < 0.05:
            if slope > 0:
                return "Significant increasing trend"
            else:
                return "Significant decreasing trend"
        else:
            return "No significant trend"
    
    def temporal_clustering(self, 
                          gdf: gpd.GeoDataFrame,
                          time_column: str,
                          value_column: str,
                          n_clusters: int = 3) -> Dict:
        """
        Perform temporal clustering analysis.
        
        Args:
            gdf: GeoDataFrame with temporal data
            time_column: Column containing time information
            value_column: Column containing values to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with temporal clustering results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        # Prepare data for clustering
        gdf = gdf.copy()
        gdf[time_column] = pd.to_datetime(gdf[time_column])
        gdf = gdf.sort_values(time_column)
        
        # Create features for clustering
        time_numeric = (gdf[time_column] - gdf[time_column].min()).dt.total_seconds()
        values = gdf[value_column].values
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(np.column_stack([time_numeric, values]))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = gdf[cluster_mask]
            
            cluster_analysis[i] = {
                'size': len(cluster_data),
                'start_time': cluster_data[time_column].min(),
                'end_time': cluster_data[time_column].max(),
                'mean_value': cluster_data[value_column].mean(),
                'std_value': cluster_data[value_column].std(),
                'center': kmeans.cluster_centers_[i].tolist()
            }
        
        clustering_results = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        }
        
        self.results['temporal_clustering'] = clustering_results
        return clustering_results
    
    def temporal_autocorrelation(self, 
                               gdf: gpd.GeoDataFrame,
                               time_column: str,
                               value_column: str,
                               max_lag: int = 10) -> Dict:
        """
        Calculate temporal autocorrelation.
        
        Args:
            gdf: GeoDataFrame with temporal data
            time_column: Column containing time information
            value_column: Column containing values to analyze
            max_lag: Maximum lag to calculate
            
        Returns:
            Dictionary with temporal autocorrelation results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        # Prepare data
        gdf = gdf.copy()
        gdf[time_column] = pd.to_datetime(gdf[time_column])
        gdf = gdf.sort_values(time_column)
        values = gdf[value_column].values
        
        if len(values) < max_lag + 1:
            max_lag = len(values) - 1
        
        # Calculate autocorrelation
        autocorr_values = []
        for lag in range(1, max_lag + 1):
            if lag < len(values):
                # Calculate correlation between original series and lagged series
                original = values[:-lag]
                lagged = values[lag:]
                
                if len(original) > 1 and len(lagged) > 1:
                    correlation = np.corrcoef(original, lagged)[0, 1]
                    if not np.isnan(correlation):
                        autocorr_values.append(correlation)
                    else:
                        autocorr_values.append(0)
                else:
                    autocorr_values.append(0)
            else:
                autocorr_values.append(0)
        
        # Find significant lags
        significant_lags = []
        for i, ac in enumerate(autocorr_values):
            if abs(ac) > 0.2:  # Threshold for significance
                significant_lags.append(i + 1)
        
        autocorr_results = {
            'autocorrelation_values': autocorr_values,
            'significant_lags': significant_lags,
            'max_autocorrelation': max(autocorr_values) if autocorr_values else 0,
            'min_autocorrelation': min(autocorr_values) if autocorr_values else 0,
            'mean_autocorrelation': np.mean(autocorr_values) if autocorr_values else 0,
            'max_lag': max_lag
        }
        
        self.results['temporal_autocorrelation'] = autocorr_results
        return autocorr_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all temporal analyses."""
        report = "=== TEMPORAL ANALYSIS REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['trend', 'seasonal', 'residual', 'original', 'cluster_labels', 'cluster_centers']:
                        if isinstance(value, float):
                            report += f"{key}: {value:.4f}\n"
                        elif isinstance(value, dict):
                            report += f"{key}:\n"
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, float):
                                    report += f"  {sub_key}: {sub_value:.4f}\n"
                                else:
                                    report += f"  {sub_key}: {sub_value}\n"
                        else:
                            report += f"{key}: {value}\n"
            report += "\n"
        
        return report
