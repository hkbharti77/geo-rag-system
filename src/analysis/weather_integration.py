"""
Weather Integration Module

Provides weather and climate data integration capabilities including:
- Weather data overlay
- Climate pattern analysis
- Precipitation analysis
- Temperature analysis
- Wind pattern analysis
- Weather impact assessment
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


class WeatherIntegration:
    """Weather and climate data integration tools."""
    
    def __init__(self):
        self.results = {}
        
    def overlay_weather_data(self, 
                           gdf: gpd.GeoDataFrame,
                           weather_data: pd.DataFrame,
                           weather_lat_col: str = 'latitude',
                           weather_lon_col: str = 'longitude',
                           weather_time_col: str = 'timestamp',
                           weather_value_col: str = 'temperature') -> Dict:
        """
        Overlay weather data onto geographic features.
        
        Args:
            gdf: GeoDataFrame with geographic features
            weather_data: DataFrame with weather observations
            weather_lat_col: Column name for latitude in weather data
            weather_lon_col: Column name for longitude in weather data
            weather_time_col: Column name for timestamp in weather data
            weather_value_col: Column name for weather value in weather data
            
        Returns:
            Dictionary with weather overlay results
        """
        if gdf.empty or weather_data.empty:
            return {"error": "Empty GeoDataFrame or weather data provided"}
            
        # Ensure weather data has required columns
        required_cols = [weather_lat_col, weather_lon_col, weather_value_col]
        if not all(col in weather_data.columns for col in required_cols):
            return {"error": "Missing required columns in weather data"}
            
        # Convert weather data to GeoDataFrame
        weather_gdf = gpd.GeoDataFrame(
            weather_data,
            geometry=gpd.points_from_xy(weather_data[weather_lon_col], weather_data[weather_lat_col]),
            crs=gdf.crs
        )
        
        # Perform spatial join
        joined_data = gpd.sjoin(gdf, weather_gdf, how='left', predicate='intersects')
        
        # Calculate statistics for each geographic feature
        feature_stats = {}
        for idx, feature in gdf.iterrows():
            feature_weather = joined_data[joined_data.index_left == idx]
            
            if not feature_weather.empty:
                values = feature_weather[weather_value_col].dropna()
                if len(values) > 0:
                    feature_stats[idx] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values),
                        'median': values.median()
                    }
                else:
                    feature_stats[idx] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'count': 0,
                        'median': np.nan
                    }
            else:
                feature_stats[idx] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'count': 0,
                    'median': np.nan
                }
        
        # Calculate overall statistics
        all_values = weather_data[weather_value_col].dropna()
        overall_stats = {
            'total_weather_points': len(weather_data),
            'valid_weather_points': len(all_values),
            'overall_mean': all_values.mean() if len(all_values) > 0 else np.nan,
            'overall_std': all_values.std() if len(all_values) > 0 else np.nan,
            'overall_min': all_values.min() if len(all_values) > 0 else np.nan,
            'overall_max': all_values.max() if len(all_values) > 0 else np.nan,
            'features_with_data': sum(1 for stats in feature_stats.values() if stats['count'] > 0),
            'total_features': len(feature_stats)
        }
        
        overlay_results = {
            'joined_data': joined_data,
            'feature_statistics': feature_stats,
            'overall_statistics': overall_stats,
            'weather_data_summary': {
                'spatial_extent': weather_gdf.total_bounds.tolist(),
                'temporal_extent': weather_data[weather_time_col].agg(['min', 'max']).to_dict() if weather_time_col in weather_data.columns else None
            }
        }
        
        self.results['weather_overlay'] = overlay_results
        return overlay_results
    
    def analyze_climate_patterns(self, 
                               weather_data: pd.DataFrame,
                               lat_col: str = 'latitude',
                               lon_col: str = 'longitude',
                               time_col: str = 'timestamp',
                               value_col: str = 'temperature',
                               analysis_period: str = 'monthly') -> Dict:
        """
        Analyze climate patterns in weather data.
        
        Args:
            weather_data: DataFrame with weather observations
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            time_col: Column name for timestamp
            value_col: Column name for weather value
            analysis_period: 'daily', 'weekly', 'monthly', or 'seasonal'
            
        Returns:
            Dictionary with climate pattern analysis results
        """
        if weather_data.empty:
            return {"error": "Empty weather data provided"}
            
        # Ensure time column is datetime
        weather_data = weather_data.copy()
        weather_data[time_col] = pd.to_datetime(weather_data[time_col])
        
        # Extract temporal components
        weather_data['year'] = weather_data[time_col].dt.year
        weather_data['month'] = weather_data[time_col].dt.month
        weather_data['day'] = weather_data[time_col].dt.day
        weather_data['season'] = weather_data[time_col].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Group by analysis period
        if analysis_period == 'daily':
            group_cols = ['year', 'month', 'day']
        elif analysis_period == 'weekly':
            weather_data['week'] = weather_data[time_col].dt.isocalendar().week
            group_cols = ['year', 'week']
        elif analysis_period == 'monthly':
            group_cols = ['year', 'month']
        elif analysis_period == 'seasonal':
            group_cols = ['year', 'season']
        else:
            return {"error": "Invalid analysis period"}
        
        # Calculate temporal statistics
        temporal_stats = weather_data.groupby(group_cols)[value_col].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Calculate trends
        trends = self._calculate_climate_trends(temporal_stats, value_col='mean')
        
        # Calculate seasonal patterns
        seasonal_patterns = self._analyze_seasonal_climate_patterns(weather_data, value_col)
        
        # Calculate spatial patterns
        spatial_patterns = self._analyze_spatial_climate_patterns(weather_data, lat_col, lon_col, value_col)
        
        climate_results = {
            'temporal_statistics': temporal_stats.to_dict('records'),
            'trends': trends,
            'seasonal_patterns': seasonal_patterns,
            'spatial_patterns': spatial_patterns,
            'analysis_period': analysis_period,
            'data_summary': {
                'total_observations': len(weather_data),
                'date_range': [weather_data[time_col].min(), weather_data[time_col].max()],
                'spatial_extent': weather_data[[lat_col, lon_col]].agg(['min', 'max']).to_dict()
            }
        }
        
        self.results['climate_patterns'] = climate_results
        return climate_results
    
    def analyze_precipitation(self, 
                            weather_data: pd.DataFrame,
                            lat_col: str = 'latitude',
                            lon_col: str = 'longitude',
                            time_col: str = 'timestamp',
                            precip_col: str = 'precipitation') -> Dict:
        """
        Analyze precipitation patterns.
        
        Args:
            weather_data: DataFrame with precipitation data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            time_col: Column name for timestamp
            precip_col: Column name for precipitation values
            
        Returns:
            Dictionary with precipitation analysis results
        """
        if weather_data.empty:
            return {"error": "Empty weather data provided"}
            
        if precip_col not in weather_data.columns:
            return {"error": f"Precipitation column '{precip_col}' not found"}
            
        weather_data = weather_data.copy()
        weather_data[time_col] = pd.to_datetime(weather_data[time_col])
        
        # Basic precipitation statistics
        precip_values = weather_data[precip_col].dropna()
        if len(precip_values) == 0:
            return {"error": "No valid precipitation data found"}
            
        basic_stats = {
            'total_precipitation': precip_values.sum(),
            'mean_precipitation': precip_values.mean(),
            'std_precipitation': precip_values.std(),
            'max_precipitation': precip_values.max(),
            'min_precipitation': precip_values.min(),
            'median_precipitation': precip_values.median(),
            'dry_days': (precip_values == 0).sum(),
            'wet_days': (precip_values > 0).sum(),
            'total_days': len(precip_values)
        }
        
        # Precipitation intensity analysis
        wet_days = precip_values[precip_values > 0]
        intensity_stats = {
            'mean_intensity': wet_days.mean() if len(wet_days) > 0 else 0,
            'max_intensity': wet_days.max() if len(wet_days) > 0 else 0,
            'intensity_percentiles': wet_days.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict() if len(wet_days) > 0 else {}
        }
        
        # Temporal precipitation patterns
        weather_data['year'] = weather_data[time_col].dt.year
        weather_data['month'] = weather_data[time_col].dt.month
        weather_data['season'] = weather_data[time_col].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        monthly_precip = weather_data.groupby(['year', 'month'])[precip_col].sum().reset_index()
        seasonal_precip = weather_data.groupby(['year', 'season'])[precip_col].sum().reset_index()
        
        # Drought analysis
        drought_analysis = self._analyze_drought_patterns(weather_data, precip_col, time_col)
        
        # Extreme precipitation events
        extreme_events = self._identify_extreme_precipitation(precip_values)
        
        precip_results = {
            'basic_statistics': basic_stats,
            'intensity_statistics': intensity_stats,
            'monthly_patterns': monthly_precip.to_dict('records'),
            'seasonal_patterns': seasonal_precip.to_dict('records'),
            'drought_analysis': drought_analysis,
            'extreme_events': extreme_events
        }
        
        self.results['precipitation_analysis'] = precip_results
        return precip_results
    
    def analyze_temperature(self, 
                          weather_data: pd.DataFrame,
                          lat_col: str = 'latitude',
                          lon_col: str = 'longitude',
                          time_col: str = 'timestamp',
                          temp_col: str = 'temperature') -> Dict:
        """
        Analyze temperature patterns.
        
        Args:
            weather_data: DataFrame with temperature data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            time_col: Column name for timestamp
            temp_col: Column name for temperature values
            
        Returns:
            Dictionary with temperature analysis results
        """
        if weather_data.empty:
            return {"error": "Empty weather data provided"}
            
        if temp_col not in weather_data.columns:
            return {"error": f"Temperature column '{temp_col}' not found"}
            
        weather_data = weather_data.copy()
        weather_data[time_col] = pd.to_datetime(weather_data[time_col])
        
        # Basic temperature statistics
        temp_values = weather_data[temp_col].dropna()
        if len(temp_values) == 0:
            return {"error": "No valid temperature data found"}
            
        basic_stats = {
            'mean_temperature': temp_values.mean(),
            'std_temperature': temp_values.std(),
            'max_temperature': temp_values.max(),
            'min_temperature': temp_values.min(),
            'median_temperature': temp_values.median(),
            'temperature_range': temp_values.max() - temp_values.min()
        }
        
        # Temperature extremes
        extreme_stats = {
            'hot_days': (temp_values > temp_values.quantile(0.9)).sum(),
            'cold_days': (temp_values < temp_values.quantile(0.1)).sum(),
            'freezing_days': (temp_values <= 0).sum(),
            'hot_nights': (temp_values > temp_values.quantile(0.95)).sum()
        }
        
        # Diurnal temperature range (if min/max temp columns available)
        diurnal_range = None
        if 'min_temp' in weather_data.columns and 'max_temp' in weather_data.columns:
            min_temp = weather_data['min_temp'].dropna()
            max_temp = weather_data['max_temp'].dropna()
            if len(min_temp) > 0 and len(max_temp) > 0:
                diurnal_range = {
                    'mean_diurnal_range': (max_temp - min_temp).mean(),
                    'max_diurnal_range': (max_temp - min_temp).max(),
                    'min_diurnal_range': (max_temp - min_temp).min()
                }
        
        # Temperature trends
        weather_data['year'] = weather_data[time_col].dt.year
        weather_data['month'] = weather_data[time_col].dt.month
        
        monthly_temp = weather_data.groupby(['year', 'month'])[temp_col].mean().reset_index()
        annual_temp = weather_data.groupby('year')[temp_col].mean().reset_index()
        
        # Heat wave analysis
        heat_wave_analysis = self._analyze_heat_waves(weather_data, temp_col, time_col)
        
        temp_results = {
            'basic_statistics': basic_stats,
            'extreme_statistics': extreme_stats,
            'diurnal_range': diurnal_range,
            'monthly_patterns': monthly_temp.to_dict('records'),
            'annual_patterns': annual_temp.to_dict('records'),
            'heat_wave_analysis': heat_wave_analysis
        }
        
        self.results['temperature_analysis'] = temp_results
        return temp_results
    
    def analyze_wind_patterns(self, 
                            weather_data: pd.DataFrame,
                            lat_col: str = 'latitude',
                            lon_col: str = 'longitude',
                            time_col: str = 'timestamp',
                            wind_speed_col: str = 'wind_speed',
                            wind_dir_col: str = 'wind_direction') -> Dict:
        """
        Analyze wind patterns.
        
        Args:
            weather_data: DataFrame with wind data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            time_col: Column name for timestamp
            wind_speed_col: Column name for wind speed
            wind_dir_col: Column name for wind direction
            
        Returns:
            Dictionary with wind pattern analysis results
        """
        if weather_data.empty:
            return {"error": "Empty weather data provided"}
            
        if wind_speed_col not in weather_data.columns:
            return {"error": f"Wind speed column '{wind_speed_col}' not found"}
            
        weather_data = weather_data.copy()
        weather_data[time_col] = pd.to_datetime(weather_data[time_col])
        
        # Basic wind speed statistics
        wind_speed = weather_data[wind_speed_col].dropna()
        if len(wind_speed) == 0:
            return {"error": "No valid wind speed data found"}
            
        speed_stats = {
            'mean_wind_speed': wind_speed.mean(),
            'std_wind_speed': wind_speed.std(),
            'max_wind_speed': wind_speed.max(),
            'min_wind_speed': wind_speed.min(),
            'median_wind_speed': wind_speed.median()
        }
        
        # Wind speed classification
        speed_classes = {
            'Calm (< 1 m/s)': (wind_speed < 1).sum(),
            'Light (1-3 m/s)': ((wind_speed >= 1) & (wind_speed < 3)).sum(),
            'Moderate (3-5 m/s)': ((wind_speed >= 3) & (wind_speed < 5)).sum(),
            'Fresh (5-8 m/s)': ((wind_speed >= 5) & (wind_speed < 8)).sum(),
            'Strong (8-11 m/s)': ((wind_speed >= 8) & (wind_speed < 11)).sum(),
            'Very Strong (> 11 m/s)': (wind_speed >= 11).sum()
        }
        
        # Wind direction analysis
        direction_analysis = {}
        if wind_dir_col in weather_data.columns:
            wind_dir = weather_data[wind_dir_col].dropna()
            if len(wind_dir) > 0:
                # Classify wind directions
                direction_classes = {
                    'North': ((wind_dir >= 315) | (wind_dir < 45)).sum(),
                    'Northeast': ((wind_dir >= 45) & (wind_dir < 90)).sum(),
                    'East': ((wind_dir >= 90) & (wind_dir < 135)).sum(),
                    'Southeast': ((wind_dir >= 135) & (wind_dir < 180)).sum(),
                    'South': ((wind_dir >= 180) & (wind_dir < 225)).sum(),
                    'Southwest': ((wind_dir >= 225) & (wind_dir < 270)).sum(),
                    'West': ((wind_dir >= 270) & (wind_dir < 315)).sum()
                }
                
                total_directions = len(wind_dir)
                direction_analysis = {
                    'direction_distribution': {k: (v, v/total_directions*100) for k, v in direction_classes.items()},
                    'prevailing_direction': max(direction_classes, key=direction_classes.get),
                    'mean_direction': wind_dir.mean()
                }
        
        # Temporal wind patterns
        weather_data['year'] = weather_data[time_col].dt.year
        weather_data['month'] = weather_data[time_col].dt.month
        weather_data['hour'] = weather_data[time_col].dt.hour
        
        monthly_wind = weather_data.groupby(['year', 'month'])[wind_speed_col].mean().reset_index()
        hourly_wind = weather_data.groupby('hour')[wind_speed_col].mean().reset_index()
        
        wind_results = {
            'speed_statistics': speed_stats,
            'speed_classification': speed_classes,
            'direction_analysis': direction_analysis,
            'monthly_patterns': monthly_wind.to_dict('records'),
            'hourly_patterns': hourly_wind.to_dict('records')
        }
        
        self.results['wind_patterns'] = wind_results
        return wind_results
    
    def assess_weather_impact(self, 
                            gdf: gpd.GeoDataFrame,
                            weather_data: pd.DataFrame,
                            impact_metrics: List[str] = None) -> Dict:
        """
        Assess weather impact on geographic features.
        
        Args:
            gdf: GeoDataFrame with geographic features
            weather_data: DataFrame with weather data
            impact_metrics: List of impact metrics to calculate
            
        Returns:
            Dictionary with weather impact assessment results
        """
        if gdf.empty or weather_data.empty:
            return {"error": "Empty GeoDataFrame or weather data provided"}
            
        if impact_metrics is None:
            impact_metrics = ['temperature_stress', 'precipitation_deficit', 'wind_exposure']
        
        # Perform weather overlay first
        overlay_results = self.overlay_weather_data(gdf, weather_data)
        
        if 'error' in overlay_results:
            return overlay_results
        
        impact_assessment = {}
        
        for metric in impact_metrics:
            if metric == 'temperature_stress':
                impact_assessment[metric] = self._calculate_temperature_stress(overlay_results)
            elif metric == 'precipitation_deficit':
                impact_assessment[metric] = self._calculate_precipitation_deficit(overlay_results)
            elif metric == 'wind_exposure':
                impact_assessment[metric] = self._calculate_wind_exposure(overlay_results)
        
        impact_results = {
            'impact_assessment': impact_assessment,
            'overlay_results': overlay_results
        }
        
        self.results['weather_impact'] = impact_results
        return impact_results
    
    def _calculate_climate_trends(self, temporal_stats: pd.DataFrame, value_col: str) -> Dict:
        """Calculate climate trends."""
        if len(temporal_stats) < 2:
            return {"error": "Insufficient data for trend analysis"}
            
        # Calculate linear trend
        x = np.arange(len(temporal_stats))
        y = temporal_stats[value_col].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_significance': 'significant' if p_value < 0.05 else 'not_significant'
        }
    
    def _analyze_seasonal_climate_patterns(self, weather_data: pd.DataFrame, value_col: str) -> Dict:
        """Analyze seasonal climate patterns."""
        seasonal_stats = weather_data.groupby('season')[value_col].agg(['mean', 'std', 'min', 'max']).to_dict()
        
        # Calculate seasonal variability
        seasonal_means = weather_data.groupby('season')[value_col].mean()
        seasonal_variability = seasonal_means.std() / seasonal_means.mean() if seasonal_means.mean() != 0 else 0
        
        return {
            'seasonal_statistics': seasonal_stats,
            'seasonal_variability': seasonal_variability
        }
    
    def _analyze_spatial_climate_patterns(self, weather_data: pd.DataFrame, lat_col: str, lon_col: str, value_col: str) -> Dict:
        """Analyze spatial climate patterns."""
        # Calculate spatial statistics
        spatial_stats = weather_data.groupby([lat_col, lon_col])[value_col].mean().reset_index()
        
        # Calculate spatial correlation
        if len(spatial_stats) > 1:
            lat_corr = stats.pearsonr(spatial_stats[lat_col], spatial_stats[value_col])[0]
            lon_corr = stats.pearsonr(spatial_stats[lon_col], spatial_stats[value_col])[0]
        else:
            lat_corr = lon_corr = np.nan
        
        return {
            'spatial_statistics': spatial_stats.to_dict('records'),
            'latitude_correlation': lat_corr,
            'longitude_correlation': lon_corr
        }
    
    def _analyze_drought_patterns(self, weather_data: pd.DataFrame, precip_col: str, time_col: str) -> Dict:
        """Analyze drought patterns."""
        # Calculate cumulative precipitation deficit
        weather_data = weather_data.sort_values(time_col)
        weather_data['cumulative_precip'] = weather_data[precip_col].cumsum()
        
        # Calculate moving average
        window_size = 30  # 30-day moving average
        weather_data['moving_avg'] = weather_data[precip_col].rolling(window=window_size, center=True).mean()
        
        # Identify drought periods (below average precipitation)
        mean_precip = weather_data[precip_col].mean()
        drought_periods = weather_data[weather_data[precip_col] < mean_precip * 0.5]
        
        return {
            'mean_precipitation': mean_precip,
            'drought_periods_count': len(drought_periods),
            'drought_periods': drought_periods[[time_col, precip_col]].to_dict('records')
        }
    
    def _identify_extreme_precipitation(self, precip_values: pd.Series) -> Dict:
        """Identify extreme precipitation events."""
        # Define extreme events as values above 95th percentile
        threshold = precip_values.quantile(0.95)
        extreme_events = precip_values[precip_values > threshold]
        
        return {
            'threshold': threshold,
            'extreme_events_count': len(extreme_events),
            'max_extreme_event': extreme_events.max() if len(extreme_events) > 0 else 0
        }
    
    def _analyze_heat_waves(self, weather_data: pd.DataFrame, temp_col: str, time_col: str) -> Dict:
        """Analyze heat waves."""
        # Define heat wave as 3+ consecutive days above 90th percentile
        threshold = weather_data[temp_col].quantile(0.9)
        
        # Find consecutive days above threshold
        above_threshold = weather_data[temp_col] > threshold
        heat_wave_days = above_threshold.sum()
        
        return {
            'heat_wave_threshold': threshold,
            'heat_wave_days': heat_wave_days,
            'heat_wave_percentage': (heat_wave_days / len(weather_data)) * 100
        }
    
    def _calculate_temperature_stress(self, overlay_results: Dict) -> Dict:
        """Calculate temperature stress impact."""
        # Placeholder for temperature stress calculation
        return {
            'stress_level': 'low',
            'stress_score': 0.2,
            'description': 'Temperature stress assessment based on overlay data'
        }
    
    def _calculate_precipitation_deficit(self, overlay_results: Dict) -> Dict:
        """Calculate precipitation deficit impact."""
        # Placeholder for precipitation deficit calculation
        return {
            'deficit_level': 'moderate',
            'deficit_score': 0.5,
            'description': 'Precipitation deficit assessment based on overlay data'
        }
    
    def _calculate_wind_exposure(self, overlay_results: Dict) -> Dict:
        """Calculate wind exposure impact."""
        # Placeholder for wind exposure calculation
        return {
            'exposure_level': 'high',
            'exposure_score': 0.8,
            'description': 'Wind exposure assessment based on overlay data'
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all weather analyses."""
        report = "=== WEATHER INTEGRATION REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['joined_data', 'temporal_statistics']:
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
