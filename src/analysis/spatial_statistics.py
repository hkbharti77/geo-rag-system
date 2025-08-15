"""
Spatial Statistics Module

Provides spatial statistical analysis including:
- Point density analysis
- Spatial clustering (K-means, DBSCAN)
- Hot spot analysis
- Spatial autocorrelation
- Nearest neighbor analysis
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SpatialStatistics:
    """Spatial statistical analysis tools for geographic data."""
    
    def __init__(self):
        self.results = {}
        
    def point_density_analysis(self, 
                              gdf: gpd.GeoDataFrame, 
                              column: str = None,
                              cell_size: float = 1000,
                              method: str = 'kernel') -> Dict:
        """
        Perform point density analysis.
        
        Args:
            gdf: GeoDataFrame with point geometries
            column: Column to use for weighted density (optional)
            cell_size: Size of grid cells in meters
            method: 'kernel' or 'grid' density estimation
            
        Returns:
            Dictionary with density results and statistics
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        # Ensure we have point geometries
        if not all(isinstance(geom, Point) for geom in gdf.geometry):
            gdf = gdf.copy()
            gdf['geometry'] = gdf.geometry.centroid
            
        # Create grid for density calculation
        bounds = gdf.total_bounds
        x_coords = np.arange(bounds[0], bounds[2], cell_size)
        y_coords = np.arange(bounds[1], bounds[3], cell_size)
        
        density_grid = np.zeros((len(y_coords)-1, len(x_coords)-1))
        
        if method == 'kernel':
            # Kernel density estimation
            for i, x in enumerate(x_coords[:-1]):
                for j, y in enumerate(y_coords[:-1]):
                    cell_center = Point(x + cell_size/2, y + cell_size/2)
                    distances = [cell_center.distance(point) for point in gdf.geometry]
                    
                    if column:
                        weights = gdf[column].values
                    else:
                        weights = np.ones(len(gdf))
                    
                    # Gaussian kernel
                    bandwidth = cell_size * 2
                    kernel_weights = weights * np.exp(-0.5 * (np.array(distances) / bandwidth) ** 2)
                    density_grid[j, i] = np.sum(kernel_weights)
        
        else:  # Grid method
            for i, x in enumerate(x_coords[:-1]):
                for j, y in enumerate(y_coords[:-1]):
                    cell = Polygon([
                        (x, y), (x + cell_size, y), 
                        (x + cell_size, y + cell_size), (x, y + cell_size)
                    ])
                    points_in_cell = gdf[gdf.geometry.within(cell)]
                    
                    if column:
                        density_grid[j, i] = points_in_cell[column].sum()
                    else:
                        density_grid[j, i] = len(points_in_cell)
        
        # Calculate statistics
        density_stats = {
            'mean_density': np.mean(density_grid),
            'max_density': np.max(density_grid),
            'min_density': np.min(density_grid),
            'std_density': np.std(density_grid),
            'total_points': len(gdf),
            'density_grid': density_grid,
            'bounds': bounds,
            'cell_size': cell_size
        }
        
        self.results['density_analysis'] = density_stats
        return density_stats
    
    def spatial_clustering(self, 
                          gdf: gpd.GeoDataFrame,
                          method: str = 'kmeans',
                          n_clusters: int = 5,
                          eps: float = 1000,
                          min_samples: int = 5) -> Dict:
        """
        Perform spatial clustering analysis.
        
        Args:
            gdf: GeoDataFrame with point geometries
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters for K-means
            eps: Epsilon for DBSCAN
            min_samples: Minimum samples for DBSCAN
            
        Returns:
            Dictionary with clustering results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords)
            cluster_centers = kmeans.cluster_centers_
            
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(coords)
            cluster_centers = []
            
            for label in set(cluster_labels):
                if label != -1:  # Skip noise points
                    cluster_points = coords[cluster_labels == label]
                    cluster_centers.append(np.mean(cluster_points, axis=0))
            cluster_centers = np.array(cluster_centers)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for label in set(cluster_labels):
            if label != -1:  # Skip noise for DBSCAN
                cluster_points = coords[cluster_labels == label]
                cluster_stats[label] = {
                    'size': len(cluster_points),
                    'center': cluster_centers[label] if method == 'kmeans' else cluster_centers[list(set(cluster_labels)).index(label)],
                    'density': len(cluster_points) / len(gdf)
                }
        
        clustering_results = {
            'method': method,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_stats': cluster_stats,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'noise_points': np.sum(cluster_labels == -1) if method == 'dbscan' else 0
        }
        
        self.results['clustering'] = clustering_results
        return clustering_results
    
    def hot_spot_analysis(self, 
                         gdf: gpd.GeoDataFrame,
                         column: str,
                         distance_threshold: float = 1000) -> Dict:
        """
        Perform hot spot analysis using Getis-Ord Gi* statistic.
        
        Args:
            gdf: GeoDataFrame with point geometries
            column: Column to analyze for hot spots
            distance_threshold: Distance threshold for neighbor definition
            
        Returns:
            Dictionary with hot spot analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        values = gdf[column].values
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        # Calculate distance matrix
        dist_matrix = squareform(pdist(coords))
        
        # Create spatial weights matrix
        weights_matrix = (dist_matrix <= distance_threshold).astype(float)
        np.fill_diagonal(weights_matrix, 0)  # Exclude self
        
        # Calculate Getis-Ord Gi* statistic
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        gi_star = []
        for i in range(n):
            # Sum of weighted values for neighbors
            sum_wx = np.sum(weights_matrix[i] * values)
            # Sum of weights
            sum_w = np.sum(weights_matrix[i])
            # Sum of squared weights
            sum_w2 = np.sum(weights_matrix[i] ** 2)
            
            # Expected value
            expected = sum_w * mean_val
            
            # Variance
            variance = std_val ** 2 * (sum_w2 * (n - 1) - sum_w ** 2) / (n - 1)
            
            # Gi* statistic
            if variance > 0:
                gi_stat = (sum_wx - expected) / np.sqrt(variance)
            else:
                gi_stat = 0
                
            gi_star.append(gi_stat)
        
        # Classify hot spots
        hot_spots = []
        cold_spots = []
        
        for i, gi in enumerate(gi_star):
            if gi > 1.96:  # 95% confidence level
                hot_spots.append(i)
            elif gi < -1.96:
                cold_spots.append(i)
        
        hot_spot_results = {
            'gi_star_values': gi_star,
            'hot_spots': hot_spots,
            'cold_spots': cold_spots,
            'mean_gi_star': np.mean(gi_star),
            'std_gi_star': np.std(gi_star),
            'n_hot_spots': len(hot_spots),
            'n_cold_spots': len(cold_spots),
            'distance_threshold': distance_threshold
        }
        
        self.results['hot_spot_analysis'] = hot_spot_results
        return hot_spot_results
    
    def spatial_autocorrelation(self, 
                               gdf: gpd.GeoDataFrame,
                               column: str,
                               distance_threshold: float = 1000) -> Dict:
        """
        Calculate Moran's I spatial autocorrelation.
        
        Args:
            gdf: GeoDataFrame with point geometries
            column: Column to analyze
            distance_threshold: Distance threshold for neighbor definition
            
        Returns:
            Dictionary with Moran's I results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        values = gdf[column].values
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        # Calculate distance matrix
        dist_matrix = squareform(pdist(coords))
        
        # Create spatial weights matrix
        weights_matrix = (dist_matrix <= distance_threshold).astype(float)
        np.fill_diagonal(weights_matrix, 0)
        
        # Calculate Moran's I
        n = len(values)
        mean_val = np.mean(values)
        
        # Numerator: sum of weighted deviations
        numerator = 0
        denominator = 0
        sum_w = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    w_ij = weights_matrix[i, j]
                    sum_w += w_ij
                    numerator += w_ij * (values[i] - mean_val) * (values[j] - mean_val)
                    denominator += (values[i] - mean_val) ** 2
        
        if denominator > 0 and sum_w > 0:
            morans_i = (n * numerator) / (2 * sum_w * denominator)
        else:
            morans_i = 0
        
        # Calculate expected value and variance for significance testing
        expected_i = -1 / (n - 1)
        
        # Simplified variance calculation
        variance_i = (n ** 2 - 3 * n + 3) / ((n - 1) * (n + 1)) - expected_i ** 2
        
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
        else:
            z_score = 0
        
        autocorr_results = {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'z_score': z_score,
            'p_value': 2 * (1 - abs(z_score)),  # Two-tailed test
            'interpretation': self._interpret_morans_i(morans_i, z_score),
            'distance_threshold': distance_threshold
        }
        
        self.results['spatial_autocorrelation'] = autocorr_results
        return autocorr_results
    
    def nearest_neighbor_analysis(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Perform nearest neighbor analysis.
        
        Args:
            gdf: GeoDataFrame with point geometries
            
        Returns:
            Dictionary with nearest neighbor analysis results
        """
        if gdf.empty:
            return {"error": "Empty GeoDataFrame provided"}
            
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        # Calculate nearest neighbor distances
        nn_distances = []
        for i, point in enumerate(coords):
            distances = []
            for j, other_point in enumerate(coords):
                if i != j:
                    dist = np.sqrt(np.sum((point - other_point) ** 2))
                    distances.append(dist)
            if distances:
                nn_distances.append(min(distances))
        
        # Calculate statistics
        mean_nn = np.mean(nn_distances)
        std_nn = np.std(nn_distances)
        
        # Expected nearest neighbor distance for random distribution
        area = gdf.total_bounds
        area_size = (area[2] - area[0]) * (area[3] - area[1])
        expected_nn = 0.5 / np.sqrt(len(gdf) / area_size)
        
        # Nearest neighbor ratio
        nn_ratio = mean_nn / expected_nn if expected_nn > 0 else 0
        
        # Z-score for significance
        se_nn = 0.26136 / np.sqrt(len(gdf) * len(gdf) / area_size)
        z_score = (mean_nn - expected_nn) / se_nn if se_nn > 0 else 0
        
        nn_results = {
            'mean_nn_distance': mean_nn,
            'std_nn_distance': std_nn,
            'expected_nn_distance': expected_nn,
            'nn_ratio': nn_ratio,
            'z_score': z_score,
            'interpretation': self._interpret_nn_ratio(nn_ratio, z_score),
            'area_size': area_size,
            'point_density': len(gdf) / area_size
        }
        
        self.results['nearest_neighbor'] = nn_results
        return nn_results
    
    def _interpret_morans_i(self, morans_i: float, z_score: float) -> str:
        """Interpret Moran's I results."""
        if abs(z_score) < 1.96:
            return "No significant spatial autocorrelation"
        elif morans_i > 0:
            return "Positive spatial autocorrelation (clustering)"
        else:
            return "Negative spatial autocorrelation (dispersion)"
    
    def _interpret_nn_ratio(self, nn_ratio: float, z_score: float) -> str:
        """Interpret nearest neighbor ratio results."""
        if abs(z_score) < 1.96:
            return "Random distribution"
        elif nn_ratio < 1:
            return "Clustered distribution"
        else:
            return "Dispersed distribution"
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all analyses."""
        report = "=== SPATIAL STATISTICS REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['density_grid', 'cluster_labels', 'cluster_centers']:
                        if isinstance(value, float):
                            report += f"{key}: {value:.4f}\n"
                        else:
                            report += f"{key}: {value}\n"
            report += "\n"
        
        return report
