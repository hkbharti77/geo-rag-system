"""
Elevation Processing Module

Provides elevation and terrain analysis capabilities including:
- Digital Elevation Model (DEM) processing
- Slope and aspect calculation
- Terrain roughness analysis
- Viewshed analysis
- Watershed delineation
- Terrain classification
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class ElevationProcessor:
    """Elevation and terrain analysis tools."""
    
    def __init__(self):
        self.results = {}
        
    def process_dem(self, 
                   elevation_data: np.ndarray,
                   cell_size: float = 30.0,
                   no_data_value: float = -9999.0) -> Dict:
        """
        Process Digital Elevation Model (DEM) data.
        
        Args:
            elevation_data: 2D numpy array of elevation values
            cell_size: Size of each cell in meters
            no_data_value: Value representing no data
            
        Returns:
            Dictionary with DEM processing results
        """
        if elevation_data is None or elevation_data.size == 0:
            return {"error": "Invalid elevation data provided"}
            
        # Create a copy and handle no data values
        dem = elevation_data.copy()
        valid_mask = dem != no_data_value
        
        if not np.any(valid_mask):
            return {"error": "No valid elevation data found"}
            
        # Basic statistics
        valid_elevations = dem[valid_mask]
        dem_stats = {
            'min_elevation': np.min(valid_elevations),
            'max_elevation': np.max(valid_elevations),
            'mean_elevation': np.mean(valid_elevations),
            'std_elevation': np.std(valid_elevations),
            'median_elevation': np.median(valid_elevations),
            'cell_size': cell_size,
            'rows': dem.shape[0],
            'columns': dem.shape[1],
            'total_cells': dem.size,
            'valid_cells': np.sum(valid_mask),
            'no_data_cells': np.sum(~valid_mask)
        }
        
        # Elevation distribution
        elevation_bins = np.linspace(dem_stats['min_elevation'], dem_stats['max_elevation'], 20)
        elevation_hist, _ = np.histogram(valid_elevations, bins=elevation_bins)
        
        dem_stats['elevation_distribution'] = {
            'bins': elevation_bins.tolist(),
            'histogram': elevation_hist.tolist()
        }
        
        self.results['dem_processing'] = dem_stats
        return dem_stats
    
    def calculate_slope_aspect(self, 
                             elevation_data: np.ndarray,
                             cell_size: float = 30.0,
                             no_data_value: float = -9999.0) -> Dict:
        """
        Calculate slope and aspect from elevation data.
        
        Args:
            elevation_data: 2D numpy array of elevation values
            cell_size: Size of each cell in meters
            no_data_value: Value representing no data
            
        Returns:
            Dictionary with slope and aspect results
        """
        if elevation_data is None or elevation_data.size == 0:
            return {"error": "Invalid elevation data provided"}
            
        dem = elevation_data.copy()
        valid_mask = dem != no_data_value
        
        if not np.any(valid_mask):
            return {"error": "No valid elevation data found"}
            
        # Calculate gradients using Sobel operators
        sobel_x = ndimage.sobel(dem, axis=1)
        sobel_y = ndimage.sobel(dem, axis=0)
        
        # Calculate slope (in degrees)
        slope_rad = np.arctan(np.sqrt(sobel_x**2 + sobel_y**2) / (2 * cell_size))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect (in degrees, 0-360)
        aspect_rad = np.arctan2(sobel_y, sobel_x)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360  # Convert to 0-360 range
        
        # Handle no data values
        slope_deg[~valid_mask] = no_data_value
        aspect_deg[~valid_mask] = no_data_value
        
        # Calculate statistics
        valid_slope = slope_deg[valid_mask]
        valid_aspect = aspect_deg[valid_mask]
        
        slope_stats = {
            'min_slope': np.min(valid_slope),
            'max_slope': np.max(valid_slope),
            'mean_slope': np.mean(valid_slope),
            'std_slope': np.std(valid_slope),
            'median_slope': np.median(valid_slope)
        }
        
        aspect_stats = {
            'mean_aspect': np.mean(valid_aspect),
            'std_aspect': np.std(valid_aspect),
            'aspect_direction': self._classify_aspect_direction(valid_aspect)
        }
        
        # Slope classification
        slope_classes = self._classify_slopes(valid_slope)
        
        results = {
            'slope_data': slope_deg,
            'aspect_data': aspect_deg,
            'slope_statistics': slope_stats,
            'aspect_statistics': aspect_stats,
            'slope_classification': slope_classes,
            'cell_size': cell_size
        }
        
        self.results['slope_aspect'] = results
        return results
    
    def calculate_terrain_roughness(self, 
                                  elevation_data: np.ndarray,
                                  window_size: int = 3,
                                  no_data_value: float = -9999.0) -> Dict:
        """
        Calculate terrain roughness using various metrics.
        
        Args:
            elevation_data: 2D numpy array of elevation values
            window_size: Size of the analysis window
            no_data_value: Value representing no data
            
        Returns:
            Dictionary with terrain roughness results
        """
        if elevation_data is None or elevation_data.size == 0:
            return {"error": "Invalid elevation data provided"}
            
        dem = elevation_data.copy()
        valid_mask = dem != no_data_value
        
        if not np.any(valid_mask):
            return {"error": "No valid elevation data found"}
            
        # Calculate different roughness metrics
        
        # 1. Standard deviation within moving window
        roughness_std = ndimage.generic_filter(
            dem, np.std, size=window_size, mode='constant', cval=no_data_value
        )
        
        # 2. Range within moving window
        def range_filter(arr):
            valid_vals = arr[arr != no_data_value]
            return np.max(valid_vals) - np.min(valid_vals) if len(valid_vals) > 0 else no_data_value
        
        roughness_range = ndimage.generic_filter(
            dem, range_filter, size=window_size, mode='constant', cval=no_data_value
        )
        
        # 3. Terrain Ruggedness Index (TRI)
        tri = self._calculate_tri(dem, window_size, no_data_value)
        
        # 4. Vector Ruggedness Measure (VRM)
        vrm = self._calculate_vrm(dem, window_size, no_data_value)
        
        # Calculate statistics
        valid_std = roughness_std[valid_mask]
        valid_range = roughness_range[valid_mask]
        valid_tri = tri[valid_mask]
        valid_vrm = vrm[valid_mask]
        
        roughness_stats = {
            'std_roughness': {
                'mean': np.mean(valid_std),
                'std': np.std(valid_std),
                'min': np.min(valid_std),
                'max': np.max(valid_std)
            },
            'range_roughness': {
                'mean': np.mean(valid_range),
                'std': np.std(valid_range),
                'min': np.min(valid_range),
                'max': np.max(valid_range)
            },
            'tri': {
                'mean': np.mean(valid_tri),
                'std': np.std(valid_tri),
                'min': np.min(valid_tri),
                'max': np.max(valid_tri)
            },
            'vrm': {
                'mean': np.mean(valid_vrm),
                'std': np.std(valid_vrm),
                'min': np.min(valid_vrm),
                'max': np.max(valid_vrm)
            }
        }
        
        results = {
            'roughness_std': roughness_std,
            'roughness_range': roughness_range,
            'tri': tri,
            'vrm': vrm,
            'roughness_statistics': roughness_stats,
            'window_size': window_size
        }
        
        self.results['terrain_roughness'] = results
        return results
    
    def _calculate_tri(self, dem: np.ndarray, window_size: int, no_data_value: float) -> np.ndarray:
        """Calculate Terrain Ruggedness Index."""
        def tri_filter(arr):
            center = arr[len(arr)//2]
            if center == no_data_value:
                return no_data_value
            valid_vals = arr[arr != no_data_value]
            if len(valid_vals) == 0:
                return no_data_value
            return np.sqrt(np.sum((valid_vals - center) ** 2))
        
        return ndimage.generic_filter(
            dem, tri_filter, size=window_size, mode='constant', cval=no_data_value
        )
    
    def _calculate_vrm(self, dem: np.ndarray, window_size: int, no_data_value: float) -> np.ndarray:
        """Calculate Vector Ruggedness Measure."""
        def vrm_filter(arr):
            center = arr[len(arr)//2]
            if center == no_data_value:
                return no_data_value
            valid_vals = arr[arr != no_data_value]
            if len(valid_vals) < 2:
                return no_data_value
            
            # Calculate unit vectors
            vectors = []
            for val in valid_vals:
                if val != center:
                    diff = val - center
                    magnitude = abs(diff)
                    if magnitude > 0:
                        vectors.append(diff / magnitude)
            
            if len(vectors) == 0:
                return no_data_value
            
            # Calculate vector sum
            vector_sum = np.sum(vectors)
            vector_magnitude = np.sqrt(np.sum(vector_sum ** 2))
            
            # Calculate VRM
            n = len(vectors)
            vrm = 1 - (vector_magnitude / n)
            
            return vrm
        
        return ndimage.generic_filter(
            dem, vrm_filter, size=window_size, mode='constant', cval=no_data_value
        )
    
    def viewshed_analysis(self, 
                         elevation_data: np.ndarray,
                         viewpoint: Tuple[int, int],
                         cell_size: float = 30.0,
                         max_distance: float = 5000.0,
                         no_data_value: float = -9999.0) -> Dict:
        """
        Perform viewshed analysis from a given viewpoint.
        
        Args:
            elevation_data: 2D numpy array of elevation values
            viewpoint: Tuple of (row, col) coordinates for viewpoint
            cell_size: Size of each cell in meters
            max_distance: Maximum distance to analyze in meters
            no_data_value: Value representing no data
            
        Returns:
            Dictionary with viewshed analysis results
        """
        if elevation_data is None or elevation_data.size == 0:
            return {"error": "Invalid elevation data provided"}
            
        dem = elevation_data.copy()
        rows, cols = dem.shape
        
        # Validate viewpoint
        vp_row, vp_col = viewpoint
        if not (0 <= vp_row < rows and 0 <= vp_col < cols):
            return {"error": "Invalid viewpoint coordinates"}
            
        if dem[vp_row, vp_col] == no_data_value:
            return {"error": "Viewpoint is in no-data area"}
            
        # Initialize viewshed
        viewshed = np.zeros_like(dem, dtype=bool)
        viewshed[vp_row, vp_col] = True  # Viewpoint is always visible
        
        viewpoint_elevation = dem[vp_row, vp_col]
        max_cells = int(max_distance / cell_size)
        
        # Analyze visibility in all directions
        for angle in range(0, 360, 1):  # 1-degree increments
            rad = np.radians(angle)
            cos_angle = np.cos(rad)
            sin_angle = np.sin(rad)
            
            max_angle = -np.inf
            
            for distance in range(1, max_cells + 1):
                # Calculate target cell coordinates
                target_row = int(vp_row + distance * sin_angle)
                target_col = int(vp_col + distance * cos_angle)
                
                # Check bounds
                if not (0 <= target_row < rows and 0 <= target_col < cols):
                    break
                
                # Check if target is no data
                if dem[target_row, target_col] == no_data_value:
                    break
                
                # Calculate elevation angle to target
                target_elevation = dem[target_row, target_col]
                actual_distance = distance * cell_size
                
                if actual_distance > 0:
                    elevation_angle = np.arctan2(target_elevation - viewpoint_elevation, actual_distance)
                    
                    # Check if target is visible
                    if elevation_angle > max_angle:
                        viewshed[target_row, target_col] = True
                        max_angle = elevation_angle
        
        # Calculate statistics
        total_cells = np.sum(dem != no_data_value)
        visible_cells = np.sum(viewshed)
        visibility_percentage = (visible_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate visible area
        visible_area = visible_cells * (cell_size ** 2)
        
        viewshed_stats = {
            'viewpoint': viewpoint,
            'viewpoint_elevation': viewpoint_elevation,
            'total_cells': total_cells,
            'visible_cells': visible_cells,
            'visibility_percentage': visibility_percentage,
            'visible_area_m2': visible_area,
            'max_distance': max_distance,
            'cell_size': cell_size
        }
        
        results = {
            'viewshed': viewshed,
            'viewshed_statistics': viewshed_stats
        }
        
        self.results['viewshed_analysis'] = results
        return results
    
    def watershed_delineation(self, 
                            elevation_data: np.ndarray,
                            pour_point: Tuple[int, int],
                            cell_size: float = 30.0,
                            no_data_value: float = -9999.0) -> Dict:
        """
        Perform watershed delineation from a pour point.
        
        Args:
            elevation_data: 2D numpy array of elevation values
            pour_point: Tuple of (row, col) coordinates for pour point
            cell_size: Size of each cell in meters
            no_data_value: Value representing no data
            
        Returns:
            Dictionary with watershed delineation results
        """
        if elevation_data is None or elevation_data.size == 0:
            return {"error": "Invalid elevation data provided"}
            
        dem = elevation_data.copy()
        rows, cols = dem.shape
        
        # Validate pour point
        pp_row, pp_col = pour_point
        if not (0 <= pp_row < rows and 0 <= pp_col < cols):
            return {"error": "Invalid pour point coordinates"}
            
        if dem[pp_row, pp_col] == no_data_value:
            return {"error": "Pour point is in no-data area"}
        
        # Simple watershed delineation using flow direction
        watershed = np.zeros_like(dem, dtype=bool)
        watershed[pp_row, pp_col] = True
        
        # Flow direction: 8-connectivity
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Iterative watershed expansion
        changed = True
        iteration = 0
        max_iterations = rows * cols  # Safety limit
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for row in range(rows):
                for col in range(cols):
                    if dem[row, col] != no_data_value and not watershed[row, col]:
                        # Check if any neighbor flows to this cell
                        for dr, dc in directions:
                            nr, nc = row + dr, col + dc
                            if (0 <= nr < rows and 0 <= nc < cols and 
                                watershed[nr, nc] and dem[nr, nc] != no_data_value):
                                
                                # Check if this cell is the lowest neighbor
                                is_lowest = True
                                for dr2, dc2 in directions:
                                    nr2, nc2 = nr + dr2, nc + dc2
                                    if (0 <= nr2 < rows and 0 <= nc2 < cols and 
                                        dem[nr2, nc2] != no_data_value and
                                        dem[nr2, nc2] < dem[nr, nc]):
                                        is_lowest = False
                                        break
                                
                                if is_lowest:
                                    watershed[row, col] = True
                                    changed = True
                                    break
        
        # Calculate watershed statistics
        total_cells = np.sum(dem != no_data_value)
        watershed_cells = np.sum(watershed)
        watershed_percentage = (watershed_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate watershed area
        watershed_area = watershed_cells * (cell_size ** 2)
        
        # Calculate watershed elevation statistics
        watershed_elevations = dem[watershed]
        watershed_elevations = watershed_elevations[watershed_elevations != no_data_value]
        
        elevation_stats = {
            'min_elevation': np.min(watershed_elevations) if len(watershed_elevations) > 0 else 0,
            'max_elevation': np.max(watershed_elevations) if len(watershed_elevations) > 0 else 0,
            'mean_elevation': np.mean(watershed_elevations) if len(watershed_elevations) > 0 else 0,
            'std_elevation': np.std(watershed_elevations) if len(watershed_elevations) > 0 else 0
        }
        
        watershed_stats = {
            'pour_point': pour_point,
            'total_cells': total_cells,
            'watershed_cells': watershed_cells,
            'watershed_percentage': watershed_percentage,
            'watershed_area_m2': watershed_area,
            'elevation_statistics': elevation_stats,
            'iterations': iteration,
            'cell_size': cell_size
        }
        
        results = {
            'watershed': watershed,
            'watershed_statistics': watershed_stats
        }
        
        self.results['watershed_delineation'] = results
        return results
    
    def _classify_aspect_direction(self, aspects: np.ndarray) -> Dict:
        """Classify aspect directions."""
        directions = {
            'North': np.sum((aspects >= 315) | (aspects < 45)),
            'Northeast': np.sum((aspects >= 45) & (aspects < 90)),
            'East': np.sum((aspects >= 90) & (aspects < 135)),
            'Southeast': np.sum((aspects >= 135) & (aspects < 180)),
            'South': np.sum((aspects >= 180) & (aspects < 225)),
            'Southwest': np.sum((aspects >= 225) & (aspects < 270)),
            'West': np.sum((aspects >= 270) & (aspects < 315))
        }
        
        total = len(aspects)
        if total > 0:
            directions = {k: (v, v/total*100) for k, v in directions.items()}
        
        return directions
    
    def _classify_slopes(self, slopes: np.ndarray) -> Dict:
        """Classify slopes into categories."""
        classifications = {
            'Flat (0-2°)': np.sum(slopes < 2),
            'Gentle (2-5°)': np.sum((slopes >= 2) & (slopes < 5)),
            'Moderate (5-10°)': np.sum((slopes >= 5) & (slopes < 10)),
            'Steep (10-20°)': np.sum((slopes >= 10) & (slopes < 20)),
            'Very Steep (20-30°)': np.sum((slopes >= 20) & (slopes < 30)),
            'Extreme (>30°)': np.sum(slopes >= 30)
        }
        
        total = len(slopes)
        if total > 0:
            classifications = {k: (v, v/total*100) for k, v in classifications.items()}
        
        return classifications
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all elevation analyses."""
        report = "=== ELEVATION PROCESSING REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['elevation_data', 'slope_data', 'aspect_data', 'viewshed', 'watershed']:
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
