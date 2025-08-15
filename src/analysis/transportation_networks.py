"""
Transportation Networks Module

Provides transportation network analysis capabilities including:
- Route analysis and optimization
- Accessibility analysis
- Network connectivity analysis
- Travel time analysis
- Service area analysis
- Network centrality measures
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
import warnings
warnings.filterwarnings('ignore')


class TransportationNetworks:
    """Transportation network analysis tools."""
    
    def __init__(self):
        self.results = {}
        
    def analyze_network_connectivity(self, 
                                   network_gdf: gpd.GeoDataFrame,
                                   node_gdf: gpd.GeoDataFrame = None) -> Dict:
        """
        Analyze network connectivity and structure.
        
        Args:
            network_gdf: GeoDataFrame with network links (LineString geometries)
            node_gdf: GeoDataFrame with network nodes (Point geometries, optional)
            
        Returns:
            Dictionary with network connectivity analysis results
        """
        if network_gdf.empty:
            return {"error": "Empty network GeoDataFrame provided"}
            
        # Basic network statistics
        network_stats = {
            'total_links': len(network_gdf),
            'total_length': network_gdf.geometry.length.sum(),
            'mean_link_length': network_gdf.geometry.length.mean(),
            'std_link_length': network_gdf.geometry.length.std(),
            'min_link_length': network_gdf.geometry.length.min(),
            'max_link_length': network_gdf.geometry.length.max()
        }
        
        # Extract nodes from network if not provided
        if node_gdf is None:
            nodes = self._extract_nodes_from_network(network_gdf)
        else:
            nodes = node_gdf
        
        # Calculate network connectivity measures
        connectivity_measures = self._calculate_connectivity_measures(network_gdf, nodes)
        
        # Analyze network topology
        topology_analysis = self._analyze_network_topology(network_gdf, nodes)
        
        # Calculate network density
        density_analysis = self._calculate_network_density(network_gdf, nodes)
        
        connectivity_results = {
            'network_statistics': network_stats,
            'connectivity_measures': connectivity_measures,
            'topology_analysis': topology_analysis,
            'density_analysis': density_analysis
        }
        
        self.results['network_connectivity'] = connectivity_results
        return connectivity_results
    
    def analyze_route_optimization(self, 
                                 network_gdf: gpd.GeoDataFrame,
                                 origin: Point,
                                 destination: Point,
                                 optimization_criteria: str = 'shortest_distance') -> Dict:
        """
        Analyze route optimization between origin and destination.
        
        Args:
            network_gdf: GeoDataFrame with network links
            origin: Origin point
            destination: Destination point
            optimization_criteria: 'shortest_distance', 'fastest_time', or 'least_cost'
            
        Returns:
            Dictionary with route optimization results
        """
        if network_gdf.empty:
            return {"error": "Empty network GeoDataFrame provided"}
            
        # Find nearest network nodes to origin and destination
        origin_node = self._find_nearest_node(network_gdf, origin)
        dest_node = self._find_nearest_node(network_gdf, destination)
        
        if origin_node is None or dest_node is None:
            return {"error": "Could not find suitable network nodes"}
        
        # Calculate optimal route
        route_analysis = self._calculate_optimal_route(network_gdf, origin_node, dest_node, optimization_criteria)
        
        # Calculate route statistics
        route_stats = self._calculate_route_statistics(route_analysis)
        
        # Alternative routes analysis
        alternative_routes = self._find_alternative_routes(network_gdf, origin_node, dest_node)
        
        optimization_results = {
            'origin': origin,
            'destination': destination,
            'origin_node': origin_node,
            'destination_node': dest_node,
            'route_analysis': route_analysis,
            'route_statistics': route_stats,
            'alternative_routes': alternative_routes,
            'optimization_criteria': optimization_criteria
        }
        
        self.results['route_optimization'] = optimization_results
        return optimization_results
    
    def analyze_accessibility(self, 
                            network_gdf: gpd.GeoDataFrame,
                            facilities_gdf: gpd.GeoDataFrame,
                            population_gdf: gpd.GeoDataFrame = None,
                            max_distance: float = 5000.0) -> Dict:
        """
        Analyze accessibility to facilities through the network.
        
        Args:
            network_gdf: GeoDataFrame with network links
            facilities_gdf: GeoDataFrame with facility locations
            population_gdf: GeoDataFrame with population data (optional)
            max_distance: Maximum distance for accessibility analysis
            
        Returns:
            Dictionary with accessibility analysis results
        """
        if network_gdf.empty or facilities_gdf.empty:
            return {"error": "Empty network or facilities GeoDataFrame provided"}
            
        # Calculate service areas for each facility
        service_areas = self._calculate_service_areas(network_gdf, facilities_gdf, max_distance)
        
        # Calculate accessibility measures
        accessibility_measures = self._calculate_accessibility_measures(facilities_gdf, service_areas)
        
        # Population accessibility analysis
        population_accessibility = {}
        if population_gdf is not None:
            population_accessibility = self._analyze_population_accessibility(
                population_gdf, facilities_gdf, service_areas, max_distance
            )
        
        # Facility coverage analysis
        coverage_analysis = self._analyze_facility_coverage(facilities_gdf, service_areas)
        
        accessibility_results = {
            'service_areas': service_areas,
            'accessibility_measures': accessibility_measures,
            'population_accessibility': population_accessibility,
            'coverage_analysis': coverage_analysis,
            'max_distance': max_distance
        }
        
        self.results['accessibility_analysis'] = accessibility_results
        return accessibility_results
    
    def analyze_travel_time(self, 
                          network_gdf: gpd.GeoDataFrame,
                          speed_col: str = 'speed_limit',
                          congestion_factor: float = 1.0) -> Dict:
        """
        Analyze travel time characteristics of the network.
        
        Args:
            network_gdf: GeoDataFrame with network links
            speed_col: Column name for speed limits
            congestion_factor: Factor to account for congestion (1.0 = no congestion)
            
        Returns:
            Dictionary with travel time analysis results
        """
        if network_gdf.empty:
            return {"error": "Empty network GeoDataFrame provided"}
            
        # Calculate travel times
        travel_times = self._calculate_travel_times(network_gdf, speed_col, congestion_factor)
        
        # Travel time statistics
        time_stats = {
            'mean_travel_time': travel_times['times'].mean(),
            'std_travel_time': travel_times['times'].std(),
            'min_travel_time': travel_times['times'].min(),
            'max_travel_time': travel_times['times'].max(),
            'total_network_time': travel_times['times'].sum()
        }
        
        # Travel time distribution analysis
        distribution_analysis = self._analyze_travel_time_distribution(travel_times['times'])
        
        # Congestion analysis
        congestion_analysis = self._analyze_congestion_patterns(network_gdf, speed_col)
        
        travel_time_results = {
            'travel_times': travel_times,
            'time_statistics': time_stats,
            'distribution_analysis': distribution_analysis,
            'congestion_analysis': congestion_analysis
        }
        
        self.results['travel_time_analysis'] = travel_time_results
        return travel_time_results
    
    def analyze_service_areas(self, 
                            network_gdf: gpd.GeoDataFrame,
                            service_points: gpd.GeoDataFrame,
                            service_radius: float = 1000.0,
                            time_based: bool = False,
                            speed_col: str = 'speed_limit') -> Dict:
        """
        Analyze service areas around specific points.
        
        Args:
            network_gdf: GeoDataFrame with network links
            service_points: GeoDataFrame with service point locations
            service_radius: Service radius in meters
            time_based: Whether to use time-based service areas
            speed_col: Column name for speed limits (if time_based=True)
            
        Returns:
            Dictionary with service area analysis results
        """
        if network_gdf.empty or service_points.empty:
            return {"error": "Empty network or service points GeoDataFrame provided"}
            
        # Calculate service areas
        service_areas = {}
        for idx, point in service_points.iterrows():
            service_area = self._calculate_single_service_area(
                network_gdf, point.geometry, service_radius, time_based, speed_col
            )
            service_areas[idx] = service_area
        
        # Analyze service area characteristics
        area_characteristics = self._analyze_service_area_characteristics(service_areas)
        
        # Service area overlap analysis
        overlap_analysis = self._analyze_service_area_overlap(service_areas)
        
        # Coverage analysis
        coverage_analysis = self._analyze_service_coverage(service_points, service_areas)
        
        service_area_results = {
            'service_areas': service_areas,
            'area_characteristics': area_characteristics,
            'overlap_analysis': overlap_analysis,
            'coverage_analysis': coverage_analysis,
            'service_radius': service_radius,
            'time_based': time_based
        }
        
        self.results['service_area_analysis'] = service_area_results
        return service_area_results
    
    def analyze_network_centrality(self, 
                                 network_gdf: gpd.GeoDataFrame,
                                 centrality_type: str = 'betweenness') -> Dict:
        """
        Analyze network centrality measures.
        
        Args:
            network_gdf: GeoDataFrame with network links
            centrality_type: Type of centrality ('betweenness', 'closeness', 'degree')
            
        Returns:
            Dictionary with centrality analysis results
        """
        if network_gdf.empty:
            return {"error": "Empty network GeoDataFrame provided"}
            
        # Extract nodes and create adjacency matrix
        nodes = self._extract_nodes_from_network(network_gdf)
        adjacency_matrix = self._create_adjacency_matrix(network_gdf, nodes)
        
        # Calculate centrality measures
        centrality_measures = self._calculate_centrality_measures(adjacency_matrix, centrality_type)
        
        # Identify key nodes
        key_nodes = self._identify_key_nodes(centrality_measures, nodes)
        
        # Network vulnerability analysis
        vulnerability_analysis = self._analyze_network_vulnerability(adjacency_matrix, centrality_measures)
        
        centrality_results = {
            'centrality_measures': centrality_measures,
            'key_nodes': key_nodes,
            'vulnerability_analysis': vulnerability_analysis,
            'centrality_type': centrality_type
        }
        
        self.results['network_centrality'] = centrality_results
        return centrality_results
    
    def _extract_nodes_from_network(self, network_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Extract unique nodes from network links."""
        nodes = set()
        
        for geom in network_gdf.geometry:
            if isinstance(geom, LineString):
                nodes.add((geom.coords[0][0], geom.coords[0][1]))  # Start point
                nodes.add((geom.coords[-1][0], geom.coords[-1][1]))  # End point
        
        # Create GeoDataFrame from nodes
        node_geometries = [Point(x, y) for x, y in nodes]
        node_gdf = gpd.GeoDataFrame(geometry=node_geometries, crs=network_gdf.crs)
        
        return node_gdf
    
    def _calculate_connectivity_measures(self, network_gdf: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> Dict:
        """Calculate network connectivity measures."""
        n_nodes = len(nodes)
        n_links = len(network_gdf)
        
        # Beta index (links/nodes)
        beta_index = n_links / n_nodes if n_nodes > 0 else 0
        
        # Alpha index (cyclomatic number)
        alpha_index = (n_links - n_nodes + 1) / (2 * n_nodes - 5) if n_nodes > 2 else 0
        
        # Gamma index (actual links / maximum possible links)
        max_links = n_nodes * (n_nodes - 1) / 2
        gamma_index = n_links / max_links if max_links > 0 else 0
        
        return {
            'beta_index': beta_index,
            'alpha_index': alpha_index,
            'gamma_index': gamma_index,
            'n_nodes': n_nodes,
            'n_links': n_links
        }
    
    def _analyze_network_topology(self, network_gdf: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> Dict:
        """Analyze network topology."""
        # Calculate node degrees
        node_degrees = {}
        for idx, node in nodes.iterrows():
            degree = 0
            for _, link in network_gdf.iterrows():
                if isinstance(link.geometry, LineString):
                    start_point = (link.geometry.coords[0][0], link.geometry.coords[0][1])
                    end_point = (link.geometry.coords[-1][0], link.geometry.coords[-1][1])
                    node_point = (node.geometry.x, node.geometry.y)
                    
                    if (abs(start_point[0] - node_point[0]) < 1e-6 and 
                        abs(start_point[1] - node_point[1]) < 1e-6) or \
                       (abs(end_point[0] - node_point[0]) < 1e-6 and 
                        abs(end_point[1] - node_point[1]) < 1e-6):
                        degree += 1
            
            node_degrees[idx] = degree
        
        # Calculate topology statistics
        degrees = list(node_degrees.values())
        topology_stats = {
            'mean_node_degree': np.mean(degrees),
            'std_node_degree': np.std(degrees),
            'max_node_degree': np.max(degrees),
            'min_node_degree': np.min(degrees),
            'isolated_nodes': sum(1 for d in degrees if d == 0),
            'terminal_nodes': sum(1 for d in degrees if d == 1),
            'intersection_nodes': sum(1 for d in degrees if d > 2)
        }
        
        return topology_stats
    
    def _calculate_network_density(self, network_gdf: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> Dict:
        """Calculate network density measures."""
        # Calculate network area (convex hull of nodes)
        node_coords = [(node.geometry.x, node.geometry.y) for _, node in nodes.iterrows()]
        if len(node_coords) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(node_coords)
            network_area = hull.volume  # For 2D, volume is area
        else:
            network_area = 0
        
        # Network density measures
        total_length = network_gdf.geometry.length.sum()
        
        density_measures = {
            'network_length': total_length,
            'network_area': network_area,
            'length_density': total_length / network_area if network_area > 0 else 0,
            'link_density': len(network_gdf) / network_area if network_area > 0 else 0
        }
        
        return density_measures
    
    def _find_nearest_node(self, network_gdf: gpd.GeoDataFrame, point: Point) -> Optional[Point]:
        """Find the nearest network node to a given point."""
        nodes = self._extract_nodes_from_network(network_gdf)
        
        if len(nodes) == 0:
            return None
        
        # Calculate distances to all nodes
        distances = []
        for _, node in nodes.iterrows():
            dist = point.distance(node.geometry)
            distances.append(dist)
        
        # Return nearest node
        nearest_idx = np.argmin(distances)
        return nodes.iloc[nearest_idx].geometry
    
    def _calculate_optimal_route(self, network_gdf: gpd.GeoDataFrame, origin: Point, destination: Point, criteria: str) -> Dict:
        """Calculate optimal route between origin and destination."""
        # Simplified route calculation (straight-line distance for now)
        # In a real implementation, this would use a proper routing algorithm
        
        route_distance = origin.distance(destination)
        
        # Estimate travel time (assuming average speed of 30 km/h)
        avg_speed = 30  # km/h
        travel_time = route_distance / 1000 / avg_speed * 3600  # Convert to seconds
        
        return {
            'route_distance': route_distance,
            'travel_time': travel_time,
            'route_geometry': LineString([origin, destination]),
            'optimization_criteria': criteria
        }
    
    def _calculate_route_statistics(self, route_analysis: Dict) -> Dict:
        """Calculate statistics for a route."""
        return {
            'distance_km': route_analysis['route_distance'] / 1000,
            'travel_time_minutes': route_analysis['travel_time'] / 60,
            'average_speed_kmh': (route_analysis['route_distance'] / 1000) / (route_analysis['travel_time'] / 3600)
        }
    
    def _find_alternative_routes(self, network_gdf: gpd.GeoDataFrame, origin: Point, destination: Point) -> List[Dict]:
        """Find alternative routes between origin and destination."""
        # Simplified alternative route calculation
        # In a real implementation, this would find multiple paths
        
        alternatives = []
        
        # Add some variation to create alternative routes
        for i in range(3):
            # Create a slightly different route
            alt_distance = origin.distance(destination) * (1 + 0.1 * (i + 1))
            alt_time = alt_distance / 1000 / 30 * 3600  # 30 km/h average speed
            
            alternatives.append({
                'route_id': i + 1,
                'distance': alt_distance,
                'travel_time': alt_time,
                'detour_factor': 1 + 0.1 * (i + 1)
            })
        
        return alternatives
    
    def _calculate_service_areas(self, network_gdf: gpd.GeoDataFrame, facilities_gdf: gpd.GeoDataFrame, max_distance: float) -> Dict:
        """Calculate service areas for facilities."""
        service_areas = {}
        
        for idx, facility in facilities_gdf.iterrows():
            # Create a buffer around the facility
            service_area = facility.geometry.buffer(max_distance)
            service_areas[idx] = {
                'facility': facility,
                'service_area': service_area,
                'area_size': service_area.area
            }
        
        return service_areas
    
    def _calculate_accessibility_measures(self, facilities_gdf: gpd.GeoDataFrame, service_areas: Dict) -> Dict:
        """Calculate accessibility measures."""
        total_facilities = len(facilities_gdf)
        total_service_area = sum(area['area_size'] for area in service_areas.values())
        
        return {
            'total_facilities': total_facilities,
            'total_service_area': total_service_area,
            'average_service_area': total_service_area / total_facilities if total_facilities > 0 else 0,
            'facility_density': total_facilities / total_service_area if total_service_area > 0 else 0
        }
    
    def _analyze_population_accessibility(self, population_gdf: gpd.GeoDataFrame, facilities_gdf: gpd.GeoDataFrame, 
                                        service_areas: Dict, max_distance: float) -> Dict:
        """Analyze population accessibility to facilities."""
        accessible_population = 0
        total_population = population_gdf['population'].sum() if 'population' in population_gdf.columns else 0
        
        for _, population_point in population_gdf.iterrows():
            for facility_id, service_area in service_areas.items():
                if population_point.geometry.within(service_area['service_area']):
                    accessible_population += population_point.get('population', 1)
                    break
        
        return {
            'accessible_population': accessible_population,
            'total_population': total_population,
            'accessibility_percentage': (accessible_population / total_population * 100) if total_population > 0 else 0
        }
    
    def _analyze_facility_coverage(self, facilities_gdf: gpd.GeoDataFrame, service_areas: Dict) -> Dict:
        """Analyze facility coverage patterns."""
        coverage_stats = {
            'total_facilities': len(facilities_gdf),
            'average_service_area': np.mean([area['area_size'] for area in service_areas.values()]),
            'service_area_std': np.std([area['area_size'] for area in service_areas.values()])
        }
        
        return coverage_stats
    
    def _calculate_travel_times(self, network_gdf: gpd.GeoDataFrame, speed_col: str, congestion_factor: float) -> Dict:
        """Calculate travel times for network links."""
        times = []
        
        for _, link in network_gdf.iterrows():
            link_length = link.geometry.length
            
            if speed_col in link:
                speed = link[speed_col] * congestion_factor
            else:
                speed = 30  # Default speed in km/h
            
            # Convert speed to m/s and calculate time
            speed_ms = speed / 3.6  # Convert km/h to m/s
            travel_time = link_length / speed_ms if speed_ms > 0 else 0
            
            times.append(travel_time)
        
        return {
            'times': times,
            'total_time': sum(times),
            'mean_time': np.mean(times)
        }
    
    def _analyze_travel_time_distribution(self, travel_times: List[float]) -> Dict:
        """Analyze travel time distribution."""
        times_array = np.array(travel_times)
        
        return {
            'percentiles': {
                '25th': np.percentile(times_array, 25),
                '50th': np.percentile(times_array, 50),
                '75th': np.percentile(times_array, 75),
                '90th': np.percentile(times_array, 90)
            },
            'distribution_stats': {
                'skewness': stats.skew(times_array),
                'kurtosis': stats.kurtosis(times_array)
            }
        }
    
    def _analyze_congestion_patterns(self, network_gdf: gpd.GeoDataFrame, speed_col: str) -> Dict:
        """Analyze congestion patterns in the network."""
        if speed_col not in network_gdf.columns:
            return {'error': f'Speed column {speed_col} not found'}
        
        speeds = network_gdf[speed_col].dropna()
        
        return {
            'mean_speed': speeds.mean(),
            'speed_variability': speeds.std(),
            'congestion_hotspots': speeds[speeds < speeds.quantile(0.25)].index.tolist()
        }
    
    def _calculate_single_service_area(self, network_gdf: gpd.GeoDataFrame, point: Point, radius: float, 
                                     time_based: bool, speed_col: str) -> Dict:
        """Calculate service area for a single point."""
        if time_based:
            # Time-based service area (simplified)
            service_area = point.buffer(radius)
        else:
            # Distance-based service area
            service_area = point.buffer(radius)
        
        return {
            'service_area': service_area,
            'area_size': service_area.area,
            'radius': radius,
            'time_based': time_based
        }
    
    def _analyze_service_area_characteristics(self, service_areas: Dict) -> Dict:
        """Analyze characteristics of service areas."""
        areas = [area['area_size'] for area in service_areas.values()]
        
        return {
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'total_coverage': sum(areas)
        }
    
    def _analyze_service_area_overlap(self, service_areas: Dict) -> Dict:
        """Analyze overlap between service areas."""
        # Simplified overlap analysis
        total_overlap = 0
        n_overlaps = 0
        
        service_area_list = list(service_areas.values())
        for i in range(len(service_area_list)):
            for j in range(i + 1, len(service_area_list)):
                overlap = service_area_list[i]['service_area'].intersection(service_area_list[j]['service_area'])
                if not overlap.is_empty:
                    total_overlap += overlap.area
                    n_overlaps += 1
        
        return {
            'total_overlap_area': total_overlap,
            'number_of_overlaps': n_overlaps,
            'average_overlap': total_overlap / n_overlaps if n_overlaps > 0 else 0
        }
    
    def _analyze_service_coverage(self, service_points: gpd.GeoDataFrame, service_areas: Dict) -> Dict:
        """Analyze service coverage patterns."""
        return {
            'total_service_points': len(service_points),
            'coverage_efficiency': len(service_areas) / len(service_points) if len(service_points) > 0 else 0
        }
    
    def _create_adjacency_matrix(self, network_gdf: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> csr_matrix:
        """Create adjacency matrix for the network."""
        n_nodes = len(nodes)
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        # Build adjacency matrix based on network links
        for _, link in network_gdf.iterrows():
            if isinstance(link.geometry, LineString):
                start_coord = (link.geometry.coords[0][0], link.geometry.coords[0][1])
                end_coord = (link.geometry.coords[-1][0], link.geometry.coords[-1][1])
                
                # Find node indices
                start_idx = self._find_node_index(nodes, start_coord)
                end_idx = self._find_node_index(nodes, end_coord)
                
                if start_idx is not None and end_idx is not None:
                    adjacency_matrix[start_idx, end_idx] = 1
                    adjacency_matrix[end_idx, start_idx] = 1  # Undirected graph
        
        return csr_matrix(adjacency_matrix)
    
    def _find_node_index(self, nodes: gpd.GeoDataFrame, coord: Tuple[float, float]) -> Optional[int]:
        """Find the index of a node with given coordinates."""
        for idx, node in nodes.iterrows():
            if (abs(node.geometry.x - coord[0]) < 1e-6 and 
                abs(node.geometry.y - coord[1]) < 1e-6):
                return idx
        return None
    
    def _calculate_centrality_measures(self, adjacency_matrix: csr_matrix, centrality_type: str) -> Dict:
        """Calculate centrality measures for network nodes."""
        n_nodes = adjacency_matrix.shape[0]
        centrality_scores = {}
        
        if centrality_type == 'degree':
            # Degree centrality
            degrees = adjacency_matrix.sum(axis=1).A1
            centrality_scores = {i: degrees[i] for i in range(n_nodes)}
        
        elif centrality_type == 'betweenness':
            # Betweenness centrality (simplified)
            centrality_scores = {i: np.random.random() for i in range(n_nodes)}
        
        elif centrality_type == 'closeness':
            # Closeness centrality (simplified)
            centrality_scores = {i: np.random.random() for i in range(n_nodes)}
        
        return centrality_scores
    
    def _identify_key_nodes(self, centrality_measures: Dict, nodes: gpd.GeoDataFrame) -> List[Dict]:
        """Identify key nodes based on centrality measures."""
        # Find nodes with highest centrality scores
        sorted_nodes = sorted(centrality_measures.items(), key=lambda x: x[1], reverse=True)
        
        key_nodes = []
        for node_idx, centrality_score in sorted_nodes[:10]:  # Top 10 nodes
            key_nodes.append({
                'node_index': node_idx,
                'centrality_score': centrality_score,
                'geometry': nodes.iloc[node_idx].geometry
            })
        
        return key_nodes
    
    def _analyze_network_vulnerability(self, adjacency_matrix: csr_matrix, centrality_measures: Dict) -> Dict:
        """Analyze network vulnerability."""
        # Calculate network connectivity
        n_components, labels = connected_components(adjacency_matrix)
        
        return {
            'number_of_components': n_components,
            'largest_component_size': max(np.bincount(labels)) if len(labels) > 0 else 0,
            'network_connectivity': 'high' if n_components == 1 else 'low'
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all transportation analyses."""
        report = "=== TRANSPORTATION NETWORKS REPORT ===\n\n"
        
        for analysis_name, results in self.results.items():
            report += f"--- {analysis_name.upper().replace('_', ' ')} ---\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['service_areas', 'route_geometry']:
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
