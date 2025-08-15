import json
import folium
import geopandas as gpd
import streamlit as st
# Import streamlit_folium with fallback
try:
	from streamlit_folium import st_folium
except Exception:
	def st_folium(*args, **kwargs):
		st.warning("streamlit-folium is not available; interactive map rendering is disabled.")
		return None


def render_map(gdf: gpd.GeoDataFrame, height: int = 500):
	center = [20.0, 0.0]
	if len(gdf) > 0:
		cent = gdf.geometry.unary_union.centroid
		center = [cent.y, cent.x]
	m = folium.Map(location=center, zoom_start=2)
	if len(gdf) > 0:
		folium.GeoJson(json.loads(gdf.to_json())).add_to(m)
	return st_folium(m, height=height, use_container_width=True)
