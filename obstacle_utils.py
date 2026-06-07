import numpy as np
import rasterio.features
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

def extract_obstacle_polygons(slope_map, transform, max_slope=30.0, simplify_tol=10.0, min_area=500.0):
    """
    Converts high-slope areas (raster) into valid Shapely Polygons (vector).
    
    Args:
        slope_map: 2D numpy array of slope degrees.
        transform: The rasterio Affine transform (to map pixels -> meters).
        max_slope: Threshold in degrees. Slopes > this are obstacles.
        simplify_tol: Tolerance in meters for smoothing edges.
                      Higher = fewer vertices (faster GNC solver).
                      Lower = more accurate shape.
        min_area: Minimum area (m^2) for a polygon to count. Removes pixel noise.
    
    Returns:
        List of Shapely Polygons representing the obstacles.
    """
    
    # 1. Create Binary Mask
    # Logic: Slope > Threshold OR Data is Missing (Clouds/NaN)
    # We use astype(uint8) because features.shapes expects integer types.
    obstacle_mask = (slope_map > max_slope) | (np.isnan(slope_map))
    obstacle_mask = obstacle_mask.astype(np.uint8)

    # 2. Vectorize (Raster -> Polygons)
    # rasterio.features.shapes returns a generator of (geojson, value)
    # We only care about polygons where value == 1 (The Obstacles)
    shapes_gen = rasterio.features.shapes(obstacle_mask, transform=transform, mask=obstacle_mask==1)

    raw_polygons = []
    for geom, val in shapes_gen:
        # Convert GeoJSON dict to Shapely Polygon
        poly = shape(geom)
        raw_polygons.append(poly)

    # 3. Clean and Merge
    # Filter out tiny speckles (noise) based on min_area
    significant_polygons = [p for p in raw_polygons if p.area > min_area]

    if not significant_polygons:
        return []

    # Merge overlapping polygons to create clean continuous zones
    merged_blob = unary_union(significant_polygons)
    
    # Handle the result of the union (can be Polygon or MultiPolygon)
    if merged_blob.geom_type == 'Polygon':
        final_list = [merged_blob]
    elif merged_blob.geom_type == 'MultiPolygon':
        final_list = list(merged_blob.geoms)
    else:
        final_list = []

    # 4. Simplify
    # This reduces vertex count. Essential for optimization solvers (MICP).
    simplified_polygons = [p.simplify(simplify_tol, preserve_topology=True) for p in final_list]
    
    return simplified_polygons