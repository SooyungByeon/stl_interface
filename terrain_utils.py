import rasterio
import numpy as np
import math
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import transform as transform_coords
from rasterio.merge import merge
from rasterio.io import MemoryFile

def _get_utm_epsg_from_bounds(left, bottom, right, top):
    """Dynamically calculates the UTM EPSG code for the center of a bounding box."""
    center_lon = (left + right) / 2.0
    center_lat = (bottom + top) / 2.0
    
    # Calculate the UTM zone (1-60)
    zone_number = int((math.floor((center_lon + 180) / 6) % 60) + 1)
    
    # Northern hemisphere starts with 326, Southern with 327
    epsg_prefix = 32600 if center_lat >= 0 else 32700
    
    return f"EPSG:{epsg_prefix + zone_number}"

def _reproject_and_clean(src_dsm, src_msk):
    """Internal helper to reproject and clean a dataset (File or MemoryFile)."""
    
    # 1. Dynamically assign the correct local UTM CRS
    dst_crs = _get_utm_epsg_from_bounds(*src_dsm.bounds)
    
    transform, width, height = calculate_default_transform(
        src_dsm.crs, dst_crs, src_dsm.width, src_dsm.height, *src_dsm.bounds)
    
    # 2. Reproject DSM
    dsm_meters = np.zeros((height, width), np.float32)
    reproject(
        source=rasterio.band(src_dsm, 1),
        destination=dsm_meters,
        src_transform=src_dsm.transform,
        src_crs=src_dsm.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear)

    # 3. Reproject MSK
    if src_msk:
        msk_aligned = np.zeros((height, width), np.uint8)
        reproject(
            source=rasterio.band(src_msk, 1),
            destination=msk_aligned,
            src_transform=src_msk.transform,
            src_crs=src_msk.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        
        # 4. Clean Clouds/Water (Assuming MSK value 1 represents invalid data)
        cloud_mask = (msk_aligned & 1) == 1
        dsm_meters[cloud_mask] = np.nan
    
    # Note: We return dst_crs as well so the extraction function knows what projection was used
    return dsm_meters, transform, dst_crs

def load_and_clean_terrain(dsm_path, msk_path):
    """Standard loader for a single tile."""
    with rasterio.open(dsm_path) as src_dsm:
        try:
            src_msk = rasterio.open(msk_path)
        except:
            src_msk = None
            
        data, transform, dst_crs = _reproject_and_clean(src_dsm, src_msk)
        
        if src_msk: src_msk.close()
        return data, transform, dst_crs

def load_stitched_terrain(dsm_paths, msk_paths):
    """
    Advanced loader: Merges multiple tiles in memory, then cleans/reprojects.
    """
    # 1. Merge DSMs
    src_dsms = [rasterio.open(f) for f in dsm_paths]
    dsm_mosaic, dsm_trans = merge(src_dsms)
    
    # 2. Merge MSKs
    src_msks = [rasterio.open(f) for f in msk_paths if f]
    if src_msks:
        msk_mosaic, msk_trans = merge(src_msks, resampling=Resampling.nearest)
    else:
        msk_mosaic = None

    # 3. Wrap in MemoryFile so we can reuse the _reproject logic
    # We need to define the profile (metadata) for this new mosaic
    profile = src_dsms[0].profile.copy()
    profile.update({
        "height": dsm_mosaic.shape[1],
        "width": dsm_mosaic.shape[2],
        "transform": dsm_trans
    })

    # Close original handles
    for s in src_dsms + src_msks: s.close()

    # Process inside a virtual file
    with MemoryFile() as mem_dsm:
        with mem_dsm.open(**profile) as dataset_dsm:
            dataset_dsm.write(dsm_mosaic)
            
            if msk_mosaic is not None:
                with MemoryFile() as mem_msk:
                    with mem_msk.open(**profile) as dataset_msk:
                        dataset_msk.write(msk_mosaic)
                        return _reproject_and_clean(dataset_dsm, dataset_msk)
            else:
                return _reproject_and_clean(dataset_dsm, None)

def calculate_slope(elevation, transform):
    """Calculates slope in degrees."""
    dx = transform[0]
    dy = -transform[4]
    grad_y, grad_x = np.gradient(elevation, dy, dx)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    return np.degrees(slope_rad)

def extract_local_region(lat, lon, full_array, full_transform, src_crs, radius_m=2000):
    """Extracts the 2km patch."""
    x_utm, y_utm = transform_coords('EPSG:4326', src_crs, [lon], [lat])
    center_x, center_y = x_utm[0], y_utm[0]
    center_col, center_row = ~full_transform * (center_x, center_y)
    
    res_x = full_transform[0]
    pixels_radius = int(radius_m / res_x)
    
    row_start = max(0, int(center_row - pixels_radius))
    row_end = min(full_array.shape[0], int(center_row + pixels_radius))
    col_start = max(0, int(center_col - pixels_radius))
    col_end = min(full_array.shape[1], int(center_col + pixels_radius))
    
    if row_start >= row_end or col_start >= col_end:
        raise ValueError(f"Location ({lat:.4f}, {lon:.4f}) outside of loaded map bounds.")

    return full_array[row_start:row_end, col_start:col_end]

def identify_required_tiles(lat, lon, radius_m):
    """
    Calculates the set of 1x1 degree tiles needed to cover a radius.
    Returns a list of tuples: [(lat_floor, lon_floor), ...]
    """
    R = 6378137 # Earth's radius in meters
    
    # Convert radius to degrees (approximation)
    d_lat = (radius_m / R) * (180 / math.pi)
    d_lon = (radius_m / (R * math.cos(math.pi * lat / 180))) * (180 / math.pi)

    # Bounding box of the POI + buffer
    min_lat, max_lat = lat - d_lat, lat + d_lat
    min_lon, max_lon = lon - d_lon, lon + d_lon

    lat_range = range(math.floor(min_lat), math.ceil(max_lat))
    lon_range = range(math.floor(min_lon), math.ceil(max_lon))

    tiles = []
    for l in lat_range:
        for r in lon_range:
            tiles.append((l, r))
    return tiles