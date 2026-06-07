import pandas as pd
import re
import glob
import os
import rasterio
from terrain_utils import load_and_clean_terrain, load_stitched_terrain, calculate_slope, extract_local_region, identify_required_tiles
from JAXA_downloader import JAXADownloader

class TerrainDatabase:
    def __init__(self, data_folder):
        self.tile_index = []
        self.data_folder = data_folder
        self._index_tiles()
        
    def _index_tiles(self):
        """Scans folder for DSM.tif files."""
        # Recursive search for *DSM.tif
        tif_files = glob.glob(os.path.join(self.data_folder, "**", "*DSM.tif"), recursive=True)
        print(f"Indexing {len(tif_files)} terrain tiles...")
        
        for fp in tif_files:
            try:
                with rasterio.open(fp) as src:
                    bounds = src.bounds
                    self.tile_index.append({
                        "file": fp,
                        "bounds": bounds # (left, bottom, right, top)
                    })
            except:
                pass

    def get_patch_for_poi(self, lat, lon, radius=2000):
        """
        Smart fetcher:
        1. Identifies which tiles cover the 2km area around the POI.
        2. If multiple tiles are needed (edge case), it stitches them.
        """
        # Approx 2km in degrees (very rough estimate for bounding box search)
        # 1. Check if we need to download tiles first
        downloader = JAXADownloader()
        
        # Calculate which 1x1 degree tiles are needed for the buffer
        # This handles the case where your 2km radius crosses a tile boundary
        needed_tiles = identify_required_tiles(lat, lon, radius)
        
        for t_lat, t_lon in needed_tiles:
            downloader.download_tile(t_lat, t_lon, self.data_folder)
            
        # 2. Re-index if new files were added
        self._index_tiles()

        # ... (rest of your existing logic for stitching and extraction)
        deg_radius = 0.025 
        req_min_lat, req_max_lat = lat - deg_radius, lat + deg_radius
        req_min_lon, req_max_lon = lon - deg_radius, lon + deg_radius

        # Find ALL overlapping tiles
        overlapping_tiles = []
        for tile in self.tile_index:
            b = tile["bounds"]
            # Check for intersection
            if not (req_max_lon < b.left or req_min_lon > b.right or 
                    req_max_lat < b.bottom or req_min_lat > b.top):
                overlapping_tiles.append(tile["file"])
        
        if not overlapping_tiles:
            raise ValueError(f"No tile covers location {lat:.4f}, {lon:.4f}")

        # LOAD DATA
        if len(overlapping_tiles) == 1:
            target_file = overlapping_tiles[0]
            msk_file = target_file.replace("DSM.tif", "MSK.tif")
            elev, transform, crs = load_and_clean_terrain(target_file, msk_file)
        else:
            dsm_files = overlapping_tiles
            msk_files = [f.replace("DSM.tif", "MSK.tif") for f in dsm_files]
            elev, transform, crs = load_stitched_terrain(dsm_files, msk_files)

        # PROCESS
        slope = calculate_slope(elev, transform)

        # EXTRACT
        local_elev = extract_local_region(lat, lon, elev, transform, crs, radius_m=radius)
        local_slope = extract_local_region(lat, lon, slope, transform, crs, radius_m=radius)

        return local_elev, local_slope, transform[0]

def _parse_dms_string(coord_str):
    """Helper to parse a single DMS string like 22°48'18.7\"S 47°49'13.9\"E"""
    if not isinstance(coord_str, str): return None
    
    # Regex designed to be flexible with spaces and symbols
    pattern = r"(\d+)[°\s]+(\d+)['\s]+([\d\.]+)[^\w]*([NSEW])"
    matches = re.findall(pattern, coord_str, re.IGNORECASE)
    
    if len(matches) == 2:
        lat = float(matches[0][0]) + float(matches[0][1])/60 + float(matches[0][2])/3600
        if matches[0][3].upper() in ['S','W']: lat *= -1
        
        lon = float(matches[1][0]) + float(matches[1][1])/60 + float(matches[1][2])/3600
        if matches[1][3].upper() in ['S','W']: lon *= -1
        
        return (lat, lon)
    return None

def load_poi_excel(file_path):
    """Parses Excel file for BOTH 'Site Coordinates' and 'DZ Center'."""
    try:
        df = pd.read_excel(file_path) # Reads .xlsx
    except:
        df = pd.read_csv(file_path)   # Fallback for .csv
        
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Identify key columns
    site_coord_col = next((c for c in df.columns if "Site Coordinates" in c), None)
    dz_coord_col = next((c for c in df.columns if "DZ Center" in c), None)
    site_name_col = next((c for c in df.columns if "Site" == c or "Site Name" in c), "Name")
    
    pois = {}
    
    # Iterate through rows
    for _, row in df.iterrows():
        base_name = str(row[site_name_col]).strip()
        if base_name == "nan": continue

        # 1. Parse Site Coordinates
        if site_coord_col and pd.notna(row[site_coord_col]):
            coords = _parse_dms_string(str(row[site_coord_col]))
            if coords:
                pois[f"{base_name} (Site)"] = coords
        
        # 2. Parse DZ Center Coordinates
        if dz_coord_col and pd.notna(row[dz_coord_col]):
            coords = _parse_dms_string(str(row[dz_coord_col]))
            if coords:
                pois[f"{base_name} (DZ)"] = coords
            
    return pois

import re
# Assuming _parse_dms_string and db (TerrainDatabase) are already loaded

def get_pois_from_paste():
    print("Paste your coordinates below (one site per line).")
    print("Format can be Decimal (40.425, -86.908) or DMS (22°48'18.7\"S 47°49'13.9\"E).")
    print("Press Enter on an EMPTY line to finish and run the pipeline:\n")
    
    lines = []
    while True:
        try:
            line = input()
            # Stop reading if the user hits enter on a blank line
            if not line.strip(): 
                break
            lines.append(line)
        except EOFError:
            break
            
    pois = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # Strategy 1: Check if it looks like DMS formatting
        if any(c in line for c in ['°', "'", '"']):
            coords = _parse_dms_string(line)
            if coords:
                pois[f"Pasted_Site_{i+1}"] = coords
        else:
            # Strategy 2: Extract decimal degrees
            # Finds any positive/negative floating point numbers or integers
            nums = re.findall(r'-?\d+\.\d+|-?\d+', line)
            if len(nums) >= 2:
                lat, lon = float(nums[0]), float(nums[1])
                # Basic validation to ensure they are real coordinates
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    pois[f"Pasted_Site_{i+1}"] = (lat, lon)
                    
    return pois