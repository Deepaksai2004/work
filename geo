# Install required packages (run this cell first)
!pip install rasterio xarray netcdf4 matplotlib cartopy geopandas
 
import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from google.colab import files
import warnings
 
# Suppress specific warnings
warnings.filterwarnings('ignore', message=".*Variable 'spatial_ref' has an unsupported number of dimensions.*")
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
 
def raster_to_netcdf(input_raster_path, output_netcdf_path, variable_name="depth", num_slices=5):
    """
    Convert a raster image to NetCDF format compatible with ArcGIS Pro
 
    Parameters:
    input_raster_path (str): Path to input raster file (e.g., .tif)
    output_netcdf_path (str): Path for output NetCDF file
    variable_name (str): Name for the data variable in NetCDF
    num_slices (int): Number of depth slices to create (interpolated between min and max values)
    """
 
    try:
        # Read raster using rasterio
        with rasterio.open(input_raster_path) as src:
            # Get raster properties
            cols = src.width
            rows = src.height
            bands = src.count
 
            # Get transform and CRS
            transform = src.transform
            crs = src.crs
 
            # Read raster data - handle single vs multi-band
            if bands > 1:
                print(f"Multi-band raster detected ({bands} bands). Using first band only.")
                print("For multi-temporal or multi-spectral data, consider processing each band separately.")
                raster_array = src.read(1)  # Read first band only
            else:
                raster_array = src.read(1)  # Read single band
 
            nodata_value = src.nodata
 
            # Get bounds
            bounds = src.bounds
 
        # Handle NoData values - replace with a proper fill value
        fill_value = -9999.0
        if nodata_value is not None:
            raster_array = np.where(raster_array == nodata_value, fill_value, raster_array)
        else:
            # Replace any NaN values with fill_value
            raster_array = np.where(np.isnan(raster_array), fill_value, raster_array)
 
        # Get depth values from raster range (excluding fill values)
        valid_data = raster_array[raster_array != fill_value]
        if len(valid_data) == 0:
            raise ValueError("No valid data found in raster")
 
        min_depth = float(np.min(valid_data))
        max_depth = float(np.max(valid_data))
 
        # Create coordinate arrays
        x_coords = np.linspace(bounds.left, bounds.right, cols)
        y_coords = np.linspace(bounds.top, bounds.bottom, rows)
 
        # Create depth coordinate (z-values) with interpolation
        depth_values = np.linspace(min_depth, max_depth, num_slices)
 
        # Create 3D voxel data by repeating the 2D array for each depth slice
        voxel_array = np.repeat(raster_array[np.newaxis, :, :], num_slices, axis=0)
 
        # Get CRS information
        epsg_code = None
        crs_wkt = None
 
        if crs is not None:
            epsg_code = crs.to_epsg()
            crs_wkt = crs.to_wkt()
 
        # Determine coordinate units based on CF conventions
        if crs and crs.is_geographic:
            coord_units = "degrees_east" if variable_name == "longitude" else "degrees_north"
            x_standard_name = "longitude"
            y_standard_name = "latitude"
        else:
            coord_units = "meters"
            x_standard_name = "projection_x_coordinate"
            y_standard_name = "projection_y_coordinate"
 
        # Create xarray Dataset without _FillValue in attributes
        ds = xr.Dataset(
            {
                variable_name: (["depth", "y", "x"], voxel_array, {
                    "units": "meters",
                    "long_name": f"{variable_name.title()} values from raster",
                    "missing_value": fill_value,  # Use missing_value instead of _FillValue in attributes
                    "grid_mapping": "spatial_ref"
                })
            },
            coords={
                "x": (["x"], x_coords, {
                    "units": coord_units,
                    "long_name": "x coordinate",
                    "standard_name": x_standard_name,
                    "axis": "X"
                }),
                "y": (["y"], y_coords, {
                    "units": coord_units,
                    "long_name": "y coordinate",
                    "standard_name": y_standard_name,
                    "axis": "Y"
                }),
                "depth": (["depth"], depth_values, {
                    "units": "meters",
                    "long_name": "depth",
                    "positive": "down",
                    "standard_name": "depth",
                    "axis": "Z"
                })
            }
        )
 
        # Add CRS information as a scalar variable with proper attributes
        if crs:
            crs_attrs = {
                "crs_wkt": crs_wkt,
                "spatial_ref": crs_wkt,
                "grid_mapping_name": "latitude_longitude" if crs.is_geographic else "universal_transverse_mercator",
                "longitude_of_prime_meridian": 0.0,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
                "epsg_code": str(epsg_code) if epsg_code else "unknown",
                "long_name": "CRS definition",
                "comment": "Coordinate Reference System information"
            }
 
            # Add as scalar variable (0-dimensional)
            ds["spatial_ref"] = xr.DataArray(
                data=np.int32(epsg_code if epsg_code else 0),
                dims=(),
                attrs=crs_attrs
            )
 
        # Add global attributes
        ds.attrs = {
            "title": f"Converted raster data: {os.path.basename(input_raster_path)}",
            "source": f"Converted from {input_raster_path}",
            "Conventions": "CF-1.8",
            "history": f"Created from raster using Python script in Google Colab",
            "geospatial_bounds": f"POLYGON(({bounds.left} {bounds.bottom}, {bounds.right} {bounds.bottom}, {bounds.right} {bounds.top}, {bounds.left} {bounds.top}, {bounds.left} {bounds.bottom}))",
            "geospatial_lat_min": float(bounds.bottom),
            "geospatial_lat_max": float(bounds.top),
            "geospatial_lon_min": float(bounds.left),
            "geospatial_lon_max": float(bounds.right),
            "depth_slices": num_slices,
            "depth_interpolation": f"Linear between {min_depth:.2f} and {max_depth:.2f}"
        }
 
        # Encoding settings - specify _FillValue here only
        encoding = {
            variable_name: {
                "zlib": True,
                "complevel": 4,
                "_FillValue": fill_value,  # This is where _FillValue should be specified
                "dtype": "float32"
            },
            "x": {"_FillValue": None, "dtype": "float64"},
            "y": {"_FillValue": None, "dtype": "float64"},
            "depth": {"_FillValue": None, "dtype": "float32"},
            "spatial_ref": {"_FillValue": None, "dtype": "int32"}
        }
 
        # Save to NetCDF
        ds.to_netcdf(output_netcdf_path, encoding=encoding, format="NETCDF4")
 
        print(f"Successfully converted {input_raster_path} to {output_netcdf_path}")
        print(f"Raster dimensions: {cols} x {rows} (bands: {bands})")
        print(f"Data range: {min_depth:.2f} to {max_depth:.2f}")
        print(f"Created {num_slices} depth slices between min and max values")
        print(f"CRS: {crs}")
        if epsg_code:
            print(f"EPSG Code: {epsg_code}")
 
        return ds, True
 
    except Exception as e:
        print(f"Error converting raster to NetCDF: {str(e)}")
        return None, False
 
def visualize_netcdf(netcdf_path_or_dataset, variable_name="depth", fill_value=-9999.0):
    """
    Visualize the NetCDF data using matplotlib and cartopy
    """
 
    try:
        # Load dataset
        if isinstance(netcdf_path_or_dataset, str):
            ds = xr.open_dataset(netcdf_path_or_dataset)
        else:
            ds = netcdf_path_or_dataset
 
        # Get the data variable
        data_var = ds[variable_name]
 
        # Replace fill values with NaN for visualization
        data_var = data_var.where(data_var != fill_value)
 
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
 
        # Plot 1: Simple matplotlib plot (first depth slice)
        im1 = axes[0].imshow(data_var.isel(depth=0),
                            extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                            cmap='viridis', aspect='auto')
        axes[0].set_title(f'{variable_name.title()} Data (First Depth Slice)')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
 
        # Plot 2: Middle depth slice
        middle_slice = len(ds.depth) // 2
        if 'spatial_ref' in ds and 'epsg_code' in ds.spatial_ref.attrs and ds.spatial_ref.attrs['epsg_code'] == 4326:
            # Geographic coordinates
            ax2 = plt.axes(projection=ccrs.PlateCarree())
            im2 = data_var.isel(depth=middle_slice).plot(ax=ax2, transform=ccrs.PlateCarree(),
                                            cmap='viridis', add_colorbar=False)
            ax2.add_feature(cfeature.COASTLINE)
            ax2.add_feature(cfeature.BORDERS)
            ax2.gridlines(draw_labels=True)
            ax2.set_title(f'{variable_name.title()} Data (Middle Depth Slice)')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
        else:
            # Projected coordinates
            im2 = axes[1].imshow(data_var.isel(depth=middle_slice),
                               extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                               cmap='plasma', aspect='auto')
            axes[1].set_title(f'{variable_name.title()} Data (Middle Depth Slice)')
            axes[1].set_xlabel('X Coordinate')
            axes[1].set_ylabel('Y Coordinate')
            plt.colorbar(im2, ax=axes[1], shrink=0.8)
 
        plt.tight_layout()
        plt.show()
 
        # Print dataset info
        print("\nDataset Information:")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Variables: {list(ds.data_vars.keys())}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        print(f"Depth slices: {len(ds.depth)} values from {float(ds.depth[0]):.2f} to {float(ds.depth[-1]):.2f}")
 
        return ds
 
    except Exception as e:
        print(f"Error visualizing NetCDF: {str(e)}")
        return None
 
def upload_and_convert():
    """
    Upload raster file and convert to NetCDF in Colab
    """
    print("Please upload your raster file (TIF, GeoTIFF, etc.)")
    uploaded = files.upload()
 
    if not uploaded:
        print("No file uploaded!")
        return None, None
 
    # Get the uploaded file
    input_file = list(uploaded.keys())[0]
    output_file = os.path.splitext(input_file)[0] + '.nc'
 
    print(f"\nConverting {input_file} to {output_file}")
 
    # Get number of slices from user
    num_slices = input("Enter number of depth slices to create (default 5): ").strip()
    num_slices = int(num_slices) if num_slices.isdigit() else 5
 
    # Convert to NetCDF
    ds, success = raster_to_netcdf(input_file, output_file, variable_name="elevation", num_slices=num_slices)
 
    if success:
        print(f"\nConversion successful!")
 
        # Visualize the result
        print("\nGenerating visualization...")
        visualize_netcdf(ds, "elevation")
 
        # Download the NetCDF file
        print("\nDownloading NetCDF file...")
        files.download(output_file)
 
        return ds, output_file
    else:
        return None, None
 
def main():
    """
    Main function for Google Colab
    Choose between uploading a file or using a sample
    """
 
    print("=== Raster to NetCDF Converter for Google Colab ===\n")
    print("Choose an option:")
    print("1. Upload your own raster file")
    print("2. Use a sample raster (if you have a URL)")
 
    choice = input("\nEnter your choice (1 or 2): ").strip()
 
    if choice == "1":
        # Upload and convert
        ds, output_file = upload_and_convert()
 
    elif choice == "2":
        # Use sample data or URL
        sample_url = input("Enter the URL of your raster file: ").strip()
        if sample_url:
            # Download the file first
            import urllib.request
            filename = "sample_raster.tif"
            try:
                urllib.request.urlretrieve(sample_url, filename)
                output_file = "sample_output.nc"
 
                # Get number of slices from user
                num_slices = input("Enter number of depth slices to create (default 5): ").strip()
                num_slices = int(num_slices) if num_slices.isdigit() else 5
 
                ds, success = raster_to_netcdf(filename, output_file, variable_name="elevation", num_slices=num_slices)
 
                if success:
                    visualize_netcdf(ds, "elevation")
                    files.download(output_file)
 
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
        else:
            print("No URL provided!")
 
    else:
        print("Invalid choice!")
 
def convert_existing_file(input_path, output_path, variable_name="depth", num_slices=5):
    """
    Convert an existing raster file to NetCDF
    """
    ds, success = raster_to_netcdf(input_path, output_path, variable_name, num_slices)
 
    if success:
        # Visualize
        visualize_netcdf(ds, variable_name)
 
        # Provide download
        files.download(output_path)
 
        print(f"\nNetCDF file ready for ArcGIS Pro!")
        print("Instructions for ArcGIS Pro:")
        print("1. Download the .nc file to your local machine")
        print("2. Open ArcGIS Pro")
        print("3. Use 'Add Data' to add the NetCDF file")
        print("4. Or use the 'Make NetCDF Raster Layer' tool")
 
        return ds
 
    return None
 
# Uncomment and run this line to start the interactive process
if __name__ == "__main__":
    main()
 
# Alternative: Run this for direct conversion (modify paths as needed)
# ds = convert_existing_file("your_raster.tif", "output.nc", "elevation", num_slices=10)
 
 
 
 
#is my visulization in the arcgispro is correct for 3d (real time) analysis , this this approach is good?geo
