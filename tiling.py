"""
This script generates tiles from larger images using a sliding window approach
with distinct augmentations to eliminate redundancies (Abrahams 2024,
10.48550/arXiv.2404.10927).

Altered version of https://github.com/elliesch/flipnslide/tree/main 
for compatibility with georeferenced data (rioxarray).
"""

import geopandas as gpd
import glob
import numpy as np
import os
import pandas as pd
import rioxarray
import shutil
import xarray as xr
from shapely.geometry import box
from tqdm import tqdm


class FlipnSlideBatch:
    def __init__(
        self,
        save_dir: str,
        tile_size: int = 2560,
        tile_unit: str = "m",
        save_as: str = "netcdf",
        overwrite: bool = False,
        verbose: bool = False,
    ):
        self.save_dir = save_dir
        self.tile_size = tile_size
        self.tile_unit = tile_unit
        self.save_as = save_as
        self.overwrite = overwrite
        self.verbose = verbose
        """
        Processes a folder of images to create tiles.

        Args:
            save_dir (str): Path to folder to write tiles to
            tile_size (int): Edge length of the tiles to create
            tile_unit (str): Unit of the tile size. Options are 'm' and 'px'
            save_as (str): Format to save the tiles. Options are 'netcdf' and 'tif'
            overwrite (bool): Overwrite existing files in the save_dir.
            verbose (bool): If True, print progress information
        """

    def process(self, in_folder, pattern="*.nc"):
        """
        Process a folder of images to create tiles.

        Args:
            in_folder (str): Path to folder containing images
            pattern (str): Pattern to match files in the folder
        """
        # check if save_dir is set
        if self.save_dir is None:
            raise ValueError("No save_dir specified.")
        else:
            if not self.overwrite and os.path.exists(self.save_dir):
                raise ValueError(
                    f"Directory {self.save_dir} already exists. Set overwrite=True to overwrite."
                )
            elif self.overwrite and os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
            else:
                os.makedirs(self.save_dir)

        # get all files in the folder
        in_arrs = glob.glob(os.path.join(in_folder, pattern))
        if self.verbose:
            print(f"Found {len(in_arrs)} files to be processed in {in_folder}")

        # construct an empty GeoDataFrame to store tile information
        gdf = gpd.GeoDataFrame()

        # run tiling
        for arr_path in tqdm(
            in_arrs, desc="Creating tiles via FlipNSlide", disable=not self.verbose
        ):
            arr = xr.open_dataarray(arr_path)
            prefix = os.path.splitext(os.path.split(arr_path)[-1])[0]
            fns = FlipnSlideSingle(
                tile_size=self.tile_size,
                tile_unit=self.tile_unit,
                save_dir=self.save_dir,
                save_prefix=prefix,
                save_as=self.save_as,
                verbose=False,
            )
            tile_df = fns.process(arr)
            if len(gdf):
                tile_df = tile_df.to_crs(gdf.crs)
            gdf = pd.concat([gdf, tile_df])
        gdf.reset_index(drop=True, inplace=True)

        # summarize and save the tiles
        if self.verbose:
            print(f"Generated {len(gdf)} tiles")
        gdf.to_file(os.path.join(self.save_dir, "tiles.geojson"), driver="GeoJSON")


class FlipnSlideSingle:
    def __init__(
        self,
        tile_size: int = 2560,
        tile_unit: str = "m",
        save_dir: str = None,
        save_prefix: str = None,
        save_as: str = "netcdf",
        verbose: bool = False,
    ):
        self.tile_size = tile_size
        self.tile_unit = tile_unit
        self.save_dir = save_dir
        self.save_prefix = "" if save_prefix is None else f"{save_prefix}_"
        self.save_as = save_as
        self.verbose = verbose
        """
        Processes a single image to create tiles.

        Args:
            tile_size (int): Edge length of the tiles to create
            tile_unit (str): Unit of the tile size. Options are 'm' and 'px'
            save_dir (str): Path to folder to write tiles to.
                If None, tiles are not saved but stored in memory
            save_prefix (str): Prefix to add to the name of the saved tiles.
                Ignored if save_dir is None.
            save_as (str): Format to save the tiles. Options are 'netcdf' and 'tif'.
                Ignored if save_dir is None.
            verbose (bool): If True, print progress information
        """

    def process(self, image):
        """
        Args:
            image (xarray.DataArray): An xarray DataArray object representing the image
        """
        # calculate tile size in px
        if self.tile_unit == "m":
            if not image.rio.crs.is_projected:
                raise ValueError(
                    "CRS is not projected, resolution calculation not possible"
                )
            res_m = np.mean(image.rio.resolution())
            tile_size_pxl = int(self.tile_size / res_m)
        else:
            tile_size_pxl = self.tile_size

        # crop image to square divisible by tile size
        if image.shape[-1] % tile_size_pxl != 0 or image.shape[-2] % tile_size_pxl != 0:
            if self.verbose:
                print(
                    "Image is being cropped to a square that is divisible by the tile size..."
                )

            image = self._crop(image, tile_size_pxl)

        # generate & save tiles
        tiles = self._sliding_transforms(image, tile_size_pxl)
        if self.verbose:
            print(f"Generated {len(tiles)} tiles.")

        return tiles

    def _crop(self, image, tile_size):
        """
        Crop the image to the nearest multiple of tile_size while preserving metadata.

        Args:
            image (xarray.DataArray): An xarray DataArray object representing the image
            tile_size (int): Edge length of the tiles to create (in px)
        """
        height, width = image.sizes["y"], image.sizes["x"]
        new_height = (height // tile_size) * tile_size
        new_width = (width // tile_size) * tile_size
        cropped = image.isel(y=slice(0, new_height), x=slice(0, new_width))
        return cropped

    def _sliding_transforms(self, image, tile_size):
        """
        Generate overlapping tiles with distinct augmentations to eliminate redundancies.

        Args:
            image (xarray.DataArray): An xarray DataArray object representing the image
            tile_size (int): Edge length of the tiles to create (in px)
        """
        stride = tile_size // 2

        # Main tiles: sliding with stride of half the tile size
        fold_idx = np.arange(0, image.sizes["y"] - tile_size + 1, stride)

        # Inner tiles: 25% and 75% sliding logic (with flip and rotation augmentations)
        offset = tile_size // 4
        inner_image = image.isel(
            y=slice(offset, image.sizes["y"] - offset),
            x=slice(offset, image.sizes["x"] - offset),
        )
        fold_idx_inner = np.arange(0, inner_image.sizes["y"] - tile_size + 1, stride)

        # Initialize lists to store tiles and their indices
        tiles = []
        idx_tiles = []
        tile_count = 0

        # Create progress bar
        if self.verbose:
            iter_main = int(len(fold_idx) ** 2)
            if len(fold_idx_inner) > 0:
                iter_inner = int((len(fold_idx_inner)) ** 2)
            else:
                iter_inner = 0
            pbar = tqdm(total=(iter_main + iter_inner), desc="Generating tiles")

        try:
            # Process main tiles
            for idx_x in range(len(fold_idx)):
                for idx_y in range(len(fold_idx)):
                    tile = image.isel(
                        y=slice(fold_idx[idx_x], fold_idx[idx_x] + tile_size),
                        x=slice(fold_idx[idx_y], fold_idx[idx_y] + tile_size),
                    )

                    # Apply augmentations based on position
                    if (idx_x % 2 != 0) & (idx_y % 2 != 0):
                        tile.values = np.rot90(tile.values, k=3, axes=(-2, -1))  # 270
                        idx_tiles.append(1)

                    elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
                        tile.values = np.rot90(tile.values, k=2, axes=(-2, -1))  # 180
                        idx_tiles.append(2)

                    elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
                        tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))  # 90
                        idx_tiles.append(3)

                    else:
                        idx_tiles.append(0)

                    # save tiles
                    if self.save_dir:
                        if self.save_as == "netcdf":
                            file_path = os.path.join(
                                self.save_dir, f"{self.save_prefix}{tile_count}.nc"
                            )
                            tile.to_netcdf(file_path)
                        elif self.save_as == "tif":
                            file_path = os.path.join(
                                self.save_dir, f"{self.save_prefix}{tile_count}.tif"
                            )
                            tile.rio.to_raster(file_path)
                        else:
                            raise ValueError(f"Invalid save_as format: {self.save_as}.")

                        # create Geodataframe on processed tile
                        crs = tile.rio.crs
                        bbox = box(*tile.rio.bounds())
                        idx_original = idx_tiles[-1] == 0

                        tiles.append(
                            gpd.GeoDataFrame(
                                {"tile_path": file_path, "undistorted": idx_original},
                                geometry=[bbox],
                                crs=crs,
                                index=[0],
                            )
                        )
                    # save tile in memory
                    else:
                        tiles.append(tile)

                    # increment tile count
                    if self.verbose:
                        pbar.update(1)
                    tile_count += 1

            # Process inner tiles
            for idx_x in range(len(fold_idx_inner)):
                for idx_y in range(len(fold_idx_inner)):
                    tile = inner_image.isel(
                        y=slice(
                            fold_idx_inner[idx_x], fold_idx_inner[idx_x] + tile_size
                        ),
                        x=slice(
                            fold_idx_inner[idx_y], fold_idx_inner[idx_y] + tile_size
                        ),
                    )

                    if (idx_x % 2 == 0) & (idx_y % 2 == 0):
                        tile.values = tile.values[..., ::-1]  # horizontal flip
                        idx_tiles.append(4)

                    elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
                        tile.values = tile.values[..., ::-1, :]  # vertical flip
                        idx_tiles.append(5)

                    elif (idx_x % 2 != 0) & (idx_y % 2 != 0):
                        tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))
                        tile.values = tile.values[..., ::-1]
                        idx_tiles.append(6)

                    elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
                        tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))
                        tile.values = tile.values[..., ::-1, :]
                        idx_tiles.append(7)

                    # save tiles
                    if self.save_dir:
                        if self.save_as == "netcdf":
                            file_path = os.path.join(
                                self.save_dir, f"{self.save_prefix}{tile_count}.nc"
                            )
                            tile.to_netcdf(file_path)
                        elif self.save_as == "tif":
                            file_path = os.path.join(
                                self.save_dir, f"{self.save_prefix}{tile_count}.tif"
                            )
                            tile.rio.to_raster(file_path)
                        else:
                            raise ValueError(f"Invalid save_as format: {self.save_as}.")

                        # create Geodataframe on processed tile
                        crs = tile.rio.crs
                        bbox = box(*tile.rio.bounds())
                        idx_original = idx_tiles[-1] == 0

                        tiles.append(
                            gpd.GeoDataFrame(
                                {"tile_path": file_path, "undistorted": idx_original},
                                geometry=[bbox],
                                crs=crs,
                                index=[0],
                            )
                        )
                    # save tile in memory
                    else:
                        tiles.append(tile)

                    # increment tile count
                    if self.verbose:
                        pbar.update(1)
                    tile_count += 1

            # finish processing of tiles
            if self.save_dir:
                tiles = pd.concat(tiles, ignore_index=True)

        finally:
            if self.verbose:
                pbar.close()

        return tiles


# # old version
# # differences
# # a) keeps all tiles in memory before saving
# # b) wrong adjuster specifications
# # c) no progress bar
# # d) no batch processing for a folder of images

# class FlipnSlideRio:

#     def __init__(self, tile_size: int = 256, verbose: bool = False):
#         self.tile_size = tile_size
#         self.verbose = verbose

#     def process(self, image):
#         """
#         Generate tiles using the specified tile style.

#         Args:
#             image (xarray.DataArray): An xarray DataArray object representing the image.
#         """
#         # Crop image to square divisible by tile size
#         if (
#             image.shape[-1] % self.tile_size != 0
#             or image.shape[-2] % self.tile_size != 0
#         ):
#             if self.verbose:
#                 print(
#                     "Image is being cropped to a square that is divisible by the tile size..."
#                 )

#             image = self._crop(image, self.tile_size)

#         # Generate tiles
#         tiles, tiles_idx = self._sliding_transforms(image)
#         if self.verbose:
#             print(f"Generated {len(tiles)} tiles.")

#         return tiles, tiles_idx

#     def save_tiles(self, tiles, tiles_idx, save_path, save_as="netcdf"):
#         """
#         Save each tile to disk with geospatial attributes.
#         """
#         # save tiles
#         tile_paths = []
#         for i, tile in enumerate(tiles):
#             if save_as == "netcdf":
#                 file_path = f"{save_path}_{i}.nc"
#                 tile.to_netcdf(file_path)
#                 tile_paths.append(file_path)
#             elif save_as == "tif":
#                 file_path = f"{save_path}_{i}.tif"
#                 tile.rio.to_raster(file_path)
#                 tile_paths.append(file_path)
#             else:
#                 raise ValueError(f"save_as={save_as} is no option.")

#         # save information on tiles not being rotated or flipped
#         crs = tiles[0].rio.crs
#         bboxes = [box(*x.rio.bounds()) for x in tiles]
#         idx_original = np.array(tiles_idx) == 0
#         tile_gdf = gpd.GeoDataFrame(
#             {"tile_paths": tile_paths, "undistorted": idx_original},
#             geometry=bboxes,
#             crs=crs,
#         )
#         return tile_gdf

#     def _crop(self, image):
#         """
#         Crop the image to the nearest multiple of tile_size while preserving metadata.
#         """
#         height, width = image.sizes["y"], image.sizes["x"]
#         new_height = (height // self.tile_size) * self.tile_size
#         new_width = (width // self.tile_size) * self.tile_size
#         cropped = image.isel(y=slice(0, new_height), x=slice(0, new_width))
#         return cropped

#     def _sliding_transforms(self, image):
#         """
#         Generate overlapping tiles with distinct augmentations to eliminate redundancies.
#         """
#         stride = self.tile_size // 2

#         # Initialize lists to store tiles and their indices
#         tiles = []
#         idx_tiles = []

#         # Main tiles: sliding with stride of half the tile size
#         fold_idx = np.arange(0, image.sizes["y"] - self.tile_size + 1, stride)

#         for idx_x in range(len(fold_idx)):
#             for idx_y in range(len(fold_idx)):
#                 tile = image.isel(
#                     y=slice(fold_idx[idx_x], fold_idx[idx_x] + self.tile_size),
#                     x=slice(fold_idx[idx_y], fold_idx[idx_y] + self.tile_size),
#                 )

#                 # Apply augmentations based on position
#                 if (idx_x % 2 != 0) & (idx_y % 2 != 0):
#                     tile.values = np.rot90(tile.values, k=3, axes=(-2, -1))  # 270
#                     # track the indices
#                     idx_tiles.append(1)

#                 elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
#                     tile.values = np.rot90(tile.values, k=2, axes=(-2, -1))  # 180
#                     # track the indices
#                     idx_tiles.append(2)

#                 elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
#                     tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))  # 90
#                     # track the indices
#                     idx_tiles.append(3)

#                 else:
#                     # track the indices
#                     idx_tiles.append(0)

#                 tiles.append(tile)

#         # Inner tiles with 25% and 75% sliding logic (with flip and rotation augmentations)
#         offset = self.tile_size // 4
#         inner_image = image.isel(
#             y=slice(offset, image.sizes["y"] - offset),
#             x=slice(offset, image.sizes["x"] - offset),
#         )
#         fold_idx_inner = np.arange(
#             0, inner_image.sizes["y"] - self.tile_size + 1, stride
#         )

#         # Determine adjuster based on tile size
#         if self.tile_size == 64:
#             adjuster = 0
#         elif self.tile_size == 128:
#             adjuster = 0
#         elif self.tile_size == 256:
#             adjuster = 1
#         elif self.tile_size == 512:
#             adjuster = 2

#         for idx_x in range(len(fold_idx_inner) - adjuster):
#             for idx_y in range(len(fold_idx_inner) - adjuster):
#                 tile = inner_image.isel(
#                     y=slice(
#                         fold_idx_inner[idx_x], fold_idx_inner[idx_x] + self.tile_size
#                     ),
#                     x=slice(
#                         fold_idx_inner[idx_y], fold_idx_inner[idx_y] + self.tile_size
#                     ),
#                 )

#                 if (idx_x % 2 == 0) & (idx_y % 2 == 0):
#                     tile.values = tile.values[..., ::-1]  # horizontal flip
#                     # track the indices
#                     idx_tiles.append(4)

#                 elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
#                     tile.values = tile.values[..., ::-1, :]  # vertical flip
#                     # track the indices
#                     idx_tiles.append(5)

#                 elif (idx_x % 2 != 0) & (idx_y % 2 != 0):
#                     tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))
#                     tile.values = tile.values[..., ::-1]
#                     # track the indices
#                     idx_tiles.append(6)

#                 elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
#                     tile.values = np.rot90(tile.values, k=1, axes=(-2, -1))
#                     tile.values = tile.values[..., ::-1, :]
#                     # track the indices
#                     idx_tiles.append(7)

#                 tiles.append(tile)

#         return tiles, idx_tiles
