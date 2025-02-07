"""
This script generates tiles from an image using a sliding window approach
with distinct augmentations to eliminate redundancies. Altered version of
https://github.com/elliesch/flipnslide/tree/main for compatibility with
georeferenced data (rioxarray).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from shapely.geometry import box
from tqdm import tqdm


class FlipnSlideRio:

    def __init__(self, tile_size: int = 256, verbose: bool = False):
        self.tile_size = tile_size
        self.verbose = verbose

    def process(self, image, save_path=None, save_as="netcdf"):
        """
        Generate tiles using the specified tile style.

        Args:
            image (xarray.DataArray): An xarray DataArray object representing the image.
            save_path (str): Path to save the tiles. If None, tiles are not saved but stored in memory.
            save_as (str): Format to save the tiles. Options are 'netcdf' and 'tif'.
        """
        # Crop image to square divisible by tile size
        if (
            image.shape[-1] % self.tile_size != 0
            or image.shape[-2] % self.tile_size != 0
        ):
            if self.verbose:
                print(
                    "Image is being cropped to a square that is divisible by the tile size..."
                )

            image = self._crop(image)

        # Generate & save tiles
        tiles = self._sliding_transforms(image, save_path, save_as)
        if self.verbose:
            print(f"Generated {len(tiles)} tiles.")

        return tiles

    def _crop(self, image):
        """
        Crop the image to the nearest multiple of tile_size while preserving metadata.
        """
        height, width = image.sizes["y"], image.sizes["x"]
        new_height = (height // self.tile_size) * self.tile_size
        new_width = (width // self.tile_size) * self.tile_size
        cropped = image.isel(y=slice(0, new_height), x=slice(0, new_width))
        return cropped

    def _sliding_transforms(self, image, save_path, save_as):
        """
        Generate overlapping tiles with distinct augmentations to eliminate redundancies.
        """
        stride = self.tile_size // 2

        # Main tiles: sliding with stride of half the tile size
        fold_idx = np.arange(0, image.sizes["y"] - self.tile_size + 1, stride)

        # Inner tiles: 25% and 75% sliding logic (with flip and rotation augmentations)
        offset = self.tile_size // 4
        inner_image = image.isel(
            y=slice(offset, image.sizes["y"] - offset),
            x=slice(offset, image.sizes["x"] - offset),
        )
        fold_idx_inner = np.arange(
            0, inner_image.sizes["y"] - self.tile_size + 1, stride
        )

        # Create progress bar
        if self.verbose:
            iter_main = int(len(fold_idx) ** 2)
            if len(fold_idx_inner) > 0:
                iter_inner = int((len(fold_idx_inner)) ** 2)
            else:
                iter_inner = 0
            pbar = tqdm(total=(iter_main + iter_inner), desc="Generating tiles")

        # Initialize lists to store tiles and their indices
        tiles = []
        idx_tiles = []
        tile_count = 0

        # Process main tiles
        for idx_x in range(len(fold_idx)):
            for idx_y in range(len(fold_idx)):
                tile = image.isel(
                    y=slice(fold_idx[idx_x], fold_idx[idx_x] + self.tile_size),
                    x=slice(fold_idx[idx_y], fold_idx[idx_y] + self.tile_size),
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
                if save_path:
                    if save_as == "netcdf":
                        file_path = f"{save_path}_{tile_count}.nc"
                        tile.to_netcdf(file_path)
                    elif save_as == "tif":
                        file_path = f"{save_path}_{tile_count}.tif"
                        tile.rio.to_raster(file_path)
                    else:
                        raise ValueError(f"Invalid save_as format: {save_as}.")

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
                        fold_idx_inner[idx_x], fold_idx_inner[idx_x] + self.tile_size
                    ),
                    x=slice(
                        fold_idx_inner[idx_y], fold_idx_inner[idx_y] + self.tile_size
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
                if save_path:
                    if save_as == "netcdf":
                        file_path = f"{save_path}_{tile_count}.nc"
                        tile.to_netcdf(file_path)
                    elif save_as == "tif":
                        file_path = f"{save_path}_{tile_count}.tif"
                        tile.rio.to_raster(file_path)
                    else:
                        raise ValueError(f"Invalid save_as format: {save_as}.")

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

        if self.verbose:
            pbar.close()

        # finish processing of tiles
        if save_path:
            tiles = pd.concat(tiles, ignore_index=True)

        return tiles


# # old version
# # differences
# # a) keeps all tiles in memory before saving
# # b) wrong adjuster specifications


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
