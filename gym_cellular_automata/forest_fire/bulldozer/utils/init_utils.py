"""Initialization utilities for the Advanced Bulldozer environment."""

import numpy as np
import jax.numpy as jnp
import math
import itertools
from flax import struct


def init_vegetation(row_count, column_count, num_envs):
    """Initialize vegetation matrix with random patches."""
    veg_matrix = np.zeros((num_envs, row_count, column_count), dtype=int)

    for env in range(num_envs):
        # Create random patches of different vegetation types
        num_patches = np.random.randint(4, 8)
        for _ in range(num_patches):
            # Random patch center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            patch_height = np.random.randint(3, row_count // 2)
            patch_width = np.random.randint(3, column_count // 2)

            # Random vegetation type (1-3)
            veg_type = np.random.randint(1, 6)

            # Fill patch
            row_start = max(0, center_row - patch_height // 2)
            row_end = min(row_count, center_row + patch_height // 2)
            col_start = max(0, center_col - patch_width // 2)
            col_end = min(column_count, center_col + patch_width // 2)

            veg_matrix[env, row_start:row_end, col_start:col_end] = veg_type

        # Fill any remaining zeros randomly
        zero_mask = veg_matrix[env] == 0
        size = np.sum(zero_mask)
        veg_matrix[env][zero_mask] = np.random.randint(1, 4, size=size)

    return veg_matrix


def init_density(row_count, column_count, num_envs):
    """Initialize density matrix with random patches."""
    den_matrix = np.zeros((num_envs, row_count, column_count), dtype=int)

    for env in range(num_envs):
        # Create random patches of different density types
        num_patches = np.random.randint(4, 8)
        for _ in range(num_patches):
            # Random patch center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            patch_height = np.random.randint(3, row_count // 2)
            patch_width = np.random.randint(3, column_count // 2)

            # Random density type (1-3)
            den_type = np.random.randint(1, 6)

            # Fill patch
            row_start = max(0, center_row - patch_height // 2)
            row_end = min(row_count, center_row + patch_height // 2)
            col_start = max(0, center_col - patch_width // 2)
            col_end = min(column_count, center_col + patch_width // 2)

            den_matrix[env, row_start:row_end, col_start:col_end] = den_type

        # Fill any remaining zeros randomly
        zero_mask = den_matrix[env] == 0
        size = np.sum(zero_mask)
        den_matrix[env][zero_mask] = np.random.randint(1, 4, size=size)

    return den_matrix


def init_altitude(row_count, column_count, num_envs):
    """Initialize altitude matrix with hills and slopes."""
    altitude = np.zeros((num_envs, row_count, column_count))

    for env in range(num_envs):
        # Start with a base of random noise
        altitude[env] = np.random.uniform(0, 5, (row_count, column_count))

        # Add more small hills
        num_hills = np.random.randint(6, 10)
        for _ in range(num_hills):
            # Random hill center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            radius = np.random.randint(2, min(row_count, column_count) // 4)
            height = np.random.uniform(2, 6)

            # Create hill using distance from center
            for i in range(row_count):
                for j in range(column_count):
                    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                    if distance < radius:
                        # Smooth hill falloff using cosine
                        factor = np.cos(distance / radius * np.pi / 2)
                        altitude[env, i, j] += height * factor

        # Add some gentle slopes
        num_slopes = np.random.randint(4, 8)
        for _ in range(num_slopes):
            start_row = np.random.randint(0, row_count - 4)
            start_col = np.random.randint(0, column_count - 4)
            width = np.random.randint(3, column_count // 4)
            height = np.random.randint(3, row_count // 4)
            height_diff = np.random.uniform(1, 4)

            for i in range(start_row, min(start_row + height, row_count)):
                for j in range(start_col, min(start_col + width, column_count)):
                    progress = (i - start_row) / height
                    altitude[env, i, j] += height_diff * progress

    return altitude / 10


def create_up_to_k_mappings(n, k):
    """Create mappings for combinations of size 0 to k from n items.

    Returns mappings between indices and binary arrays
    (e.g., [0,1,1,0] means items 1 and 2 are selected)
    """
    binary_vectors = []
    binary_to_id = {}

    current_id = 0
    # Generate combinations for each size from 0 to k
    for i in range(k + 1):
        for combo in itertools.combinations(range(n), i):
            # Create binary array (e.g., [0,1,1,0] for combo (1,2))
            binary = [0] * n
            for idx in combo:
                binary[idx] = 1
            binary = tuple(binary)  # Make it hashable for dict key

            binary_vectors.append(binary)
            binary_to_id[binary] = current_id
            current_id += 1

    id_to_binary = jnp.array(binary_vectors, dtype=jnp.int32)
    return id_to_binary, binary_to_id


# Convenience functions for uniform initialization
def init_density_same(row_count, column_count, num_envs):
    """Creates a density matrix filled entirely with middle value (3)"""
    return np.full((num_envs, row_count, column_count), 3, dtype=int)


def init_vegetation_same(row_count, column_count, num_envs):
    """Creates a vegetation matrix filled entirely with middle value (3)"""
    return np.full((num_envs, row_count, column_count), 3, dtype=int)


def init_altitude_same(row_count, column_count, num_envs):
    """Creates an altitude matrix filled entirely with zeros"""
    return np.zeros((num_envs, row_count, column_count), dtype=int)


def tg(x):
    return math.degrees(math.atan(x))


def get_slope(altitude, row_count, column_count, num_envs):
    # Initialize slope matrix with 3x3 sub-matrices of zeros
    slope_matrix = np.zeros((num_envs, row_count, column_count, 3, 3))

    for env in range(num_envs):
        # Skip edges since they remain flat (all zeros)
        for row in range(1, row_count - 1):
            for col in range(1, column_count - 1):
                # Get 3x3 neighborhood of altitudes
                neighborhood = altitude[env, row - 1 : row + 2, col - 1 : col + 2]
                current = altitude[env, row, col]

                # Calculate slopes
                diffs = current - neighborhood

                # Apply diagonal scaling
                diagonals = [(0, 0), (0, 2), (2, 0), (2, 2)]
                for i, j in diagonals:
                    diffs[i, j] /= 1.414

                # Convert to angles
                slope_matrix[env, row, col] = np.degrees(np.arctan(diffs))

                # Center point is always 0
                slope_matrix[env, row, col, 1, 1] = 0

    # Bin slope values into 10 ranges and count cells in each bin
    slope_flat = slope_matrix.flatten()
    slope_min, slope_max = np.min(slope_flat), np.max(slope_flat)
    bins = np.linspace(slope_min, slope_max, num=11)  # 11 edges make 10 bins
    hist, _ = np.histogram(slope_flat, bins=bins)
    print("\nSlope distribution:")
    for i, count in enumerate(hist):
        print(f"Range {bins[i]:.2f} to {bins[i+1]:.2f}: {count} cells")
    return slope_matrix


wind_thetas = [
    # North Wind (blowing from South to North)
    [[45, 0, 45], [90, 0, 90], [135, 180, 135]],
    # Northeast Wind
    [[90, 45, 0], [135, 0, 45], [180, 135, 90]],
    # East Wind (blowing from West to East)
    [[135, 90, 45], [180, 0, 0], [135, 90, 45]],
    # Southeast Wind
    [[180, 135, 90], [135, 0, 45], [90, 45, 0]],
    # South Wind (blowing from North to South)
    [[135, 180, 135], [90, 0, 90], [45, 0, 45]],
    # Southwest Wind
    [[90, 135, 180], [45, 0, 135], [0, 45, 90]],
    # West Wind (blowing from East to West)
    [[45, 90, 135], [0, 0, 180], [45, 90, 135]],
    # Northwest Wind
    [[0, 45, 90], [45, 0, 135], [90, 135, 180]],
]

# thetas = [[45, 0, 45], [90, 0, 90], [135, 180, 135]]


def calc_pw(theta):
    c_1, c_2 = 0.045, 0.131
    V = 10
    t = np.radians(theta)
    ft = np.exp(V * c_2 * (np.cos(t) - 1))
    return np.exp(c_1 * V) * ft, ft


def get_winds(use_hidden):
    winds = []
    if True:
        thetas = wind_thetas
    else:
        thetas = [wind_thetas[0] for _ in range(len(wind_thetas))]
    for thetas in wind_thetas:
        wind_matrix = np.zeros((3, 3))
        theta_array = np.array(thetas)
        wind_matrix, ft = calc_pw(theta_array)
        wind_matrix[1, 1] = 0
        winds.append((wind_matrix, ft))
    return winds
