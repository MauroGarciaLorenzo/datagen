import numpy as np
from scipy.spatial import distance_matrix, ConvexHull
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import warnings


def read_data(csv_path):
    """
    Read and preprocess data from CSV file.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        np.ndarray: Normalized numeric data
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")

    numeric_df = df.select_dtypes(include=[np.number])
    # Print removed columns to warn about data delection
    print([col for col in df.columns if col not in numeric_df.columns])

    if numeric_df.empty:
        raise ValueError("No numeric columns found in the dataset")

    data = numeric_df.to_numpy()

    if len(data) == 0:
        raise ValueError("Dataset is empty")

    # Remove rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) == 0:
        raise ValueError("No valid data points after removing NaN values")

    return MinMaxScaler().fit_transform(data)


def star_discrepancy_proxy(points, n_samples=10000):
    """
    Approximate star discrepancy using Monte Carlo sampling.
    Optimized for high-dimensional data.

    Star discrepancy measures how uniformly distributed points are
    in the unit hypercube [0,1]^d.

    Args:
        points (np.ndarray): Points in [0,1]^d
        n_samples (int): Number of anchor points to sample

    Returns:
        float: Approximate star discrepancy
    """
    if len(points) == 0:
        return 0.0

    d = points.shape[1]

    # For high dimensions, use only random sampling
    # Grid sampling becomes intractable
    if d > 10:
        anchors = np.random.rand(n_samples, d)
    else:
        # Use both random points and grid points for better coverage
        n_random = n_samples // 2
        n_grid = n_samples - n_random

        # Random anchor points
        random_anchors = np.random.rand(n_random, d)

        # Grid anchor points for better coverage
        grid_size = int(np.ceil(n_grid ** (1 / d)))
        coords = [np.linspace(0, 1, grid_size) for _ in range(d)]
        grid_points = np.array(np.meshgrid(*coords)).T.reshape(-1, d)
        # Sample from grid if too many points
        if len(grid_points) > n_grid:
            idx = np.random.choice(len(grid_points), n_grid, replace=False)
            grid_anchors = grid_points[idx]
        else:
            grid_anchors = grid_points

        anchors = np.vstack([random_anchors, grid_anchors])

    # Vectorized computation for better performance
    # Check if points are in [0, anchor] boxes for all anchors at once
    discrepancies = []

    # Process in batches to avoid memory issues
    batch_size = min(1000, n_samples)
    for i in range(0, len(anchors), batch_size):
        batch_anchors = anchors[i:i + batch_size]

        # Vectorized comparison: points[:, None, :] <= batch_anchors[None, :, :]
        # This creates a (n_points, batch_size, d) boolean array
        in_boxes = np.all(points[:, None, :] <= batch_anchors[None, :, :],
                          axis=2)
        count_inside = np.mean(in_boxes, axis=0)  # Average over points

        # Volume of [0, anchor] boxes
        box_volumes = np.prod(batch_anchors, axis=1)

        batch_discrepancies = np.abs(count_inside - box_volumes)
        discrepancies.extend(batch_discrepancies)

    return max(discrepancies)


def maximin_distance(points):
    """
    Compute maximin distance (minimum distance between any two points).

    Args:
        points (np.ndarray): Array of points

    Returns:
        float: Maximin distance
    """
    if len(points) < 2:
        return np.inf

    # Remove duplicate points
    unique_points = np.unique(points, axis=0)

    if len(unique_points) < 2:
        return np.inf

    # Use NearestNeighbors for efficiency
    nn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(unique_points)
    distances, _ = nn.kneighbors(unique_points)
    return np.min(
        distances[:, 1])  # Second column is nearest neighbor distance


def dispersion(points, n_test=10000):
    """
    Estimate dispersion (fill distance) by sampling random points.

    Dispersion is the maximum distance from any point in the domain
    to the nearest dataset point.

    Args:
        points (np.ndarray): Dataset points
        n_test (int): Number of test points to sample

    Returns:
        float: Estimated dispersion
    """
    if len(points) == 0:
        return np.inf

    d = points.shape[1]
    test_points = np.random.rand(n_test, d)

    # Use NearestNeighbors for efficiency
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
    distances, _ = nn.kneighbors(test_points)

    return np.max(distances[:, 0])


def separation_distance(points):
    """
    Compute separation distance (minimum distance between distinct points).

    Args:
        points (np.ndarray): Array of points

    Returns:
        float: Separation distance
    """
    return maximin_distance(points)  # Same as maximin distance


def fill_distance_estimate(points, n_test=10000):
    """
    Estimate fill distance using Monte Carlo sampling.

    Fill distance is the supremum of distances from any point in the domain
    to the nearest dataset point.

    Args:
        points (np.ndarray): Dataset points
        n_test (int): Number of test points

    Returns:
        float: Estimated fill distance
    """
    return dispersion(points, n_test)


def mesh_ratio(fill_dist, sep_dist):
    """
    Compute mesh ratio (fill distance / separation distance).

    A measure of the quality of point distribution.
    Lower values indicate better distribution.

    Args:
        points (np.ndarray): Dataset points
        n_test (int): Number of test points for fill distance estimation

    Returns:
        float: Mesh ratio
    """
    # fill_dist = fill_distance_estimate(points, n_test)
    # sep_dist = separation_distance(points)

    if sep_dist == 0 or np.isinf(sep_dist):
        return np.inf

    return fill_dist / sep_dist


def convex_hull_volume(points):
    """
    Compute convex hull volume with proper error handling.
    Note: Not reliable for high-dimensional data (d > 20).

    Args:
        points (np.ndarray): Array of points

    Returns:
        float: Convex hull volume, or NaN if not computable
    """
    if len(points) == 0:
        return 0.0

    d = points.shape[1]
    n = points.shape[0]

    # For high dimensions, convex hull is not reliable
    if d > 20:
        warnings.warn(
            f"Convex hull volume not reliable for dimension {d} > 20")
        return np.nan

    if d > n:
        return np.nan  # Not enough points to span the space

    # Remove duplicate points
    unique_points = np.unique(points, axis=0)

    if len(unique_points) < d + 1:
        return np.nan  # Not enough unique points for convex hull in this dimension

    try:
        hull = ConvexHull(unique_points)
        return hull.volume
    except (ValueError, RuntimeError) as e:
        warnings.warn(f"Could not compute convex hull: {e}")
        return np.nan


def coverage_radius(points):
    """
    Compute coverage radius (radius of smallest ball containing all points).

    Args:
        points (np.ndarray): Array of points

    Returns:
        float: Coverage radius
    """
    if len(points) <= 1:
        return 0.0

    # Simple approximation: half the maximum pairwise distance
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.max(distances)


def packing_efficiency(points):
    """
    Estimate packing efficiency based on point distribution.
    Note: Less reliable for high-dimensional data.

    Args:
        points (np.ndarray): Array of points

    Returns:
        float: Packing efficiency estimate
    """
    if len(points) < 2:
        return 0.0

    d = points.shape[1]

    # For high dimensions, use simplified estimate
    if d > 20:
        # Use average nearest neighbor distance as proxy
        try:
            nn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
            distances, _ = nn.kneighbors(points)
            avg_nn_dist = np.mean(distances[:, 1])
            # Normalized by dimension
            return min(1.0, avg_nn_dist * np.sqrt(d))
        except:
            return np.nan

    # Use minimum distance and convex hull volume for lower dimensions
    min_dist = maximin_distance(points)
    hull_vol = convex_hull_volume(points)

    if np.isnan(hull_vol) or hull_vol == 0:
        return 0.0

    # Estimate volume occupied by spheres of radius min_dist/2
    try:
        sphere_vol = np.pi ** (d / 2) / np.math.gamma(d / 2 + 1) * (
                    min_dist / 2) ** d
        total_sphere_vol = len(points) * sphere_vol
        return min(total_sphere_vol / hull_vol, 1.0)
    except (OverflowError, ValueError):
        return np.nan


def high_dim_coverage_metrics(points, n_samples=5000):
    """
    Specialized coverage metrics for high-dimensional data.

    Args:
        points (np.ndarray): High-dimensional points
        n_samples (int): Number of samples (reduced for efficiency)

    Returns:
        dict: High-dimensional specific metrics
    """
    if len(points) == 0:
        return {}

    metrics = {}
    d = points.shape[1]
    n = points.shape[0]

    # 1. Average pairwise distance (more meaningful in high-D)
    if n <= 5000:  # Only for reasonable dataset sizes
        sample_idx = np.random.choice(n, min(1000, n), replace=False)
        sample_points = points[sample_idx]
        nn = NearestNeighbors(n_neighbors=min(10, len(sample_points)),
                              algorithm='auto').fit(sample_points)
        distances, _ = nn.kneighbors(sample_points)
        metrics['avg_k_nearest_distance'] = np.mean(
            distances[:, 1:])  # Exclude self
    else:
        metrics['avg_k_nearest_distance'] = np.nan

    # 2. Dimension-normalized dispersion
    dispersion_val = dispersion(points, min(n_samples, 5000))
    metrics['normalized_dispersion'] = dispersion_val * np.sqrt(d)

    # 3. Effective dimension (participation ratio)
    # Measure how many dimensions are "actively used"
    try:
        pca_variances = np.var(points, axis=0)
        pca_variances = pca_variances[
            pca_variances > 1e-10]  # Remove near-zero variances
        if len(pca_variances) > 0:
            participation_ratio = (np.sum(pca_variances) ** 2) / np.sum(
                pca_variances ** 2)
            metrics['effective_dimension'] = participation_ratio / d
        else:
            metrics['effective_dimension'] = 0.0
    except:
        metrics['effective_dimension'] = np.nan

    # 4. Coordinate-wise uniformity (per-dimension KS test approximation)
    coord_uniformity = []
    for dim in range(min(d, 50)):  # Sample first 50 dimensions
        coord_vals = points[:, dim]
        # Simple uniformity check: compare quantiles
        expected_quantiles = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        actual_quantiles = np.quantile(coord_vals, expected_quantiles)
        uniformity_score = 1.0 - np.mean(
            np.abs(actual_quantiles - expected_quantiles))
        coord_uniformity.append(max(0, uniformity_score))

    metrics['avg_coordinate_uniformity'] = np.mean(
        coord_uniformity) if coord_uniformity else np.nan

    return metrics


def compute_all_metrics(points, n_samples=10000, n_test=10000,
                        high_dim_mode=None):
    """
    Compute all coverage metrics for the given points.
    Automatically detects high-dimensional data and adjusts accordingly.

    Args:
        points (np.ndarray): Array of points
        n_samples (int): Number of samples for star discrepancy
        n_test (int): Number of test points for dispersion estimation
        high_dim_mode (bool): Force high-dimensional mode (auto-detected if None)

    Returns:
        dict: Dictionary of all computed metrics
    """
    metrics = {}

    try:
        metrics['n_points'] = len(points)
        metrics['dimension'] = points.shape[1] if len(points) > 0 else 0

        if len(points) == 0:
            return {k: np.nan for k in ['star_discrepancy', 'maximin_distance',
                                        'dispersion', 'separation_distance',
                                        'fill_distance', 'mesh_ratio',
                                        'convex_hull_volume',
                                        'coverage_radius',
                                        'packing_efficiency']}

        d = points.shape[1]

        # Auto-detect high-dimensional mode
        if high_dim_mode is None:
            high_dim_mode = d > 50

        if high_dim_mode:
            print(f"High-dimensional mode activated for {d}D data")
            # Reduce sample sizes for efficiency
            n_samples = min(n_samples, 5000)
            n_test = min(n_test, 5000)

        # Standard metrics (adjusted for high dimensions)
        #metrics['star_discrepancy'] = star_discrepancy_proxy(points, n_samples)
        metrics['maximin_distance'] = maximin_distance(points)
        metrics['dispersion'] = dispersion(points, n_test)
        #metrics['separation_distance'] = separation_distance(points)
        #metrics['fill_distance'] = fill_distance_estimate(points, n_test)
        metrics['mesh_ratio'] = mesh_ratio(metrics['dispersion'], metrics['maximin_distance'])
        metrics['convex_hull_volume'] = convex_hull_volume(points)
        metrics['coverage_radius'] = coverage_radius(points)
        metrics['packing_efficiency'] = packing_efficiency(points)

        # High-dimensional specific metrics
        if high_dim_mode:
            hd_metrics = high_dim_coverage_metrics(points, n_samples)
            metrics.update(hd_metrics)

    except Exception as e:
        warnings.warn(f"Error computing metrics: {e}")

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of computed metrics
    """
    print("=" * 60)
    print("COVERAGE METRICS ANALYSIS")
    print("=" * 60)
    print(f"Dataset size: {metrics.get('n_points', 'N/A')} points")
    print(f"Dimension: {metrics.get('dimension', 'N/A')}")

    d = metrics.get('dimension', 0)
    if d > 50:
        print(f"HIGH-DIMENSIONAL DATA DETECTED (d={d})")
        print("Some metrics may be less reliable in high dimensions.")

    print("-" * 60)

    # Standard metrics
    print("STANDARD COVERAGE METRICS:")
    print(
        f"  Star discrepancy (approx): {metrics.get('star_discrepancy', np.nan):.6f}")
    print(f"  Maximin distance: {metrics.get('maximin_distance', np.nan):.6f}")
    print(
        f"  Dispersion (fill distance): {metrics.get('dispersion', np.nan):.6f}")
    print(
        f"  Separation distance: {metrics.get('separation_distance', np.nan):.6f}")
    print(f"  Mesh ratio: {metrics.get('mesh_ratio', np.nan):.6f}")

    hull_vol = metrics.get('convex_hull_volume', np.nan)
    if np.isnan(hull_vol):
        print("  Convex hull volume: Not computable")
    else:
        print(f"  Convex hull volume: {hull_vol:.6f}")

    print(f"  Coverage radius: {metrics.get('coverage_radius', np.nan):.6f}")
    print(
        f"  Packing efficiency: {metrics.get('packing_efficiency', np.nan):.6f}")

    # High-dimensional specific metrics
    if 'avg_k_nearest_distance' in metrics:
        print("\nHIGH-DIMENSIONAL SPECIFIC METRICS:")
        print(
            f"  Avg k-nearest distance: {metrics.get('avg_k_nearest_distance', np.nan):.6f}")
        print(
            f"  Normalized dispersion: {metrics.get('normalized_dispersion', np.nan):.6f}")
        print(
            f"  Effective dimension ratio: {metrics.get('effective_dimension', np.nan):.6f}")
        print(
            f"  Avg coordinate uniformity: {metrics.get('avg_coordinate_uniformity', np.nan):.6f}")

    print("-" * 60)
    print("INTERPRETATION:")
    print("• Lower star discrepancy = more uniform distribution")
    print("• Higher maximin distance = better point separation")
    print("• Lower dispersion = better space filling")
    print("• Lower mesh ratio = better overall distribution quality")

    if d > 50:
        print("\nHIGH-DIMENSIONAL INTERPRETATION:")
        print(
            "• Effective dimension ratio: fraction of dimensions actively used")
        print("• Coordinate uniformity: how uniform each dimension is (0-1)")
        print("• Normalized dispersion: dispersion adjusted for dimension")
        print("• Note: Traditional metrics may be less meaningful in high-D")

    if d > 100:
        print(f"\nWARNING: Very high dimension ({d}D)")
        print(
            "Consider dimensionality reduction or focus on coordinate-wise metrics")


if __name__ == "__main__":
    csv_file = "../results/datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664/cases_df.csv"

    try:
        # Read and preprocess data
        X = read_data(csv_file)
        print(f"Loaded data: {X.shape[0]} points in {X.shape[1]} dimensions")

        # Compute all metrics with automatic high-dimensional detection
        metrics = compute_all_metrics(X, n_samples=10000, n_test=10000)

        # Print results
        print_metrics(metrics)

        # Save results if needed
        # pd.DataFrame([metrics]).to_csv("coverage_metrics_results.csv", index=False)

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your CSV file path and data format.")

        print("-" * 50)
        print("INTERPRETATION:")
        print("• Lower star discrepancy = more uniform distribution")
        print("• Higher maximin distance = better point separation")
        print("• Lower dispersion = better space filling")
        print("• Lower mesh ratio = better overall distribution quality")
        print("• Higher packing efficiency = more efficient space usage")

        if __name__ == "__main__":
            csv_file = "../results/cases_df.csv"

    try:
        # Read and preprocess data
        X = read_data(csv_file)

        # Compute all metrics
        metrics = compute_all_metrics(X, n_samples=10000, n_test=10000)

        # Print results
        print_metrics(metrics)

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your CSV file path and data format.")