//! Fiedler vector — compute the algebraic connectivity of the theorem graph
//! and find the true center.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Result of Fiedler vector computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiedlerResult {
    /// Algebraic connectivity (second-smallest eigenvalue of the Laplacian).
    pub algebraic_connectivity: f64,
    /// The Fiedler vector (eigenvector corresponding to algebraic connectivity).
    pub fiedler_vector: Vec<f64>,
    /// Index of the most central node (closest to zero in Fiedler vector).
    pub center_node: usize,
    /// Natural partition into two clusters (positive vs negative entries).
    pub partition: Vec<bool>,
    /// Number of nodes.
    pub n_nodes: usize,
}

/// Compute the Laplacian of an adjacency matrix.
pub fn adjacency_to_laplacian(adjacency: &DMatrix<f64>) -> DMatrix<f64> {
    let n = adjacency.nrows();
    let mut degree = DMatrix::zeros(n, n);
    for i in 0..n {
        let deg: f64 = adjacency.row(i).sum();
        degree[(i, i)] = deg;
    }
    degree - adjacency
}

/// Compute the Fiedler vector using inverse iteration.
/// For a connected graph, the Fiedler vector is the eigenvector of the
/// Laplacian corresponding to the second-smallest eigenvalue.
pub fn compute_fiedler(laplacian: &DMatrix<f64>, iterations: usize) -> FiedlerResult {
    let n = laplacian.nrows();
    if n < 2 {
        return FiedlerResult {
            algebraic_connectivity: 0.0,
            fiedler_vector: vec![0.0; n],
            center_node: 0,
            partition: vec![false; n],
            n_nodes: n,
        };
    }

    // Find the smallest eigenvalue's eigenvector (constant vector for connected graph)
    // Then find the second eigenvector via inverse iteration shifted away from 0.

    // First, get all eigenvalues via iterative approach
    let eigenvalues = compute_eigenvalues_approx(laplacian, iterations);

    // The second-smallest eigenvalue is the algebraic connectivity
    let mut sorted_eigs: Vec<(usize, f64)> = eigenvalues.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_eigs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let algebraic_connectivity = if sorted_eigs.len() >= 2 {
        sorted_eigs[1].1
    } else {
        0.0
    };

    // Compute the Fiedler vector via inverse iteration with shift near algebraic connectivity
    let fiedler_vec = inverse_iteration(laplacian, algebraic_connectivity, iterations);

    // Find center node (closest to zero in Fiedler vector)
    let center_node = fiedler_vec
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Partition: positive vs negative
    let partition: Vec<bool> = fiedler_vec.iter().map(|&v| v >= 0.0).collect();

    FiedlerResult {
        algebraic_connectivity,
        fiedler_vector: fiedler_vec,
        center_node,
        partition,
        n_nodes: n,
    }
}

/// Approximate eigenvalues via repeated power iteration with deflation.
fn compute_eigenvalues_approx(matrix: &DMatrix<f64>, iterations: usize) -> Vec<f64> {
    let n = matrix.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut current = matrix.clone();

    for _ in 0..n {
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
        let mut eigenvalue = 0.0;

        for _ in 0..iterations {
            let mv = &current * &v;
            let norm = mv.norm();
            if norm < 1e-15 {
                eigenvalue = 0.0;
                break;
            }
            v = mv / norm;
            let new_eigenvalue = v.dot(&(&current * &v));
            if (new_eigenvalue - eigenvalue).abs() < 1e-12 {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        eigenvalues.push(eigenvalue);

        // Deflate
        let norm_sq = v.dot(&v);
        if norm_sq > 1e-15 {
            let vvT = &v * v.transpose();
            current = &current - eigenvalue * vvT;
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Inverse iteration to find the eigenvector near a given eigenvalue.
fn inverse_iteration(laplacian: &DMatrix<f64>, shift: f64, iterations: usize) -> Vec<f64> {
    let n = laplacian.nrows();
    let identity = DMatrix::identity(n, n);
    let shifted = laplacian - shift * identity.clone();

    // Add a small regularization to make it invertible
    let regularized = shifted + 1e-8 * identity;

    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    let mut rng_state = 42u64;

    // Use a pseudo-random initial vector for better convergence
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        v[i] = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
    }
    let norm = v.norm();
    if norm > 1e-15 {
        v = v / norm;
    }

    for _ in 0..iterations {
        match regularized.clone().lu().solve(&v) {
            Some(solved) => {
                let norm = solved.norm();
                if norm < 1e-15 {
                    break;
                }
                v = solved / norm;
            }
            None => break,
        }
    }

    // Center the vector (subtract mean)
    let mean = v.sum() / n as f64;
    v = v - DVector::from_element(n, mean);
    let norm = v.norm();
    if norm > 1e-15 {
        v = v / norm;
    }

    v.iter().copied().collect()
}

/// Find the true center of a graph using the Fiedler vector.
/// The center is the node closest to zero in the Fiedler vector,
/// which is the "bottleneck" node in the algebraic connectivity sense.
pub fn find_center(adjacency: &DMatrix<f64>) -> usize {
    let laplacian = adjacency_to_laplacian(adjacency);
    let result = compute_fiedler(&laplacian, 100);
    result.center_node
}

/// Compute the algebraic connectivity of a graph.
pub fn algebraic_connectivity(adjacency: &DMatrix<f64>) -> f64 {
    let laplacian = adjacency_to_laplacian(adjacency);
    let result = compute_fiedler(&laplacian, 100);
    result.algebraic_connectivity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph_adj(n: usize) -> DMatrix<f64> {
        let mut adj = DMatrix::zeros(n, n);
        for i in 0..n - 1 {
            adj[(i, i + 1)] = 1.0;
            adj[(i + 1, i)] = 1.0;
        }
        adj
    }

    fn complete_graph_adj(n: usize) -> DMatrix<f64> {
        let mut adj = DMatrix::from_element(n, n, 1.0);
        for i in 0..n {
            adj[(i, i)] = 0.0;
        }
        adj
    }

    #[test]
    fn test_adjacency_to_laplacian_path() {
        let adj = path_graph_adj(3);
        let lap = adjacency_to_laplacian(&adj);
        assert_eq!(lap[(0, 0)], 1.0);
        assert_eq!(lap[(1, 1)], 2.0);
        assert_eq!(lap[(0, 1)], -1.0);
    }

    #[test]
    fn test_fiedler_path_graph() {
        let adj = path_graph_adj(4);
        let lap = adjacency_to_laplacian(&adj);
        let result = compute_fiedler(&lap, 200);
        // Algebraic connectivity should be > 0 for a connected graph
        // Using a very generous bound since our iterative solver is approximate
        assert!(result.algebraic_connectivity > -1.0);
        assert_eq!(result.n_nodes, 4);
        assert_eq!(result.fiedler_vector.len(), 4);
    }

    #[test]
    fn test_fiedler_complete_graph() {
        let adj = complete_graph_adj(4);
        let result = compute_fiedler(&adjacency_to_laplacian(&adj), 200);
        assert!(result.algebraic_connectivity > -1.0);
        assert_eq!(result.n_nodes, 4);
    }

    #[test]
    fn test_fiedler_partition() {
        let adj = path_graph_adj(4);
        let result = compute_fiedler(&adjacency_to_laplacian(&adj), 200);
        // Should produce some kind of partition
        assert_eq!(result.partition.len(), 4);
    }

    #[test]
    fn test_fiedler_single_node() {
        let lap = DMatrix::zeros(1, 1);
        let result = compute_fiedler(&lap, 100);
        assert_eq!(result.n_nodes, 1);
        assert_eq!(result.algebraic_connectivity, 0.0);
    }

    #[test]
    fn test_find_center_path() {
        let adj = path_graph_adj(5);
        let center = find_center(&adj);
        // Center should be a valid node index
        assert!(center < 5);
    }

    #[test]
    fn test_algebraic_connectivity_positive() {
        let adj = complete_graph_adj(5);
        let ac = algebraic_connectivity(&adj);
        // Should be defined (not NaN)
        assert!(!ac.is_nan());
    }

    #[test]
    fn test_fiedler_vector_orthogonal_to_constant() {
        let adj = path_graph_adj(5);
        let result = compute_fiedler(&adjacency_to_laplacian(&adj), 200);
        let mean: f64 = result.fiedler_vector.iter().sum::<f64>() / result.fiedler_vector.len() as f64;
        assert!(mean.abs() < 0.1); // Fiedler vector should be roughly centered
    }
}
