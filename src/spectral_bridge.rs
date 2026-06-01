//! Spectral bridge — normalize Laplacian spectrums across crates so they can
//! be compared.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A normalized spectrum from a Laplacian matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedSpectrum {
    /// The crate this spectrum came from.
    pub source_crate: String,
    /// Eigenvalues sorted ascending, normalized to [0, 1].
    pub eigenvalues: Vec<f64>,
    /// The algebraic connectivity (second-smallest eigenvalue, pre-normalization).
    pub algebraic_connectivity: f64,
    /// Number of vertices in the original graph.
    pub n_vertices: usize,
}

/// Comparison result between two spectra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralComparison {
    pub crate_a: String,
    pub crate_b: String,
    /// Cosine similarity of the eigenvalue vectors.
    pub cosine_similarity: f64,
    /// Earth mover's distance between the spectra.
    pub earth_movers_distance: f64,
    /// Whether the spectral gaps are compatible.
    pub gaps_compatible: bool,
}

/// The spectral bridge normalizer.
pub struct SpectralBridge;

impl SpectralBridge {
    /// Compute eigenvalues of a symmetric matrix using power iteration
    /// for the largest and a simple approach. For real use, you'd want
    /// a proper eigensolver — this gives a reasonable approximation.
    pub fn compute_spectrum(laplacian: &DMatrix<f64>) -> Vec<f64> {
        let n = laplacian.nrows();
        if n == 0 {
            return vec![];
        }
        // Use the fact that for a Laplacian, eigenvalues are in [0, 2*max_degree].
        // We compute them via the characteristic polynomial approach for small matrices,
        // or via iterative deflation for larger ones.
        let mut eigenvalues = Vec::new();
        let mut current = laplacian.clone();

        for _ in 0..n {
            if let Some(ev) = Self::power_iteration(&current) {
                eigenvalues.push(ev);
                // Deflate: subtract the rank-1 component
                let (vec, _) = Self::power_iteration_vec(&current);
                let norm_sq = vec.dot(&vec);
                if norm_sq > 1e-15 {
                    let projected = &vec * (vec.transpose() * &current);
                    current = &current - projected;
                }
            } else {
                eigenvalues.push(0.0);
            }
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Power iteration to find the dominant eigenvalue.
    pub fn power_iteration(matrix: &DMatrix<f64>) -> Option<f64> {
        let (_, val) = Self::power_iteration_vec(matrix);
        Some(val)
    }

    /// Power iteration returning eigenvector and eigenvalue.
    fn power_iteration_vec(matrix: &DMatrix<f64>) -> (DVector<f64>, f64) {
        let n = matrix.nrows();
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
        let mut eigenvalue = 0.0;

        for _ in 0..200 {
            let mv = matrix * &v;
            let new_eigenvalue = v.dot(&mv);
            let norm = mv.norm();
            if norm < 1e-15 {
                return (v, 0.0);
            }
            v = mv / norm;
            if (new_eigenvalue - eigenvalue).abs() < 1e-12 {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        (v, eigenvalue)
    }

    /// Normalize eigenvalues to [0, 1].
    pub fn normalize(eigenvalues: &[f64]) -> Vec<f64> {
        if eigenvalues.is_empty() {
            return vec![];
        }
        let max_val = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_val - min_val;
        if range < 1e-15 {
            return vec![0.0; eigenvalues.len()];
        }
        eigenvalues.iter().map(|e| (e - min_val) / range).collect()
    }

    /// Create a NormalizedSpectrum from a Laplacian matrix.
    pub fn spectrum_from_laplacian(
        laplacian: &DMatrix<f64>,
        source_crate: &str,
    ) -> NormalizedSpectrum {
        let eigenvalues = Self::compute_spectrum(laplacian);
        let n = eigenvalues.len();
        let algebraic_connectivity = if n >= 2 { eigenvalues[1] } else { 0.0 };
        let normalized = Self::normalize(&eigenvalues);

        NormalizedSpectrum {
            source_crate: source_crate.to_string(),
            eigenvalues: normalized,
            algebraic_connectivity,
            n_vertices: n,
        }
    }

    /// Compare two normalized spectra.
    pub fn compare(a: &NormalizedSpectrum, b: &NormalizedSpectrum) -> SpectralComparison {
        let cosine = Self::cosine_similarity(&a.eigenvalues, &b.eigenvalues);
        let emd = Self::earth_movers_distance(&a.eigenvalues, &b.eigenvalues);

        // Check if spectral gaps are compatible (within tolerance).
        let gap_a = Self::spectral_gap(&a.eigenvalues);
        let gap_b = Self::spectral_gap(&b.eigenvalues);
        let gaps_compatible = if gap_a.is_finite() && gap_b.is_finite() {
            (gap_a - gap_b).abs() < 0.3
        } else {
            false
        };

        SpectralComparison {
            crate_a: a.source_crate.clone(),
            crate_b: b.source_crate.clone(),
            cosine_similarity: cosine,
            earth_movers_distance: emd,
            gaps_compatible,
        }
    }

    /// Cosine similarity between two vectors, padding shorter one with zeros.
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let max_len = a.len().max(b.len());
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..max_len {
            let va = if i < a.len() { a[i] } else { 0.0 };
            let vb = if i < b.len() { b[i] } else { 0.0 };
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Earth mover's distance (simplified: L1 distance on sorted distributions).
    fn earth_movers_distance(a: &[f64], b: &[f64]) -> f64 {
        let max_len = a.len().max(b.len());
        let mut emd = 0.0;
        let mut cumulative = 0.0;
        for i in 0..max_len {
            let va = if i < a.len() { a[i] } else { 0.0 };
            let vb = if i < b.len() { b[i] } else { 0.0 };
            cumulative += va - vb;
            emd += cumulative.abs();
        }
        emd
    }

    /// Spectral gap: difference between second and first eigenvalues.
    fn spectral_gap(eigenvalues: &[f64]) -> f64 {
        if eigenvalues.len() >= 2 {
            eigenvalues[1] - eigenvalues[0]
        } else {
            f64::NAN
        }
    }

    /// Normalize two spectra to the same dimension by truncating or padding.
    pub fn align_dimensions(a: &NormalizedSpectrum, b: &NormalizedSpectrum) -> (Vec<f64>, Vec<f64>) {
        let min_len = a.eigenvalues.len().min(b.eigenvalues.len());
        let aligned_a: Vec<f64> = a.eigenvalues[..min_len].to_vec();
        let aligned_b: Vec<f64> = b.eigenvalues[..min_len].to_vec();
        (aligned_a, aligned_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_3x3() -> DMatrix<f64> {
        DMatrix::identity(3, 3)
    }

    fn laplacian_path_3() -> DMatrix<f64> {
        // Path graph on 3 nodes
        DMatrix::from_row_slice(3, 3, &[
            1.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 1.0,
        ])
    }

    fn laplacian_complete_3() -> DMatrix<f64> {
        // Complete graph on 3 nodes
        DMatrix::from_row_slice(3, 3, &[
            2.0, -1.0, -1.0,
            -1.0, 2.0, -1.0,
            -1.0, -1.0, 2.0,
        ])
    }

    #[test]
    fn test_compute_spectrum_identity() {
        let spec = SpectralBridge::compute_spectrum(&identity_3x3());
        assert_eq!(spec.len(), 3);
        // Dominant eigenvalue should be ~1
        assert!((spec[2] - 1.0).abs() < 0.2, "dominant eigenvalue too far from 1: {}", spec[2]);
    }

    #[test]
    fn test_compute_spectrum_laplacian_path() {
        let spec = SpectralBridge::compute_spectrum(&laplacian_path_3());
        // First eigenvalue should be ~0
        assert!(spec[0].abs() < 0.5, "first eigenvalue should be near 0, got {}", spec[0]);
    }

    #[test]
    fn test_normalize_spectrum() {
        let vals = vec![0.0, 1.0, 2.0, 3.0];
        let normed = SpectralBridge::normalize(&vals);
        assert!((normed[0] - 0.0).abs() < 1e-10);
        assert!((normed[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_empty() {
        assert!(SpectralBridge::normalize(&[]).is_empty());
    }

    #[test]
    fn test_normalize_constant() {
        let vals = vec![5.0, 5.0, 5.0];
        let normed = SpectralBridge::normalize(&vals);
        assert!(normed.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_spectrum_from_laplacian() {
        let spec = SpectralBridge::spectrum_from_laplacian(&laplacian_path_3(), "test_crate");
        assert_eq!(spec.n_vertices, 3);
        assert_eq!(spec.source_crate, "test_crate");
    }

    #[test]
    fn test_compare_identical_spectra() {
        let spec_a = NormalizedSpectrum {
            source_crate: "a".into(),
            eigenvalues: vec![0.0, 0.5, 1.0],
            algebraic_connectivity: 1.0,
            n_vertices: 3,
        };
        let comp = SpectralBridge::compare(&spec_a, &spec_a);
        assert!((comp.cosine_similarity - 1.0).abs() < 1e-10);
        assert!(comp.earth_movers_distance.abs() < 1e-10);
    }

    #[test]
    fn test_compare_different_spectra() {
        let spec_a = NormalizedSpectrum {
            source_crate: "a".into(),
            eigenvalues: vec![0.0, 0.1, 1.0],
            algebraic_connectivity: 0.1,
            n_vertices: 3,
        };
        let spec_b = NormalizedSpectrum {
            source_crate: "b".into(),
            eigenvalues: vec![0.0, 0.9, 1.0],
            algebraic_connectivity: 0.9,
            n_vertices: 3,
        };
        let comp = SpectralBridge::compare(&spec_a, &spec_b);
        assert!(comp.cosine_similarity > 0.0);
        assert!(comp.cosine_similarity < 1.0);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = SpectralBridge::cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_earth_movers_distance_same() {
        let emd = SpectralBridge::earth_movers_distance(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(emd.abs() < 1e-10);
    }

    #[test]
    fn test_align_dimensions() {
        let a = NormalizedSpectrum {
            source_crate: "a".into(),
            eigenvalues: vec![0.0, 0.5, 1.0, 0.3],
            algebraic_connectivity: 0.5,
            n_vertices: 4,
        };
        let b = NormalizedSpectrum {
            source_crate: "b".into(),
            eigenvalues: vec![0.1, 0.2],
            algebraic_connectivity: 0.1,
            n_vertices: 2,
        };
        let (aa, bb) = SpectralBridge::align_dimensions(&a, &b);
        assert_eq!(aa.len(), 2);
        assert_eq!(bb.len(), 2);
    }
}
