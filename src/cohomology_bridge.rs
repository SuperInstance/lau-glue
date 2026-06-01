//! Cohomology bridge — connect H¹ computations across sheaf-automata,
//! reward-hacking-detector, and observation-control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A cochain complex from a specific crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohainComplex {
    pub source_crate: String,
    pub dimension: usize,
    /// Coboundary map as a flattened matrix (dim × dim).
    pub coboundary_map: Vec<f64>,
}

/// The result of an H¹ computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohomologyGroup {
    pub source_crate: String,
    /// Dimension of H¹ (first cohomology group).
    pub betti_number: usize,
    /// Representative cocycles (basis for H¹).
    pub cocycles: Vec<Vec<f64>>,
}

/// A bridge connecting H¹ computations between two crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohomologyBridge {
    pub from: String,
    pub to: String,
    /// The connecting map in cohomology.
    pub connecting_map: Vec<f64>,
    /// Whether the bridge is exact (connecting map is zero in cohomology).
    pub is_exact: bool,
    /// Rank of the connecting map.
    pub rank: usize,
}

/// The cohomology bridge builder.
pub struct CohomologyBridgeBuilder;

impl CohomologyBridgeBuilder {
    /// Compute the rank of a matrix using Gaussian elimination.
    pub fn matrix_rank(matrix: &DMatrix<f64>, tolerance: f64) -> usize {
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        let mut m = matrix.clone();
        let mut rank = 0;
        let mut pivot_row = 0;

        for col in 0..ncols {
            if pivot_row >= nrows {
                break;
            }
            // Find the row with the largest value in this column
            let mut max_val = m[(pivot_row, col)].abs();
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..nrows {
                let val = m[(row, col)].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < tolerance {
                continue; // Skip this column
            }

            // Swap rows
            if max_row != pivot_row {
                for c in 0..ncols {
                    let tmp = m[(pivot_row, c)];
                    m[(pivot_row, c)] = m[(max_row, c)];
                    m[(max_row, c)] = tmp;
                }
            }

            // Eliminate below
            for row in (pivot_row + 1)..nrows {
                let factor = m[(row, col)] / m[(pivot_row, col)];
                for c in 0..ncols {
                    m[(row, c)] -= factor * m[(pivot_row, c)];
                }
            }

            rank += 1;
            pivot_row += 1;
        }

        rank
    }

    /// Compute H¹ from a coboundary map: H¹ = ker(d₁) / im(d₀).
    /// Simplified: returns the nullity minus the rank deficiency.
    pub fn compute_h1(coboundary: &DMatrix<f64>) -> CohomologyGroup {
        let n = coboundary.nrows();
        let m = coboundary.ncols();

        // Rank of the coboundary
        let rank = Self::matrix_rank(coboundary, 1e-10);

        // Nullity = m - rank
        let nullity = m.saturating_sub(rank);

        // H¹ dimension ≈ nullity (simplified; real H¹ requires two coboundary maps)
        let betti = nullity;

        // Generate representative cocycles
        let cocycles = Self::find_null_space_vectors(coboundary, betti);

        CohomologyGroup {
            source_crate: String::new(),
            betti_number: betti,
            cocycles,
        }
    }

    /// Find basis vectors for the null space of a matrix.
    fn find_null_space_vectors(matrix: &DMatrix<f64>, max_vectors: usize) -> Vec<Vec<f64>> {
        let m = matrix.ncols();
        let mut cocycles = Vec::new();

        for k in 0..max_vectors.min(m) {
            let mut v = vec![0.0; m];
            if m > 0 {
                v[k % m] = 1.0;
            }
            // Verify it's in the kernel
            let dv = matrix * DVector::from_vec(v.clone());
            if dv.iter().all(|x| x.abs() < 1e-8) {
                cocycles.push(v);
            }
        }

        cocycles
    }

    /// Build a bridge between two cohomology groups.
    pub fn build_bridge(from: &CohomologyGroup, to: &CohomologyGroup) -> CohomologyBridge {
        let from_dim = from.cocycles.len();
        let to_dim = to.cocycles.len();
        let dim = from_dim.max(to_dim).max(1);

        // Connecting map: identity-like (simplified)
        let connecting_map: Vec<f64> = (0..dim * dim)
            .map(|i| {
                let row = i / dim;
                let col = i % dim;
                if row == col && row < from_dim.min(to_dim) { 1.0 } else { 0.0 }
            })
            .collect();

        let rank = from_dim.min(to_dim);
        let is_exact = rank == 0;

        CohomologyBridge {
            from: from.source_crate.clone(),
            to: to.source_crate.clone(),
            connecting_map,
            is_exact,
            rank,
        }
    }

    /// Build all pairwise bridges.
    pub fn build_all_bridges(groups: &[CohomologyGroup]) -> Vec<CohomologyBridge> {
        let mut bridges = Vec::new();
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                bridges.push(Self::build_bridge(&groups[i], &groups[j]));
            }
        }
        bridges
    }

    /// Mayer-Vietoris sequence: given two sheaves with overlap,
    /// compute the connecting homomorphism in the long exact sequence.
    pub fn mayer_vietoris(
        h1_a: &CohomologyGroup,
        h1_b: &CohomologyGroup,
        h0_overlap: &CohomologyGroup,
    ) -> CohomologyGroup {
        // ∂: H⁰(overlap) → H¹(A∪B)
        // In the MV sequence: ... → H⁰(A)⊕H⁰(B) → H⁰(overlap) → H¹(A∪B) → H¹(A)⊕H¹(B) → ...
        // Simplified: dim H¹(A∪B) = dim H¹(A) + dim H¹(B) + dim H⁰(overlap) - corrections
        let betti = h1_a.betti_number + h1_b.betti_number + h0_overlap.betti_number;

        let mut cocycles = Vec::new();
        cocycles.extend_from_slice(&h1_a.cocycles);
        cocycles.extend_from_slice(&h1_b.cocycles);

        CohomologyGroup {
            source_crate: format!("{}+{}", h1_a.source_crate, h1_b.source_crate),
            betti_number: betti,
            cocycles,
        }
    }

    /// Compute Euler characteristic from Betti numbers.
    pub fn euler_characteristic(betti_numbers: &[usize]) -> i64 {
        betti_numbers
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_rank_identity() {
        let m = DMatrix::identity(3, 3);
        assert_eq!(CohomologyBridgeBuilder::matrix_rank(&m, 1e-10), 3);
    }

    #[test]
    fn test_matrix_rank_zero() {
        let m = DMatrix::from_element(3, 3, 0.0);
        assert_eq!(CohomologyBridgeBuilder::matrix_rank(&m, 1e-10), 0);
    }

    #[test]
    fn test_matrix_rank_singular() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 1.0,
            2.0, 0.0, 2.0,
        ]);
        // Row 2 = 2 * Row 1, so rank = 1
        assert_eq!(CohomologyBridgeBuilder::matrix_rank(&m, 1e-10), 1);
    }

    #[test]
    fn test_compute_h1_zero_map() {
        let m = DMatrix::from_element(3, 3, 0.0);
        let h1 = CohomologyBridgeBuilder::compute_h1(&m);
        // Nullity of zero map = 3
        assert_eq!(h1.betti_number, 3);
    }

    #[test]
    fn test_compute_h1_full_rank() {
        let m = DMatrix::identity(3, 3);
        let h1 = CohomologyBridgeBuilder::compute_h1(&m);
        assert_eq!(h1.betti_number, 0);
    }

    #[test]
    fn test_build_bridge() {
        let a = CohomologyGroup {
            source_crate: "sheaf_automata".into(),
            betti_number: 2,
            cocycles: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let b = CohomologyGroup {
            source_crate: "reward_hacking".into(),
            betti_number: 1,
            cocycles: vec![vec![1.0]],
        };
        let bridge = CohomologyBridgeBuilder::build_bridge(&a, &b);
        assert_eq!(bridge.rank, 1);
        assert!(!bridge.is_exact);
    }

    #[test]
    fn test_build_bridge_exact() {
        let a = CohomologyGroup {
            source_crate: "a".into(),
            betti_number: 0,
            cocycles: vec![],
        };
        let b = CohomologyGroup {
            source_crate: "b".into(),
            betti_number: 0,
            cocycles: vec![],
        };
        let bridge = CohomologyBridgeBuilder::build_bridge(&a, &b);
        assert!(bridge.is_exact);
        assert_eq!(bridge.rank, 0);
    }

    #[test]
    fn test_build_all_bridges() {
        let groups = vec![
            CohomologyGroup { source_crate: "a".into(), betti_number: 1, cocycles: vec![vec![1.0]] },
            CohomologyGroup { source_crate: "b".into(), betti_number: 1, cocycles: vec![vec![1.0]] },
            CohomologyGroup { source_crate: "c".into(), betti_number: 1, cocycles: vec![vec![1.0]] },
        ];
        let bridges = CohomologyBridgeBuilder::build_all_bridges(&groups);
        assert_eq!(bridges.len(), 3); // C(3,2)
    }

    #[test]
    fn test_mayer_vietoris() {
        let h1_a = CohomologyGroup {
            source_crate: "A".into(),
            betti_number: 2,
            cocycles: vec![vec![1.0], vec![0.0]],
        };
        let h1_b = CohomologyGroup {
            source_crate: "B".into(),
            betti_number: 1,
            cocycles: vec![vec![1.0]],
        };
        let h0_overlap = CohomologyGroup {
            source_crate: "overlap".into(),
            betti_number: 1,
            cocycles: vec![vec![1.0]],
        };
        let mv = CohomologyBridgeBuilder::mayer_vietoris(&h1_a, &h1_b, &h0_overlap);
        assert_eq!(mv.betti_number, 4); // 2 + 1 + 1
        assert_eq!(mv.cocycles.len(), 3);
    }

    #[test]
    fn test_euler_characteristic() {
        // Point: b0=1 → χ=1
        assert_eq!(CohomologyBridgeBuilder::euler_characteristic(&[1]), 1);
        // Circle: b0=1, b1=1 → χ=0
        assert_eq!(CohomologyBridgeBuilder::euler_characteristic(&[1, 1]), 0);
        // Sphere: b0=1, b1=0, b2=1 → χ=2
        assert_eq!(CohomologyBridgeBuilder::euler_characteristic(&[1, 0, 1]), 2);
    }
}
