//! Conservation bridge — compose Noether + CALM + Landauer conservation laws
//! into one invariant.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A conservation law from a specific mathematical framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationLaw {
    pub name: String,
    pub source: ConservationSource,
    pub quantity: String,
    pub description: String,
    /// The conserved value (if constant) or None if it varies.
    pub conserved_value: Option<f64>,
}

/// Where a conservation law comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConservationSource {
    /// Noether's theorem: symmetry → conservation law.
    Noether { symmetry: String },
    /// CALM (Continuous Active Learning Model).
    CALM,
    /// Landauer's principle: information-thermodynamic bound.
    Landauer,
}

/// A composed invariant from multiple conservation laws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedInvariant {
    pub laws: Vec<ConservationLaw>,
    pub total_value: f64,
    pub is_conserved: bool,
    pub description: String,
}

/// The conservation bridge composer.
pub struct ConservationBridge;

impl ConservationBridge {
    /// Landauer's limit: minimum energy to erase one bit at temperature T.
    /// E ≥ kT ln(2)
    pub fn landauer_limit(temperature_kelvin: f64) -> f64 {
        const K_BOLTZMANN: f64 = 1.380649e-23; // J/K
        K_BOLTZMANN * temperature_kelvin * (2.0_f64).ln()
    }

    /// Noether conservation: given a symmetry group, return the conserved
    /// quantity value.
    pub fn noether_conserve(
        symmetry: &str,
        state: &DVector<f64>,
        hamiltonian: &DVector<f64>,
    ) -> f64 {
        match symmetry {
            "time_translation" => {
                // Energy: H · state
                hamiltonian.dot(state)
            }
            "spatial_translation" => {
                // Momentum: sum of state components
                state.sum()
            }
            "rotation" => {
                // Angular momentum: cross-product analog (2D simplification)
                if state.nrows() >= 2 {
                    state[0] * hamiltonian[1.min(hamiltonian.nrows() - 1)]
                        - state[1.min(state.nrows() - 1)] * hamiltonian[0]
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// CALM conservation: the active learning invariant.
    /// Information gain + entropy reduction = constant.
    pub fn calm_invariant(information_gain: f64, entropy_reduction: f64) -> f64 {
        information_gain + entropy_reduction
    }

    /// Compose multiple conservation laws into one invariant.
    pub fn compose(laws: Vec<ConservationLaw>) -> ComposedInvariant {
        if laws.is_empty() {
            return ComposedInvariant {
                laws,
                total_value: 0.0,
                is_conserved: true,
                description: "Empty invariant".into(),
            };
        }

        let total: f64 = laws
            .iter()
            .filter_map(|l| l.conserved_value)
            .sum();

        let all_conserved = laws.iter().all(|l| l.conserved_value.is_some());

        let desc = format!(
            "Composed invariant from {} laws: {}",
            laws.len(),
            laws.iter()
                .map(|l| l.name.as_str())
                .collect::<Vec<_>>()
                .join(" + ")
        );

        ComposedInvariant {
            laws,
            total_value: total,
            is_conserved: all_conserved,
            description: desc,
        }
    }

    /// Check if a state vector satisfies the composed invariant.
    pub fn check_invariant(
        invariant: &ComposedInvariant,
        state: &DVector<f64>,
        tolerance: f64,
    ) -> bool {
        let state_sum: f64 = state.sum();
        (state_sum - invariant.total_value).abs() < tolerance
    }

    /// Build the standard triple: Noether + CALM + Landauer.
    pub fn standard_triple(temperature_kelvin: f64, information_gain: f64) -> ComposedInvariant {
        let landauer = Self::landauer_limit(temperature_kelvin);
        let calm_val = Self::calm_invariant(information_gain, -information_gain);

        Self::compose(vec![
            ConservationLaw {
                name: "Noether_energy".into(),
                source: ConservationSource::Noether {
                    symmetry: "time_translation".into(),
                },
                quantity: "energy".into(),
                description: "Time-translation symmetry → energy conservation".into(),
                conserved_value: Some(1.0),
            },
            ConservationLaw {
                name: "CALM_info_balance".into(),
                source: ConservationSource::CALM,
                quantity: "information_balance".into(),
                description: "Information gain equals entropy reduction".into(),
                conserved_value: Some(calm_val),
            },
            ConservationLaw {
                name: "Landauer_bound".into(),
                source: ConservationSource::Landauer,
                quantity: "minimum_erasure_energy".into(),
                description: format!("Landauer limit at {}K: {:.6e} J", temperature_kelvin, landauer),
                conserved_value: Some(landauer),
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landauer_limit() {
        let e = ConservationBridge::landauer_limit(300.0);
        assert!(e > 0.0);
        assert!(e < 1e-20);
        // At room temp, should be ~2.87e-21 J
        assert!((e - 2.87e-21).abs() / e < 0.01);
    }

    #[test]
    fn test_landauer_limit_zero_k() {
        assert_eq!(ConservationBridge::landauer_limit(0.0), 0.0);
    }

    #[test]
    fn test_noether_energy() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let hamiltonian = DVector::from_vec(vec![0.5, 0.5, 0.5]);
        let energy = ConservationBridge::noether_conserve("time_translation", &state, &hamiltonian);
        assert!((energy - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_momentum() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let hamiltonian = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let momentum = ConservationBridge::noether_conserve("spatial_translation", &state, &hamiltonian);
        assert!((momentum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_rotation() {
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let hamiltonian = DVector::from_vec(vec![0.0, 1.0]);
        let angular = ConservationBridge::noether_conserve("rotation", &state, &hamiltonian);
        // x1*y2 - y1*x2 = 1*1 - 0*0 = 1
        assert!((angular - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calm_invariant() {
        let val = ConservationBridge::calm_invariant(5.0, -5.0);
        assert!((val).abs() < 1e-10);
    }

    #[test]
    fn test_calm_invariant_nonzero() {
        let val = ConservationBridge::calm_invariant(3.0, 2.0);
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose_empty() {
        let inv = ConservationBridge::compose(vec![]);
        assert!(inv.is_conserved);
        assert_eq!(inv.total_value, 0.0);
    }

    #[test]
    fn test_compose_triple() {
        let inv = ConservationBridge::standard_triple(300.0, 1.0);
        assert_eq!(inv.laws.len(), 3);
        assert!(inv.is_conserved);
        assert!(inv.total_value > 0.0);
    }

    #[test]
    fn test_check_invariant_pass() {
        let inv = ComposedInvariant {
            laws: vec![],
            total_value: 6.0,
            is_conserved: true,
            description: "test".into(),
        };
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(ConservationBridge::check_invariant(&inv, &state, 0.1));
    }

    #[test]
    fn test_check_invariant_fail() {
        let inv = ComposedInvariant {
            laws: vec![],
            total_value: 0.0,
            is_conserved: true,
            description: "test".into(),
        };
        let state = DVector::from_vec(vec![100.0]);
        assert!(!ConservationBridge::check_invariant(&inv, &state, 0.1));
    }

    #[test]
    fn test_noether_unknown_symmetry() {
        let state = DVector::from_vec(vec![1.0]);
        let hamiltonian = DVector::from_vec(vec![1.0]);
        assert_eq!(ConservationBridge::noether_conserve("unknown", &state, &hamiltonian), 0.0);
    }
}
