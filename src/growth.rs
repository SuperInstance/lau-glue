//! Growth singularity detector — identify crates that are special cases of
//! more general crates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A crate with its mathematical properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateProfile {
    pub name: String,
    pub generalizes_to: Vec<String>,
    pub specializes_from: Vec<String>,
    pub dimension_parameter: Option<usize>,
    pub is_linear: bool,
    pub is_gaussian: bool,
    pub is_stationary: bool,
    pub is_discrete: bool,
    pub assumptions: Vec<String>,
}

/// A detected singularity: one crate is a special case of another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Singularity {
    pub specific_crate: String,
    pub general_crate: String,
    pub singularity_type: SingularityType,
    pub description: String,
    pub confidence: f64,
}

/// How one crate specializes another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SingularityType {
    /// Fixed dimension: general is N-dimensional, specific fixes N.
    DimensionFixed { fixed_to: usize },
    /// Linearity assumption: general is nonlinear, specific is linear.
    Linearization,
    /// Gaussian assumption: general allows any distribution, specific assumes Gaussian.
    GaussianAssumption,
    /// Stationarity: general allows time-varying, specific assumes stationarity.
    StationarityAssumption,
    /// Discretization: general is continuous, specific is discrete.
    Discretization,
    /// Custom specialization.
    Custom(String),
}

/// The growth singularity detector.
pub struct GrowthDetector {
    profiles: HashMap<String, CrateProfile>,
    singularities: Vec<Singularity>,
}

impl Default for GrowthDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl GrowthDetector {
    pub fn new() -> Self {
        GrowthDetector {
            profiles: HashMap::new(),
            singularities: Vec::new(),
        }
    }

    /// Register a crate profile.
    pub fn register(&mut self, profile: CrateProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Detect all singularities: for each pair, check if one specializes the other.
    pub fn detect(&mut self) -> &[Singularity] {
        self.singularities.clear();
        let names: Vec<String> = self.profiles.keys().cloned().collect();

        for i in 0..names.len() {
            for j in 0..names.len() {
                if i != j {
                    let sigs = self.check_specializations(&names[i], &names[j]);
                    self.singularities.extend(sigs);
                }
            }
        }

        &self.singularities
    }

    /// Check all ways in which `specific` is a special case of `general`.
    fn check_specializations(&self, specific: &str, general: &str) -> Vec<Singularity> {
        let sp = match self.profiles.get(specific) {
            Some(p) => p,
            None => return vec![],
        };
        let gen = match self.profiles.get(general) {
            Some(p) => p,
            None => return vec![],
        };

        let mut result = Vec::new();

        // Dimension specialization
        if let (Some(sp_dim), Some(gen_dim)) = (sp.dimension_parameter, gen.dimension_parameter) {
            if gen_dim > sp_dim {
                result.push(Singularity {
                    specific_crate: specific.to_string(),
                    general_crate: general.to_string(),
                    singularity_type: SingularityType::DimensionFixed { fixed_to: sp_dim },
                    description: format!(
                        "{} (dim={}) specializes {} (dim={}) by fixing dimension",
                        specific, sp_dim, general, gen_dim
                    ),
                    confidence: 0.9,
                });
            }
        }

        // Linearity specialization
        if sp.is_linear && !gen.is_linear {
            result.push(Singularity {
                specific_crate: specific.to_string(),
                general_crate: general.to_string(),
                singularity_type: SingularityType::Linearization,
                description: format!(
                    "{} is the linear special case of {}",
                    specific, general
                ),
                confidence: 0.85,
            });
        }

        // Gaussian specialization
        if sp.is_gaussian && !gen.is_gaussian {
            result.push(Singularity {
                specific_crate: specific.to_string(),
                general_crate: general.to_string(),
                singularity_type: SingularityType::GaussianAssumption,
                description: format!(
                    "{} assumes Gaussian noise, while {} allows general distributions",
                    specific, general
                ),
                confidence: 0.8,
            });
        }

        // Stationarity
        if sp.is_stationary && !gen.is_stationary {
            result.push(Singularity {
                specific_crate: specific.to_string(),
                general_crate: general.to_string(),
                singularity_type: SingularityType::StationarityAssumption,
                description: format!(
                    "{} assumes stationarity, while {} is time-varying",
                    specific, general
                ),
                confidence: 0.75,
            });
        }

        // Discretization
        if sp.is_discrete && !gen.is_discrete {
            result.push(Singularity {
                specific_crate: specific.to_string(),
                general_crate: general.to_string(),
                singularity_type: SingularityType::Discretization,
                description: format!(
                    "{} is the discrete version of {}",
                    specific, general
                ),
                confidence: 0.8,
            });
        }

        result
    }

    /// Get the "most general" crates (those that are the general case of
    /// the most other crates).
    pub fn most_general_crates(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for s in &self.singularities {
            *counts.entry(s.general_crate.clone()).or_insert(0) += 1;
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Get the "most specific" crates (those that specialize the most).
    pub fn most_specific_crates(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for s in &self.singularities {
            *counts.entry(s.specific_crate.clone()).or_insert(0) += 1;
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Get singularities.
    pub fn singularities(&self) -> &[Singularity] {
        &self.singularities
    }

    /// Count registered profiles.
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kalman_profile() -> CrateProfile {
        CrateProfile {
            name: "kalman".into(),
            generalizes_to: vec!["particle-filter".into()],
            specializes_from: vec![],
            dimension_parameter: Some(2),
            is_linear: true,
            is_gaussian: true,
            is_stationary: true,
            is_discrete: true,
            assumptions: vec!["linear dynamics".into(), "Gaussian noise".into()],
        }
    }

    fn particle_filter_profile() -> CrateProfile {
        CrateProfile {
            name: "particle-filter".into(),
            generalizes_to: vec![],
            specializes_from: vec!["kalman".into()],
            dimension_parameter: Some(10),
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: true,
            assumptions: vec![],
        }
    }

    fn continuous_filter_profile() -> CrateProfile {
        CrateProfile {
            name: "continuous-filter".into(),
            generalizes_to: vec![],
            specializes_from: vec![],
            dimension_parameter: Some(10),
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: false,
            assumptions: vec![],
        }
    }

    #[test]
    fn test_detect_linearization() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        let sings = det.detect();
        assert!(sings.iter().any(|s| matches!(s.singularity_type, SingularityType::Linearization)));
    }

    #[test]
    fn test_detect_gaussian() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        let sings = det.detect();
        assert!(sings.iter().any(|s| matches!(s.singularity_type, SingularityType::GaussianAssumption)));
    }

    #[test]
    fn test_detect_stationarity() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        let sings = det.detect();
        assert!(sings.iter().any(|s| matches!(s.singularity_type, SingularityType::StationarityAssumption)));
    }

    #[test]
    fn test_detect_dimension() {
        let mut det = GrowthDetector::new();
        det.register(CrateProfile {
            name: "low-dim".into(),
            generalizes_to: vec![],
            specializes_from: vec![],
            dimension_parameter: Some(2),
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: false,
            assumptions: vec![],
        });
        det.register(CrateProfile {
            name: "high-dim".into(),
            generalizes_to: vec![],
            specializes_from: vec![],
            dimension_parameter: Some(10),
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: false,
            assumptions: vec![],
        });
        let sings = det.detect();
        assert!(sings.iter().any(|s| matches!(s.singularity_type, SingularityType::DimensionFixed { .. })));
    }

    #[test]
    fn test_detect_discretization() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile()); // discrete
        det.register(continuous_filter_profile()); // continuous
        let sings = det.detect();
        assert!(sings.iter().any(|s| matches!(s.singularity_type, SingularityType::Discretization)));
    }

    #[test]
    fn test_most_general() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        det.detect();
        let general = det.most_general_crates();
        assert!(!general.is_empty());
        assert_eq!(general[0].0, "particle-filter");
    }

    #[test]
    fn test_most_specific() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        det.detect();
        let specific = det.most_specific_crates();
        assert!(!specific.is_empty());
    }

    #[test]
    fn test_no_singularities_identical() {
        let mut det = GrowthDetector::new();
        det.register(CrateProfile {
            name: "a".into(),
            generalizes_to: vec![],
            specializes_from: vec![],
            dimension_parameter: None,
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: false,
            assumptions: vec![],
        });
        det.register(CrateProfile {
            name: "b".into(),
            generalizes_to: vec![],
            specializes_from: vec![],
            dimension_parameter: None,
            is_linear: false,
            is_gaussian: false,
            is_stationary: false,
            is_discrete: false,
            assumptions: vec![],
        });
        let sings = det.detect();
        assert!(sings.is_empty());
    }

    #[test]
    fn test_profile_count() {
        let mut det = GrowthDetector::new();
        det.register(kalman_profile());
        det.register(particle_filter_profile());
        assert_eq!(det.profile_count(), 2);
    }
}
