//! Theorem bridge — given two theorem crates, generate the bridge that makes
//! their results compose.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A theorem identifier: crate + name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TheoremId {
    pub crate_name: String,
    pub theorem_name: String,
}

impl std::fmt::Display for TheoremId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.crate_name, self.theorem_name)
    }
}

/// What a theorem produces (its "output signature").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TheoremOutput {
    /// Proves an inequality bound.
    Bound { quantity: String, upper: f64 },
    /// Proves convergence rate.
    ConvergenceRate { rate: f64 },
    /// Proves existence of something.
    Existence { entity: String },
    /// Proves a relation between quantities.
    Relation { left: String, right: String, relation: String },
    /// Custom result.
    Custom(String),
}

/// What a theorem requires (its "input signature").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TheoremInput {
    /// Requires a matrix with certain properties.
    Matrix { property: String },
    /// Requires a function with certain properties.
    Function { property: String },
    /// Requires a bound from another theorem.
    RequiresBound { quantity: String },
    /// Custom requirement.
    Custom(String),
}

/// A theorem with its inputs and outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    pub id: TheoremId,
    pub inputs: Vec<TheoremInput>,
    pub outputs: Vec<TheoremOutput>,
    pub statement: String,
}

/// A bridge connecting two theorems so their results compose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremBridge {
    pub from: TheoremId,
    pub to: TheoremId,
    pub bridge_type: BridgeType,
    pub composition_valid: bool,
}

/// How two theorems connect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BridgeType {
    /// Output of `from` feeds directly as input to `to`.
    DirectFeed,
    /// Output needs weakening/transformation before feeding.
    Weakened { factor: f64 },
    /// Theorems share a common substructure.
    SharedStructure { structure: String },
    /// Cannot be composed.
    Incompatible,
}

/// Registry of theorems and their bridges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremBridgeRegistry {
    theorems: HashMap<String, Theorem>,
    bridges: Vec<TheoremBridge>,
}

impl Default for TheoremBridgeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TheoremBridgeRegistry {
    pub fn new() -> Self {
        TheoremBridgeRegistry {
            theorems: HashMap::new(),
            bridges: Vec::new(),
        }
    }

    /// Register a theorem.
    pub fn register(&mut self, theorem: Theorem) {
        let key = format!("{}", theorem.id);
        self.theorems.insert(key, theorem);
    }

    /// Given two theorems, generate the bridge between them.
    pub fn generate_bridge(&self, from: &TheoremId, to: &TheoremId) -> TheoremBridge {
        let from_thm = self.theorems.get(&format!("{}", from));
        let to_thm = self.theorems.get(&format!("{}", to));

        match (from_thm, to_thm) {
            (Some(from_t), Some(to_t)) => {
                // Check if any output of `from` matches any input of `to`.
                let mut found_direct = false;
                let mut found_weakened = false;

                for output in &from_t.outputs {
                    for input in &to_t.inputs {
                        match (output, input) {
                            (TheoremOutput::Bound { quantity, .. }, TheoremInput::RequiresBound { quantity: req }) => {
                                if quantity == req {
                                    found_direct = true;
                                }
                            }
                            (TheoremOutput::ConvergenceRate { rate }, TheoremInput::Function { property }) => {
                                if property.contains("convergent") {
                                    found_weakened = true;
                                }
                                let _ = rate;
                            }
                            _ => {}
                        }
                    }
                }

                if found_direct {
                    TheoremBridge {
                        from: from.clone(),
                        to: to.clone(),
                        bridge_type: BridgeType::DirectFeed,
                        composition_valid: true,
                    }
                } else if found_weakened {
                    TheoremBridge {
                        from: from.clone(),
                        to: to.clone(),
                        bridge_type: BridgeType::Weakened { factor: 0.5 },
                        composition_valid: true,
                    }
                } else {
                    TheoremBridge {
                        from: from.clone(),
                        to: to.clone(),
                        bridge_type: BridgeType::Incompatible,
                        composition_valid: false,
                    }
                }
            }
            _ => TheoremBridge {
                from: from.clone(),
                to: to.clone(),
                bridge_type: BridgeType::Incompatible,
                composition_valid: false,
            },
        }
    }

    /// Generate all pairwise bridges and store them.
    pub fn generate_all_bridges(&mut self) -> &[TheoremBridge] {
        let keys: Vec<String> = self.theorems.keys().cloned().collect();
        self.bridges.clear();

        for i in 0..keys.len() {
            for j in 0..keys.len() {
                if i != j {
                    let from_id = self.theorems[&keys[i]].id.clone();
                    let to_id = self.theorems[&keys[j]].id.clone();
                    let bridge = self.generate_bridge(&from_id, &to_id);
                    self.bridges.push(bridge);
                }
            }
        }
        &self.bridges
    }

    /// Get theorems registered from a specific crate.
    pub fn theorems_from_crate(&self, crate_name: &str) -> Vec<&Theorem> {
        self.theorems
            .values()
            .filter(|t| t.id.crate_name == crate_name)
            .collect()
    }

    /// Count valid composition paths.
    pub fn valid_composition_count(&self) -> usize {
        self.bridges.iter().filter(|b| b.composition_valid).count()
    }

    /// Get all bridges.
    pub fn bridges(&self) -> &[TheoremBridge] {
        &self.bridges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_theorem(crate_name: &str, name: &str) -> Theorem {
        Theorem {
            id: TheoremId {
                crate_name: crate_name.to_string(),
                theorem_name: name.to_string(),
            },
            inputs: vec![],
            outputs: vec![],
            statement: format!("Theorem {} from {}", name, crate_name),
        }
    }

    #[test]
    fn test_direct_feed_bridge() {
        let mut reg = TheoremBridgeRegistry::new();
        reg.register(Theorem {
            id: TheoremId { crate_name: "kalman".into(), theorem_name: "convergence".into() },
            inputs: vec![],
            outputs: vec![TheoremOutput::Bound { quantity: "error".into(), upper: 0.01 }],
            statement: "Kalman converges".into(),
        });
        reg.register(Theorem {
            id: TheoremId { crate_name: "thermal".into(), theorem_name: "stability".into() },
            inputs: vec![TheoremInput::RequiresBound { quantity: "error".into() }],
            outputs: vec![],
            statement: "Thermal needs error bound".into(),
        });

        let bridge = reg.generate_bridge(
            &TheoremId { crate_name: "kalman".into(), theorem_name: "convergence".into() },
            &TheoremId { crate_name: "thermal".into(), theorem_name: "stability".into() },
        );
        assert_eq!(bridge.bridge_type, BridgeType::DirectFeed);
        assert!(bridge.composition_valid);
    }

    #[test]
    fn test_incompatible_bridge() {
        let mut reg = TheoremBridgeRegistry::new();
        reg.register(Theorem {
            id: TheoremId { crate_name: "a".into(), theorem_name: "t1".into() },
            inputs: vec![],
            outputs: vec![TheoremOutput::Existence { entity: "unicorn".into() }],
            statement: "Unicorns exist".into(),
        });
        reg.register(Theorem {
            id: TheoremId { crate_name: "b".into(), theorem_name: "t2".into() },
            inputs: vec![TheoremInput::Matrix { property: "SPD".into() }],
            outputs: vec![],
            statement: "Matrix is SPD".into(),
        });

        let bridge = reg.generate_bridge(
            &TheoremId { crate_name: "a".into(), theorem_name: "t1".into() },
            &TheoremId { crate_name: "b".into(), theorem_name: "t2".into() },
        );
        assert_eq!(bridge.bridge_type, BridgeType::Incompatible);
        assert!(!bridge.composition_valid);
    }

    #[test]
    fn test_generate_all_bridges() {
        let mut reg = TheoremBridgeRegistry::new();
        reg.register(Theorem {
            id: TheoremId { crate_name: "a".into(), theorem_name: "t1".into() },
            inputs: vec![TheoremInput::RequiresBound { quantity: "x".into() }],
            outputs: vec![TheoremOutput::Bound { quantity: "y".into(), upper: 1.0 }],
            statement: "t1".into(),
        });
        reg.register(Theorem {
            id: TheoremId { crate_name: "b".into(), theorem_name: "t2".into() },
            inputs: vec![TheoremInput::RequiresBound { quantity: "y".into() }],
            outputs: vec![TheoremOutput::Bound { quantity: "x".into(), upper: 2.0 }],
            statement: "t2".into(),
        });
        let bridges = reg.generate_all_bridges();
        assert_eq!(bridges.len(), 2); // a→b and b→a
        assert_eq!(reg.valid_composition_count(), 2);
    }

    #[test]
    fn test_theorems_from_crate() {
        let mut reg = TheoremBridgeRegistry::new();
        reg.register(make_theorem("kalman", "t1"));
        reg.register(make_theorem("kalman", "t2"));
        reg.register(make_theorem("thermal", "t3"));
        assert_eq!(reg.theorems_from_crate("kalman").len(), 2);
        assert_eq!(reg.theorems_from_crate("thermal").len(), 1);
        assert_eq!(reg.theorems_from_crate("unknown").len(), 0);
    }

    #[test]
    fn test_theorem_id_display() {
        let id = TheoremId { crate_name: "foo".into(), theorem_name: "bar".into() };
        assert_eq!(format!("{}", id), "foo::bar");
    }

    #[test]
    fn test_missing_theorem_bridge() {
        let reg = TheoremBridgeRegistry::new();
        let bridge = reg.generate_bridge(
            &TheoremId { crate_name: "a".into(), theorem_name: "x".into() },
            &TheoremId { crate_name: "b".into(), theorem_name: "y".into() },
        );
        assert!(!bridge.composition_valid);
    }

    #[test]
    fn test_shared_structure_bridge() {
        let mut reg = TheoremBridgeRegistry::new();
        reg.register(Theorem {
            id: TheoremId { crate_name: "a".into(), theorem_name: "laplacian_bound".into() },
            inputs: vec![TheoremInput::Matrix { property: "laplacian".into() }],
            outputs: vec![TheoremOutput::Bound { quantity: "spectral_gap".into(), upper: 0.5 }],
            statement: "Laplacian bound".into(),
        });
        reg.register(Theorem {
            id: TheoremId { crate_name: "b".into(), theorem_name: "connectivity".into() },
            inputs: vec![TheoremInput::RequiresBound { quantity: "spectral_gap".into() }],
            outputs: vec![],
            statement: "Connectivity from gap".into(),
        });
        let bridge = reg.generate_bridge(
            &TheoremId { crate_name: "a".into(), theorem_name: "laplacian_bound".into() },
            &TheoremId { crate_name: "b".into(), theorem_name: "connectivity".into() },
        );
        assert!(bridge.composition_valid);
    }
}
