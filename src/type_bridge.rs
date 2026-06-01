//! Type bridge registry — auto-detect shared types across crate boundaries.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Canonical type identifiers for cross-crate bridging.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeKind {
    DMatrixF64,
    VecF64,
    VecUsize,
    ScalarF64,
    Bool,
    String,
    Tuple(Box<TypeKind>, Box<TypeKind>),
    Custom(String),
}

impl std::fmt::Display for TypeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeKind::DMatrixF64 => write!(f, "DMatrix<f64>"),
            TypeKind::VecF64 => write!(f, "Vec<f64>"),
            TypeKind::VecUsize => write!(f, "Vec<usize>"),
            TypeKind::ScalarF64 => write!(f, "f64"),
            TypeKind::Bool => write!(f, "bool"),
            TypeKind::String => write!(f, "String"),
            TypeKind::Tuple(a, b) => write!(f, "({},{})", a, b),
            TypeKind::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// A type exported by a crate, with optional dimensionality info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedType {
    pub type_kind: TypeKind,
    pub dimensions: Option<(usize, usize)>,
    pub description: String,
}

/// A bridge between two types in different crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeBridge {
    pub from_crate: String,
    pub to_crate: String,
    pub from_type: TypeKind,
    pub to_type: TypeKind,
    pub conversion: ConversionKind,
}

/// How values of one type map to another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConversionKind {
    /// Direct identity — same type, no conversion needed.
    Identity,
    /// Reinterpret dimensions (e.g. Vec → column matrix).
    Reshape,
    /// Numeric cast (e.g. f32 → f64).
    Cast,
    /// Custom conversion with a label.
    Custom(String),
}

/// The registry that maps crate → exported types and finds bridges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeBridgeRegistry {
    /// crate name → set of exported types.
    exports: HashMap<String, Vec<ExportedType>>,
    /// Computed bridges.
    bridges: Vec<TypeBridge>,
}

impl Default for TypeBridgeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeBridgeRegistry {
    pub fn new() -> Self {
        TypeBridgeRegistry {
            exports: HashMap::new(),
            bridges: Vec::new(),
        }
    }

    /// Register a type exported by a crate.
    pub fn register(&mut self, crate_name: &str, exported: ExportedType) {
        self.exports
            .entry(crate_name.to_string())
            .or_default()
            .push(exported);
    }

    /// Register multiple types at once.
    pub fn register_many(&mut self, crate_name: &str, types: Vec<ExportedType>) {
        self.exports
            .entry(crate_name.to_string())
            .or_default()
            .extend(types);
    }

    /// Find all types shared between two crates (same TypeKind).
    pub fn shared_types(&self, crate_a: &str, crate_b: &str) -> Vec<TypeKind> {
        let types_a: HashSet<TypeKind> = self
            .exports
            .get(crate_a)
            .map(|v| v.iter().map(|t| t.type_kind.clone()).collect())
            .unwrap_or_default();
        let types_b: HashSet<TypeKind> = self
            .exports
            .get(crate_b)
            .map(|v| v.iter().map(|t| t.type_kind.clone()).collect())
            .unwrap_or_default();
        types_a.intersection(&types_b).cloned().collect()
    }

    /// Auto-detect all bridges: for every pair of crates, find shared or
    /// compatible types and create bridges.
    pub fn auto_detect_bridges(&mut self) -> &[TypeBridge] {
        let crate_names: Vec<String> = self.exports.keys().cloned().collect();
        self.bridges.clear();

        for i in 0..crate_names.len() {
            for j in (i + 1)..crate_names.len() {
                let a = &crate_names[i];
                let b = &crate_names[j];
                for shared in self.shared_types(a, b) {
                    self.bridges.push(TypeBridge {
                        from_crate: a.clone(),
                        to_crate: b.clone(),
                        from_type: shared.clone(),
                        to_type: shared,
                        conversion: ConversionKind::Identity,
                    });
                }
                // Detect reshape-compatible pairs: Vec<f64> ↔ DMatrix<f64>
                let a_has_vec = self.exports.get(a).map_or(false, |v| {
                    v.iter().any(|t| t.type_kind == TypeKind::VecF64)
                });
                let b_has_mat = self.exports.get(b).map_or(false, |v| {
                    v.iter().any(|t| t.type_kind == TypeKind::DMatrixF64)
                });
                let a_has_mat = self.exports.get(a).map_or(false, |v| {
                    v.iter().any(|t| t.type_kind == TypeKind::DMatrixF64)
                });
                let b_has_vec = self.exports.get(b).map_or(false, |v| {
                    v.iter().any(|t| t.type_kind == TypeKind::VecF64)
                });
                if a_has_vec && b_has_mat {
                    self.bridges.push(TypeBridge {
                        from_crate: a.clone(),
                        to_crate: b.clone(),
                        from_type: TypeKind::VecF64,
                        to_type: TypeKind::DMatrixF64,
                        conversion: ConversionKind::Reshape,
                    });
                }
                if a_has_mat && b_has_vec {
                    self.bridges.push(TypeBridge {
                        from_crate: a.clone(),
                        to_crate: b.clone(),
                        from_type: TypeKind::DMatrixF64,
                        to_type: TypeKind::VecF64,
                        conversion: ConversionKind::Reshape,
                    });
                }
            }
        }
        &self.bridges
    }

    /// Get all bridges.
    pub fn bridges(&self) -> &[TypeBridge] {
        &self.bridges
    }

    /// Get all crates known to the registry.
    pub fn crates(&self) -> Vec<&str> {
        self.exports.keys().map(|s| s.as_str()).collect()
    }

    /// Convert a Vec<f64> into a DMatrix column vector.
    pub fn vec_to_column_matrix(v: &Vec<f64>) -> DMatrix<f64> {
        DMatrix::from_column_slice(v.len(), 1, v)
    }

    /// Convert a DMatrix column vector back into a Vec<f64>.
    pub fn column_matrix_to_vec(m: &DMatrix<f64>) -> Vec<f64> {
        m.iter().copied().collect()
    }

    /// Count total registered types across all crates.
    pub fn total_types(&self) -> usize {
        self.exports.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_shared() {
        let mut reg = TypeBridgeRegistry::new();
        reg.register("kalman", ExportedType {
            type_kind: TypeKind::DMatrixF64,
            dimensions: Some((3, 3)),
            description: "State covariance".into(),
        });
        reg.register("kalman", ExportedType {
            type_kind: TypeKind::VecF64,
            dimensions: None,
            description: "State vector".into(),
        });
        reg.register("thermal", ExportedType {
            type_kind: TypeKind::DMatrixF64,
            dimensions: Some((3, 3)),
            description: "Temperature matrix".into(),
        });
        let shared = reg.shared_types("kalman", "thermal");
        assert_eq!(shared, vec![TypeKind::DMatrixF64]);
    }

    #[test]
    fn test_auto_detect_identity_bridges() {
        let mut reg = TypeBridgeRegistry::new();
        reg.register_many("a", vec![ExportedType {
            type_kind: TypeKind::VecF64,
            dimensions: None,
            description: "v".into(),
        }]);
        reg.register_many("b", vec![ExportedType {
            type_kind: TypeKind::VecF64,
            dimensions: None,
            description: "w".into(),
        }]);
        let bridges = reg.auto_detect_bridges();
        assert!(bridges.iter().any(|b| b.conversion == ConversionKind::Identity));
    }

    #[test]
    fn test_auto_detect_reshape_bridge() {
        let mut reg = TypeBridgeRegistry::new();
        reg.register_many("vec_crate", vec![ExportedType {
            type_kind: TypeKind::VecF64,
            dimensions: None,
            description: "data".into(),
        }]);
        reg.register_many("mat_crate", vec![ExportedType {
            type_kind: TypeKind::DMatrixF64,
            dimensions: None,
            description: "matrix".into(),
        }]);
        let bridges = reg.auto_detect_bridges();
        assert!(bridges.iter().any(|b| b.conversion == ConversionKind::Reshape));
    }

    #[test]
    fn test_vec_to_column_roundtrip() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let m = TypeBridgeRegistry::vec_to_column_matrix(&v);
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 1);
        let v2 = TypeBridgeRegistry::column_matrix_to_vec(&m);
        assert_eq!(v, v2);
    }

    #[test]
    fn test_no_shared_types() {
        let mut reg = TypeBridgeRegistry::new();
        reg.register("x", ExportedType {
            type_kind: TypeKind::ScalarF64,
            dimensions: None,
            description: "scalar".into(),
        });
        reg.register("y", ExportedType {
            type_kind: TypeKind::Bool,
            dimensions: None,
            description: "flag".into(),
        });
        assert!(reg.shared_types("x", "y").is_empty());
    }

    #[test]
    fn test_total_types_count() {
        let mut reg = TypeBridgeRegistry::new();
        reg.register_many("c1", vec![
            ExportedType { type_kind: TypeKind::ScalarF64, dimensions: None, description: "a".into() },
            ExportedType { type_kind: TypeKind::VecF64, dimensions: None, description: "b".into() },
        ]);
        reg.register_many("c2", vec![
            ExportedType { type_kind: TypeKind::DMatrixF64, dimensions: None, description: "c".into() },
        ]);
        assert_eq!(reg.total_types(), 3);
    }

    #[test]
    fn test_type_kind_display() {
        assert_eq!(format!("{}", TypeKind::DMatrixF64), "DMatrix<f64>");
        assert_eq!(format!("{}", TypeKind::VecF64), "Vec<f64>");
        assert_eq!(
            format!("{}", TypeKind::Tuple(Box::new(TypeKind::ScalarF64), Box::new(TypeKind::Bool))),
            "(f64,bool)"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let reg = TypeBridgeRegistry::new();
        let json = serde_json::to_string(&reg).unwrap();
        let reg2: TypeBridgeRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(reg2.total_types(), 0);
    }
}
