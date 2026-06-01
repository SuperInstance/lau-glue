//! Dependency graph builder — given 108 crates, build the actual mathematical
//! dependency graph (not Cargo.toml deps).

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{connected_components, dijkstra, kosaraju_scc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the mathematical dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateNode {
    pub name: String,
    pub mathematical_domain: String,
    pub key_concepts: Vec<String>,
}

/// An edge representing a mathematical dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathDep {
    pub dep_type: MathDepType,
    pub description: String,
}

/// The type of mathematical dependency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathDepType {
    /// A uses definitions from B.
    UsesDefinitions,
    /// A extends/generalizes B.
    Generalizes,
    /// A depends on B's theorems.
    DependsOnTheorems,
    /// A applies B's machinery.
    Applies,
    /// A and B share a common abstraction.
    SharedAbstraction,
}

/// The mathematical dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    #[serde(skip)]
    pub graph: DiGraph<CrateNode, MathDep>,
    /// Serialize-friendly node list.
    pub nodes: Vec<CrateNode>,
    /// Serialize-friendly edge list: (from_idx, to_idx, MathDep).
    pub edges: Vec<(usize, usize, MathDep)>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        DependencyGraph {
            graph: DiGraph::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a crate node.
    pub fn add_crate(&mut self, node: CrateNode) -> usize {
        let idx = self.graph.add_node(node.clone());
        self.nodes.push(node);
        idx.index()
    }

    /// Add a mathematical dependency edge.
    pub fn add_dependency(&mut self, from: usize, to: usize, dep: MathDep) {
        let from_idx = NodeIndex::new(from);
        let to_idx = NodeIndex::new(to);
        self.graph.add_edge(from_idx, to_idx, dep.clone());
        self.edges.push((from, to, dep));
    }

    /// Number of crates.
    pub fn crate_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of mathematical dependencies.
    pub fn dep_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Find strongly connected components (mathematical clusters).
    pub fn find_clusters(&self) -> Vec<Vec<usize>> {
        let sccs = kosaraju_scc(&self.graph);
        sccs.into_iter()
            .map(|scc| scc.into_iter().map(|idx| idx.index()).collect())
            .collect()
    }

    /// Number of connected components in the undirected projection.
    pub fn connected_component_count(&self) -> usize {
        connected_components(&self.graph)
    }

    /// Get crates by domain.
    pub fn crates_by_domain(&self, domain: &str) -> Vec<&CrateNode> {
        self.nodes.iter().filter(|n| n.mathematical_domain == domain).collect()
    }

    /// Compute topological depth of each node (longest path from a root).
    pub fn compute_depths(&self) -> HashMap<usize, usize> {
        let n = self.graph.node_count();
        let mut depths = HashMap::new();

        // Find roots (nodes with no incoming edges)
        let mut has_incoming = vec![false; n];
        for edge in self.graph.raw_edges() {
            has_incoming[edge.target().index()] = true;
        }

        for root in 0..n {
            if !has_incoming[root] {
                let dists = dijkstra(&self.graph, NodeIndex::new(root), None, |_| 1.0);
                for (node, dist) in dists {
                    let d = dist as usize;
                    let current = depths.get(&node.index()).copied().unwrap_or(0);
                    if d > current {
                        depths.insert(node.index(), d);
                    }
                }
            }
        }

        // Nodes with no path from any root get depth 0
        for i in 0..n {
            depths.entry(i).or_insert(0);
        }

        depths
    }

    /// Find the longest path in the DAG (critical mathematical path).
    pub fn critical_path_length(&self) -> usize {
        let depths = self.compute_depths();
        depths.values().copied().max().unwrap_or(0)
    }

    /// Reconstruct the graph from serialized data.
    pub fn reconstruct(&mut self) {
        self.graph = DiGraph::new();
        for node in &self.nodes {
            self.graph.add_node(node.clone());
        }
        for (from, to, dep) in &self.edges {
            self.graph.add_edge(NodeIndex::new(*from), NodeIndex::new(*to), dep.clone());
        }
    }
}

/// Build a sample dependency graph for the 108 crates.
pub fn build_sample_graph() -> DependencyGraph {
    let mut g = DependencyGraph::new();

    // Core mathematical layers
    let linear_algebra = g.add_crate(CrateNode {
        name: "linear-algebra".into(),
        mathematical_domain: "algebra".into(),
        key_concepts: vec!["matrices".into(), "eigenvalues".into(), "SVD".into()],
    });
    let probability = g.add_crate(CrateNode {
        name: "probability".into(),
        mathematical_domain: "probability".into(),
        key_concepts: vec!["distributions".into(), "Bayes".into()],
    });
    let optimization = g.add_crate(CrateNode {
        name: "optimization".into(),
        mathematical_domain: "optimization".into(),
        key_concepts: vec!["gradient descent".into(), "convexity".into()],
    });
    let graph_theory = g.add_crate(CrateNode {
        name: "graph-theory".into(),
        mathematical_domain: "combinatorics".into(),
        key_concepts: vec!["graphs".into(), "Laplacian".into(), "spectral".into()],
    });
    let kalman = g.add_crate(CrateNode {
        name: "kalman".into(),
        mathematical_domain: "estimation".into(),
        key_concepts: vec!["Kalman filter".into(), "state estimation".into()],
    });
    let thermal = g.add_crate(CrateNode {
        name: "thermal".into(),
        mathematical_domain: "thermodynamics".into(),
        key_concepts: vec!["temperature".into(), "entropy".into(), "heat".into()],
    });
    let fokker_planck = g.add_crate(CrateNode {
        name: "fokker-planck".into(),
        mathematical_domain: "stochastic".into(),
        key_concepts: vec!["SDE".into(), "drift".into(), "diffusion".into()],
    });
    let eigen_policy = g.add_crate(CrateNode {
        name: "eigen-policy".into(),
        mathematical_domain: "control".into(),
        key_concepts: vec!["eigenvalue".into(), "policy".into(), "RL".into()],
    });

    // Dependencies
    g.add_dependency(kalman, linear_algebra, MathDep {
        dep_type: MathDepType::UsesDefinitions,
        description: "Kalman filter needs matrix algebra".into(),
    });
    g.add_dependency(kalman, probability, MathDep {
        dep_type: MathDepType::DependsOnTheorems,
        description: "Kalman filter uses Bayesian inference".into(),
    });
    g.add_dependency(thermal, graph_theory, MathDep {
        dep_type: MathDepType::Applies,
        description: "Thermal models on graphs use Laplacians".into(),
    });
    g.add_dependency(fokker_planck, probability, MathDep {
        dep_type: MathDepType::DependsOnTheorems,
        description: "Fokker-Planck derives from stochastic calculus".into(),
    });
    g.add_dependency(fokker_planck, thermal, MathDep {
        dep_type: MathDepType::SharedAbstraction,
        description: "Fokker-Planck and thermal share diffusion concepts".into(),
    });
    g.add_dependency(eigen_policy, linear_algebra, MathDep {
        dep_type: MathDepType::UsesDefinitions,
        description: "EigenPolicy uses eigendecomposition".into(),
    });
    g.add_dependency(eigen_policy, optimization, MathDep {
        dep_type: MathDepType::DependsOnTheorems,
        description: "Policy optimization uses convex analysis".into(),
    });
    g.add_dependency(graph_theory, linear_algebra, MathDep {
        dep_type: MathDepType::UsesDefinitions,
        description: "Spectral graph theory needs linear algebra".into(),
    });

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sample_graph() {
        let g = build_sample_graph();
        assert_eq!(g.crate_count(), 8);
        assert_eq!(g.dep_count(), 8);
    }

    #[test]
    fn test_connected_components() {
        let g = build_sample_graph();
        assert!(g.connected_component_count() >= 1);
    }

    #[test]
    fn test_find_clusters() {
        let g = build_sample_graph();
        let clusters = g.find_clusters();
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_crates_by_domain() {
        let g = build_sample_graph();
        let algebra_crates = g.crates_by_domain("algebra");
        assert_eq!(algebra_crates.len(), 1);
    }

    #[test]
    fn test_compute_depths() {
        let g = build_sample_graph();
        let depths = g.compute_depths();
        assert_eq!(depths.len(), 8);
        // All nodes should have a depth value
        assert!(depths.values().all(|&d| d < 100));
    }

    #[test]
    fn test_critical_path() {
        let g = build_sample_graph();
        let path_len = g.critical_path_length();
        assert!(path_len > 0);
    }

    #[test]
    fn test_add_and_query() {
        let mut g = DependencyGraph::new();
        let a = g.add_crate(CrateNode {
            name: "a".into(),
            mathematical_domain: "test".into(),
            key_concepts: vec![],
        });
        let b = g.add_crate(CrateNode {
            name: "b".into(),
            mathematical_domain: "test".into(),
            key_concepts: vec![],
        });
        g.add_dependency(a, b, MathDep {
            dep_type: MathDepType::UsesDefinitions,
            description: "a uses b".into(),
        });
        assert_eq!(g.crate_count(), 2);
        assert_eq!(g.dep_count(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let g = build_sample_graph();
        let json = serde_json::to_string(&g).unwrap();
        let mut g2: DependencyGraph = serde_json::from_str(&json).unwrap();
        g2.reconstruct();
        assert_eq!(g2.crate_count(), g.crate_count());
        assert_eq!(g2.dep_count(), g.dep_count());
    }
}
