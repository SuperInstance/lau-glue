# lau-glue

**Gluing 108 mathematical islands into a continent — type bridges, theorem bridges, and the topos structure the mathematics wants to be.**

[![Tests](https://img.shields.io/badge/tests-88-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What This Does

This crate provides the *glue layer* that connects disparate mathematical libraries into a coherent whole. When you have dozens of theorem crates — each correct in isolation, each speaking its own type language — you need bridges: type compatibility, theorem composition, spectral alignment, conservation law unification, cohomology connecting maps, and structural verification that the whole thing forms a topos.

The library provides 10 modules:

- **`type_bridge`** — Auto-detect shared types across crate boundaries and generate conversion bridges
- **`theorem_bridge`** — Given two theorem crates, determine if their results compose (direct feed, weakened, or incompatible)
- **`spectral_bridge`** — Normalize Laplacian spectra across crates for cross-comparison (cosine similarity, earth mover's distance)
- **`conservation_bridge`** — Compose Noether + CALM + Landauer conservation laws into one invariant
- **`agent_bridge`** — Wire Kalman → Thermal → Fokker–Planck → EigenPolicy into a unified agent loop
- **`cohomology_bridge`** — Connect H¹ computations across sheaf models via Mayer–Vietoris sequences
- **`dependency_graph`** — Build and analyze the mathematical (not Cargo) dependency graph using `petgraph`
- **`topos`** — Verify that the glued structure satisfies topos axioms (products, exponentials, subobject classifier)
- **`fiedler`** — Compute the Fiedler vector to find the algebraic center of the theorem graph
- **`growth`** — Detect which crates are special cases of others (linearization, Gaussian assumption, etc.)

## Key Idea

> The mathematics wants to be a topos.

108 crates, each proving theorems in isolation, are like 108 islands. This crate builds the bridges: type compatibility, theorem composition, spectral alignment. The result should satisfy the topos axioms — finite products, exponentials, and a subobject classifier — because that's the structure the mathematics naturally wants when you glue it all together.

## Install

```toml
[dependencies]
lau-glue = "0.1"
```

Or:

```sh
cargo add lau-glue
```

### Dependencies

- `nalgebra` — linear algebra (matrices, vectors)
- `serde` + `serde_json` — serialization
- `petgraph` — graph algorithms (SCC, Dijkstra, connected components)

## Quick Start

```rust
use lau_glue::*;

// --- Type Bridge ---
let mut reg = type_bridge::TypeBridgeRegistry::new();
reg.register("kalman", type_bridge::ExportedType {
    type_kind: type_bridge::TypeKind::DMatrixF64,
    dimensions: Some((3, 3)),
    description: "State covariance".into(),
});
reg.register("thermal", type_bridge::ExportedType {
    type_kind: type_bridge::TypeKind::DMatrixF64,
    dimensions: Some((3, 3)),
    description: "Temperature matrix".into(),
});
let shared = reg.shared_types("kalman", "thermal");
assert_eq!(shared, vec![type_bridge::TypeKind::DMatrixF64]);

// --- Theorem Bridge ---
let mut treg = theorem_bridge::TheoremBridgeRegistry::new();
// Register theorems with inputs/outputs, then:
let bridges = treg.generate_all_bridges();
let valid = treg.valid_composition_count();

// --- Conservation Bridge ---
let invariant = conservation_bridge::ConservationBridge::standard_triple(300.0, 1.0);
assert_eq!(invariant.laws.len(), 3); // Noether + CALM + Landauer

// --- Topos Verification ---
let mut topos = topos::sample_topos();
let result = topos.verify();
assert!(result.is_topos);

// --- Fiedler (graph center) ---
let center = fiedler::find_center(&adjacency_matrix);

// --- Growth Detection ---
let mut det = growth::GrowthDetector::new();
det.register(kalman_profile);
det.register(particle_filter_profile);
let singularities = det.detect(); // "kalman specializes particle-filter via linearization"
```

## API Reference

### `type_bridge`

#### `TypeKind` (enum)

Canonical type identifiers: `DMatrixF64`, `VecF64`, `VecUsize`, `ScalarF64`, `Bool`, `String`, `Tuple(A, B)`, `Custom(String)`.

#### `TypeBridgeRegistry`

| Method | Description |
|--------|-------------|
| `new()` | Empty registry |
| `register(crate, type)` | Register an exported type |
| `register_many(crate, types)` | Batch register |
| `shared_types(a, b)` | Types shared between two crates |
| `auto_detect_bridges()` | Find all identity + reshape bridges |
| `vec_to_column_matrix(v)` | Convert `Vec<f64>` → `DMatrix` column |
| `column_matrix_to_vec(m)` | Convert back |
| `crates()` | All known crate names |
| `total_types()` | Count across all crates |

#### `TypeBridge`

| Field | Description |
|-------|-------------|
| `from_crate` / `to_crate` | Source and target crates |
| `from_type` / `to_type` | Source and target types |
| `conversion` | `Identity`, `Reshape`, `Cast`, or `Custom` |

### `theorem_bridge`

#### `Theorem` and `TheoremId`

A theorem has an ID (crate + name), inputs (`TheoremInput`), outputs (`TheoremOutput`), and a statement.

#### `TheoremBridgeRegistry`

| Method | Description |
|--------|-------------|
| `register(theorem)` | Register a theorem |
| `generate_bridge(from, to)` | Compute the bridge between two theorems |
| `generate_all_bridges()` | All pairwise bridges |
| `valid_composition_count()` | Number of valid (non-incompatible) bridges |
| `theorems_from_crate(name)` | Filter by crate |

#### `BridgeType`

- `DirectFeed` — output of from feeds directly as input to to
- `Weakened { factor }` — needs weakening first
- `SharedStructure { structure }` — common substructure
- `Incompatible` — cannot compose

### `spectral_bridge`

#### `NormalizedSpectrum`

Eigenvalues normalized to [0, 1], with algebraic connectivity and source metadata.

#### `SpectralBridge` (static methods)

| Method | Description |
|--------|-------------|
| `compute_spectrum(laplacian)` | Power iteration with deflation |
| `normalize(eigenvalues)` | Normalize to [0, 1] |
| `spectrum_from_laplacian(L, crate)` | Full `NormalizedSpectrum` from Laplacian |
| `compare(a, b)` | Cosine similarity + earth mover's distance |
| `align_dimensions(a, b)` | Truncate to shared dimension |

#### `SpectralComparison`

| Field | Description |
|-------|-------------|
| `cosine_similarity` | Cosine similarity of eigenvalue vectors |
| `earth_movers_distance` | L1 EMD on sorted distributions |
| `gaps_compatible` | Whether spectral gaps are within tolerance |

### `conservation_bridge`

#### `ConservationBridge` (static methods)

| Method | Description |
|--------|-------------|
| `landauer_limit(T)` | Minimum energy to erase one bit: kT·ln(2) |
| `noether_conserve(sym, state, H)` | Noether conservation for symmetry |
| `calm_invariant(gain, entropy)` | CALM: info gain + entropy reduction = const |
| `compose(laws)` | Compose into `ComposedInvariant` |
| `standard_triple(T, gain)` | Noether + CALM + Landauer at once |
| `check_invariant(inv, state, tol)` | Verify state satisfies invariant |

#### `ConservationSource`

- `Noether { symmetry }` — from Noether's theorem
- `CALM` — Continuous Active Learning Model
- `Landauer` — information-thermodynamic bound

### `agent_bridge`

#### `AgentBridge`

Unified Kalman → Thermal → Fokker–Planck → EigenPolicy pipeline.

| Method | Description |
|--------|-------------|
| `kalman_step(...)` | Full Kalman update (predict + innovation + gain) |
| `thermal_step(state, scale)` | State → temperature, entropy, heat flow |
| `fokker_planck_step(density, drift, diff, dt)` | Finite-difference FP update |
| `eigenpolicy_step(state, reward, lr)` | Softmax policy from eigenstructure |
| `step(...)` | One full cycle through all four stages |

#### Output types: `KalmanOutput`, `ThermalOutput`, `FokkerPlanckOutput`, `EigenPolicyOutput`, `AgentLoopState`

### `cohomology_bridge`

#### `CohomologyBridgeBuilder` (static methods)

| Method | Description |
|--------|-------------|
| `matrix_rank(M, tol)` | Gaussian elimination rank |
| `compute_h1(coboundary)` | H¹ = ker(d₁) / im(d₀) → `CohomologyGroup` |
| `build_bridge(from, to)` | Connecting map between H¹ groups |
| `build_all_bridges(groups)` | All pairwise bridges |
| `mayer_vietoris(h1_a, h1_b, h0_overlap)` | Mayer–Vietoris long exact sequence |
| `euler_characteristic(betti)` | χ = Σ (−1)ⁱbᵢ |

### `dependency_graph`

#### `DependencyGraph`

Mathematical dependency graph using `petgraph::DiGraph`.

| Method | Description |
|--------|-------------|
| `add_crate(node)` | Add a `CrateNode` |
| `add_dependency(from, to, dep)` | Add directed mathematical dependency |
| `find_clusters()` | Kosaraju SCC → mathematical clusters |
| `connected_component_count()` | Connected components |
| `crates_by_domain(domain)` | Filter by mathematical domain |
| `compute_depths()` | Topological depth from roots (via Dijkstra) |
| `critical_path_length()` | Longest dependency chain |
| `build_sample_graph()` | Pre-built 8-node sample graph |

#### `MathDepType`

`UsesDefinitions`, `Generalizes`, `DependsOnTheorems`, `Applies`, `SharedAbstraction`.

### `topos`

#### `ToposCandidate`

| Method | Description |
|--------|-------------|
| `new(name)` | Empty topos candidate |
| `add_sheaf(sheaf)` | Add a sheaf |
| `verify_products()` | Check finite products exist |
| `verify_exponentials()` | Check exponentials exist |
| `verify_subobject_classifier()` | Check Ω exists |
| `verify()` | Full verification → `ToposVerification` |

#### `Sheaf`

Sections (data on open sets) + restriction maps.

#### `ToposVerification`

| Field | Description |
|-------|-------------|
| `axioms` | List of `AxiomResult` |
| `is_topos` | All axioms satisfied |
| `score` | Fraction of satisfied axioms |

### `fiedler`

#### `compute_fiedler(laplacian, iterations)` → `FiedlerResult`

| Field | Description |
|-------|-------------|
| `algebraic_connectivity` | Second-smallest Laplacian eigenvalue |
| `fiedler_vector` | Corresponding eigenvector |
| `center_node` | Node closest to zero in Fiedler vector |
| `partition` | Natural 2-way partition (positive/negative) |

#### `find_center(adjacency)` → `usize`

The "bottleneck" node — algebraically most central.

#### `algebraic_connectivity(adjacency)` → `f64`

### `growth`

#### `GrowthDetector`

| Method | Description |
|--------|-------------|
| `register(profile)` | Register a `CrateProfile` |
| `detect()` | Find all specialization relationships |
| `most_general_crates()` | Ranked by how many crates specialize them |
| `most_specific_crates()` | Ranked by how many they specialize from |

#### `SingularityType`

- `DimensionFixed { fixed_to }` — N-dimensional → fixed N
- `Linearization` — nonlinear → linear
- `GaussianAssumption` — general → Gaussian
- `StationarityAssumption` — time-varying → stationary
- `Discretization` — continuous → discrete
- `Custom(String)`

#### `CrateProfile`

Mathematical properties: dimension, linearity, Gaussian, stationarity, discreteness, assumptions.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      THE GLUE LAYER                          │
│                                                              │
│  ┌─────────────┐  Type Bridge  ┌─────────────┐              │
│  │  Crate A     │───────────────│  Crate B     │              │
│  │  DMatrix<f64>│  (identity)   │  DMatrix<f64>│              │
│  └─────────────┘               └─────────────┘              │
│         │                              │                      │
│         │ Theorem Bridge                │                      │
│         │ (DirectFeed)                  │                      │
│         ▼                              ▼                      │
│  ┌──────────────────────────────────────────┐                │
│  │         Spectral Bridge                   │                │
│  │  Normalize spectra → compare → align     │                │
│  └──────────────────┬───────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────┐                │
│  │       Conservation Bridge                 │                │
│  │  Noether + CALM + Landauer = constant     │                │
│  └──────────────────┬───────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────┐                │
│  │       Cohomology Bridge                   │                │
│  │  H¹(A) ──connecting──→ H¹(B)             │                │
│  │  Mayer-Vietoris: H⁰(∩) → H¹(A∪B)        │                │
│  └──────────────────┬───────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────┐                │
│  │       TOPOS VERIFICATION                  │                │
│  │  ✓ Products    ✓ Exponentials   ✓ Ω      │                │
│  └──────────────────────────────────────────┘                │
│                                                              │
│  Fiedler vector finds the center of the theorem graph        │
│  Growth detector finds specialization relationships          │
└─────────────────────────────────────────────────────────────┘
```

The glue layer works bottom-up:

1. **Type bridges** ensure crates can talk to each other (shared `DMatrix<f64>`, `Vec<f64>` ↔ column matrix reshapes).
2. **Theorem bridges** determine if theorem outputs can feed into other theorem inputs (direct feed, weakened, or incompatible).
3. **Spectral bridges** normalize and compare Laplacian spectra across crates using cosine similarity and earth mover's distance.
4. **Conservation bridges** compose the three conservation frameworks (Noether, CALM, Landauer) into a single invariant.
5. **Agent bridges** wire the four main theorem types (Kalman, Thermal, Fokker–Planck, EigenPolicy) into a unified loop.
6. **Cohomology bridges** connect H¹ groups via connecting maps and Mayer–Vietoris sequences.
7. **Dependency graphs** model the mathematical (not Cargo) relationships between crates.
8. **Topos verification** checks that the whole glued structure satisfies the topos axioms.
9. **Fiedler analysis** finds the algebraic center of the theorem graph.
10. **Growth detection** identifies specialization relationships (which crates are special cases of others).

## The Math

### Type Bridge

Given two crates A and B with exported type sets T_A and T_B:
- Shared types: T_A ∩ T_B (identity bridges)
- Compatible types: e.g., Vec<f64> ↔ DMatrix<f64> (reshape bridges)

### Theorem Bridge

For theorems T₁ (outputs O₁) and T₂ (inputs I₂):
- DirectFeed if ∃ o ∈ O₁, i ∈ I₂: match(o, i)
- Weakened if approximate match with factor
- Incompatible otherwise

### Spectral Bridge

Given Laplacians L_A (from crate A) and L_B (from crate B):
- Compute eigenvalues: σ(L_A) = {λ₁, …, λₙ}
- Normalize to [0, 1]: λ̃ᵢ = (λᵢ − min) / (max − min)
- Cosine similarity: cos(θ) = ⟨σ̃_A, σ̃_B⟩ / (‖σ̃_A‖ · ‖σ̃_B‖)
- Earth mover's distance: cumulative L1 on sorted distributions

### Conservation Bridge

Three conservation laws unified:
- **Noether**: symmetry ⟹ conservation (time → energy, translation → momentum, rotation → angular momentum)
- **CALM**: I_gain + ΔS = const (information gain equals entropy reduction)
- **Landauer**: E_min = kT·ln(2) per bit erased

Composed invariant: Noether + CALM + Landauer ≈ constant.

### Agent Bridge

Four-stage pipeline per step:
1. **Kalman**: x̂₊ = x̂₋ + K(z − Hx̂₋), where K = P₋Hᵀ(HP₋Hᵀ + R)⁻¹
2. **Thermal**: T = |state| · scale, S = −Σ T·ln(T)
3. **Fokker–Planck**: ∂ρ/∂t = −∇·(v·ρ) + D·∇²ρ
4. **EigenPolicy**: π = softmax(R · state), quality = ‖R · state‖

### Cohomology Bridge

H¹ = ker(d₁) / im(d₀) for each crate's coboundary complex.

Mayer–Vietoris: … → H⁰(A)⊕H⁰(B) → H⁰(A∩B) → H¹(A∪B) → H¹(A)⊕H¹(B) → …

Euler characteristic: χ = Σ (−1)ⁱbᵢ

### Fiedler Vector

For graph Laplacian L with eigenvalues 0 = λ₁ < λ₂ ≤ … ≤ λₙ:
- Algebraic connectivity: λ₂
- Fiedler vector: eigenvector of λ₂
- Center node: argmin |v_Fiedler(i)|
- Natural partition: {i : v_Fiedler(i) ≥ 0} vs {i : v_Fiedler(i) < 0}

### Topos Axioms

A topos is a category with:
1. **Finite products**: ∀ A, B, the product A × B exists
2. **Exponentials**: ∀ A, B, the internal hom B^A exists
3. **Subobject classifier**: An object Ω such that subobjects ↔ morphisms to Ω

In the sheaf topos, Ω is the presheaf of sieves.

## License

MIT
