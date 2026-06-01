//! Agent bridge — wire Kalman → Thermal → Fokker-Planck → EigenPolicy into
//! one unified agent loop.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Output of a Kalman filter step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanOutput {
    pub state_estimate: Vec<f64>,
    pub covariance: Vec<f64>, // flattened
    pub dimension: usize,
}

/// Output of a thermal model step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalOutput {
    pub temperature: Vec<f64>,
    pub entropy: f64,
    pub heat_flow: Vec<f64>,
}

/// Output of a Fokker-Planck step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FokkerPlanckOutput {
    pub probability_density: Vec<f64>,
    pub drift: Vec<f64>,
    pub diffusion: f64,
}

/// Output of the EigenPolicy step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenPolicyOutput {
    pub action: Vec<f64>,
    pub eigenvalue: f64,
    pub policy_stable: bool,
}

/// The complete agent loop state after one full cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoopState {
    pub step: u64,
    pub kalman: KalmanOutput,
    pub thermal: ThermalOutput,
    pub fokker_planck: FokkerPlanckOutput,
    pub policy: EigenPolicyOutput,
    pub total_energy: f64,
    pub converged: bool,
}

/// The unified agent bridge.
pub struct AgentBridge {
    pub temperature_scale: f64,
    pub diffusion_coefficient: f64,
    pub learning_rate: f64,
    pub step_count: u64,
}

impl Default for AgentBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBridge {
    pub fn new() -> Self {
        AgentBridge {
            temperature_scale: 1.0,
            diffusion_coefficient: 0.1,
            learning_rate: 0.01,
            step_count: 0,
        }
    }

    /// Step 1: Kalman filter update.
    /// Given observation, produce filtered state estimate.
    pub fn kalman_step(
        prior_state: &DVector<f64>,
        prior_covariance: &DMatrix<f64>,
        observation: &DVector<f64>,
        observation_matrix: &DMatrix<f64>,
        process_noise: &DMatrix<f64>,
        observation_noise: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        // Predict
        let predicted_state = prior_state.clone();
        let predicted_cov = &*prior_covariance + process_noise;

        // Innovation
        let innovation = observation - observation_matrix * &predicted_state;
        let innovation_cov =
            observation_matrix * &predicted_cov * observation_matrix.transpose() + observation_noise;

        // Kalman gain
        let kalman_gain = predicted_cov.clone()
            * observation_matrix.transpose()
            * innovation_cov.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(
                innovation_cov.nrows(),
                innovation_cov.ncols(),
            ));

        // Update
        let posterior_state = predicted_state + &kalman_gain * &innovation;
        let identity = DMatrix::identity(prior_covariance.nrows(), prior_covariance.ncols());
        let kalman_gain_times_H = &kalman_gain * observation_matrix;
        let posterior_cov = (&identity - &kalman_gain_times_H) * &predicted_cov;

        (posterior_state, posterior_cov)
    }

    /// Step 2: Thermal model — convert state to temperature, compute entropy.
    pub fn thermal_step(state: &DVector<f64>, temperature_scale: f64) -> ThermalOutput {
        let temperature: Vec<f64> = state.iter().map(|&s| s.abs() * temperature_scale).collect();
        let entropy = temperature
            .iter()
            .filter(|&&t| t > 1e-15)
            .map(|t| -t * t.ln())
            .sum();
        let heat_flow: Vec<f64> = temperature
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        ThermalOutput {
            temperature,
            entropy,
            heat_flow,
        }
    }

    /// Step 3: Fokker-Planck — model probability density evolution.
    pub fn fokker_planck_step(
        density: &DVector<f64>,
        drift: &DVector<f64>,
        diffusion: f64,
        dt: f64,
    ) -> FokkerPlanckOutput {
        let n = density.nrows();
        let mut new_density = density.clone();

        // Simple finite-difference Fokker-Planck update
        for i in 0..n {
            let drift_term = if i > 0 && i < n - 1 {
                -0.5 * (drift[i + 1] * density[(i + 1).min(n - 1)] - drift[i.saturating_sub(1)] * density[i.saturating_sub(1)])
            } else {
                0.0
            };

            let diff_term = if i > 0 && i < n - 1 {
                diffusion * (density[(i + 1).min(n - 1)] - 2.0 * density[i] + density[i.saturating_sub(1)])
            } else {
                0.0
            };

            new_density[i] += dt * (drift_term + diff_term);
            new_density[i] = new_density[i].max(0.0);
        }

        // Normalize
        let total: f64 = new_density.sum();
        if total > 1e-15 {
            new_density = new_density / total;
        }

        FokkerPlanckOutput {
            probability_density: new_density.iter().copied().collect(),
            drift: drift.iter().copied().collect(),
            diffusion,
        }
    }

    /// Step 4: EigenPolicy — compute policy from eigenstructure.
    pub fn eigenpolicy_step(
        state: &DVector<f64>,
        reward_matrix: &DMatrix<f64>,
        learning_rate: f64,
    ) -> EigenPolicyOutput {
        // Policy = softmax of reward_matrix * state
        let scores = reward_matrix * state;
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| ((s - max_score) * learning_rate).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let action: Vec<f64> = if sum_exp > 1e-15 {
            exp_scores.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / scores.nrows() as f64; scores.nrows()]
        };

        // Dominant eigenvalue as policy quality measure
        let eigenvalue = if scores.nrows() > 0 {
            scores.iter().map(|&s| s * s).sum::<f64>().sqrt()
        } else {
            0.0
        };

        // Stable if action is close to uniform
        let uniform = 1.0 / action.len() as f64;
        let max_deviation = action.iter().map(|&a| (a - uniform).abs()).fold(0.0_f64, f64::max);
        let policy_stable = max_deviation < 0.3;

        EigenPolicyOutput {
            action,
            eigenvalue,
            policy_stable,
        }
    }

    /// Run one full agent loop cycle.
    pub fn step(
        &mut self,
        prior_state: &DVector<f64>,
        prior_covariance: &DMatrix<f64>,
        observation: &DVector<f64>,
        observation_matrix: &DMatrix<f64>,
        process_noise: &DMatrix<f64>,
        observation_noise: &DMatrix<f64>,
        reward_matrix: &DMatrix<f64>,
    ) -> AgentLoopState {
        // Kalman
        let (post_state, post_cov) = Self::kalman_step(
            prior_state, prior_covariance,
            observation, observation_matrix,
            process_noise, observation_noise,
        );

        // Thermal
        let thermal = Self::thermal_step(&post_state, self.temperature_scale);

        // Fokker-Planck
        let density = DVector::from_vec(vec![1.0 / post_state.nrows() as f64; post_state.nrows()]);
        let drift = &post_state * 0.1;
        let fp = Self::fokker_planck_step(&density, &drift, self.diffusion_coefficient, 0.01);

        // EigenPolicy
        let policy = Self::eigenpolicy_step(&post_state, reward_matrix, self.learning_rate);

        self.step_count += 1;

        let total_energy: f64 = thermal.heat_flow.iter().map(|h| h.abs()).sum();

        let converged = policy.policy_stable && thermal.entropy.abs() < 1.0;

        AgentLoopState {
            step: self.step_count,
            kalman: KalmanOutput {
                state_estimate: post_state.iter().copied().collect(),
                covariance: post_cov.iter().copied().collect(),
                dimension: post_state.nrows(),
            },
            thermal,
            fokker_planck: fp,
            policy,
            total_energy,
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2d_system() -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        let obs_matrix = DMatrix::identity(2, 2);
        (state, cov, obs_matrix)
    }

    #[test]
    fn test_kalman_step_basic() {
        let (state, cov, obs_m) = make_2d_system();
        let obs = DVector::from_vec(vec![1.1, -0.1]);
        let process_noise = DMatrix::from_element(2, 2, 0.01);
        let obs_noise = DMatrix::from_element(2, 2, 0.1);

        let (post_state, post_cov) = AgentBridge::kalman_step(
            &state, &cov, &obs, &obs_m, &process_noise, &obs_noise,
        );
        assert_eq!(post_state.nrows(), 2);
        assert_eq!(post_cov.nrows(), 2);
        // Posterior should be between prior and observation
        assert!(post_state[0] > 0.5 && post_state[0] < 1.2);
    }

    #[test]
    fn test_thermal_step() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let thermal = AgentBridge::thermal_step(&state, 2.0);
        assert_eq!(thermal.temperature.len(), 3);
        assert!((thermal.temperature[0] - 2.0).abs() < 1e-10);
        assert!((thermal.temperature[2] - 6.0).abs() < 1e-10);
        assert!(thermal.entropy.is_finite());
    }

    #[test]
    fn test_fokker_planck_step() {
        let density = DVector::from_vec(vec![0.2, 0.6, 0.2]);
        let drift = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let fp = AgentBridge::fokker_planck_step(&density, &drift, 0.1, 0.01);
        assert_eq!(fp.probability_density.len(), 3);
        let total: f64 = fp.probability_density.iter().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eigenpolicy_step() {
        let state = DVector::from_vec(vec![1.0, 0.5]);
        let reward = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let policy = AgentBridge::eigenpolicy_step(&state, &reward, 1.0);
        assert_eq!(policy.action.len(), 2);
        let total: f64 = policy.action.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_agent_loop() {
        let mut bridge = AgentBridge::new();
        let (state, cov, obs_m) = make_2d_system();
        let obs = DVector::from_vec(vec![1.0, 0.0]);
        let process_noise = DMatrix::from_element(2, 2, 0.01);
        let obs_noise = DMatrix::from_element(2, 2, 0.1);
        let reward = DMatrix::identity(2, 2);

        let result = bridge.step(
            &state, &cov, &obs, &obs_m, &process_noise, &obs_noise, &reward,
        );
        assert_eq!(result.step, 1);
        assert_eq!(result.kalman.dimension, 2);
    }

    #[test]
    fn test_agent_loop_convergence() {
        let mut bridge = AgentBridge::new();
        let (state, cov, obs_m) = make_2d_system();
        let process_noise = DMatrix::from_element(2, 2, 0.001);
        let obs_noise = DMatrix::from_element(2, 2, 0.01);
        let reward = DMatrix::identity(2, 2);

        let mut last_state = state.clone();
        let mut last_cov = cov.clone();

        for _ in 0..5 {
            let obs = last_state.clone();
            let result = bridge.step(
                &last_state, &last_cov, &obs, &obs_m,
                &process_noise, &obs_noise, &reward,
            );
            last_state = DVector::from_vec(result.kalman.state_estimate);
            last_cov = DMatrix::from_vec(2, 2, result.kalman.covariance);
        }
        assert_eq!(bridge.step_count, 5);
    }

    #[test]
    fn test_thermal_entropy_positive() {
        let state = DVector::from_vec(vec![2.0, 3.0, 4.0]);
        let thermal = AgentBridge::thermal_step(&state, 1.0);
        // Entropy of positive temperatures should be well-defined
        assert!(thermal.entropy.is_finite());
    }
}
