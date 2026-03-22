# GRU Meta-AirComp-FL for RIS-Assisted Channel Estimation

This repository contains a research-oriented simulation for RIS-assisted channel estimation under federated learning and over-the-air aggregation (AirComp).

The main branch of the code uses:

- synthetic BS-RIS-user channels;
- a shared CNN+GRU backbone with personalized local heads;
- Reptile or FedAvg style aggregation;
- optional AirComp noise and OTA-aware beam/RIS optimization;
- optional CNN ablations and an oracle physics-only reference branch.

## Current Status

The default and recommended path is the synthetic-data simulation driven by `main.py` and `utils/config.py`.

The repository also contains a measurement-data loader utility in `data/RISdata.py`, and the non-synthetic branch in `main.py` now uses it through a lightweight adapter that reconstructs one effective reflection channel per geometry from measured RIS patterns and S21 traces. In practice:

- `use_synthetic_data = True` is the path that currently matches the codebase and existing logs/figures.
- `use_synthetic_data = False` uses RIS-S21 measurement data as a static channel initializer, then reuses the same mobility-driven large-scale fading plus AR(1)/OU small-scale evolution as the synthetic branch.

## What The Project Does

Each communication round follows this high-level loop:

1. Generate pilot observations for every user from the current BS beamformer, RIS phase pattern, and channels.
2. Build GRU inputs either as single-step observations with persistent hidden states or as explicit time windows.
3. Train one local model per user.
4. Aggregate only the shared backbone updates at the server, while keeping each user's prediction head local.
5. Simulate OTA aggregation distortion when AirComp is enabled.
6. Optimize the BS beamformer `f` and RIS phase vector `theta` for the next round.
7. Evolve user-dependent channels with explicit position updates, distance-driven path-loss, and AR(1)/OU small-scale fading.

## Main Components

```text
.
├── aircomp_opt/
│   ├── OTA_sim.py          # AirComp aggregation simulator
│   └── f_theta_optim.py    # Legacy and state-aware beam/RIS optimization
├── data/
│   ├── RISdata.py          # RIS-S21 measurement loader utilities
│   ├── channel.py          # Mobility-driven path-loss and AR(1)/OU small-scale evolution
│   └── pilot_gen.py        # Pilot generation and observation simulation
├── fl_core/
│   ├── agg.py              # FedAvg-style aggregation helpers
│   ├── model_vector.py     # Model/state flattening utilities
│   ├── reptile_agg.py      # Reptile aggregation helper
│   ├── state_metrics.py    # State-aware weighting metrics
│   └── trainer.py          # Local training loops
├── model/
│   ├── csi_cnn_arch.py     # CNN architecture ablation
│   ├── csi_cnn_baseline.py # Literature-style CNN baseline
│   └── csi_cnn_gru.py      # Main shared-backbone + local-head GRU model
├── utils/
│   ├── config.py           # Simulation and training configuration
│   ├── log_plotter.py      # Parse logs and plot round metrics
│   └── logger.py           # Console/file logger
├── debug/                  # Optional debug figures
├── figs/                   # Auto-generated summary figures
├── log/                    # Training logs
└── main.py                 # End-to-end simulation entry point
```

## Model Branches

`main.py` can run several branches side by side:

- `GRU`: the main method, using a shared CNN+GRU backbone and per-user local heads.
- `CNN-arch`: an architecture ablation that replaces the GRU backbone with a non-stateful CNN pooling backbone while keeping the same FL/OTA setup.
- `CNN-base`: a literature-style memoryless CNN baseline with full-model FedAvg aggregation.
- `Oracle-true`: an upper-bound reference that skips learning and optimizes the physical layer using true channel information.

The main personalization rule in the GRU branch is:

- backbone parameters are aggregated globally;
- head parameters stay on-device and are never OTA-aggregated;
- the first round warm-starts all local heads from the same global initialization.

## Requirements

The repository currently declares these runtime dependencies in `requirements.txt`:

- `numpy`
- `torch`
- `matplotlib`

Install them with:

```bash
pip install -r requirements.txt
```

If you plan to use the RIS-S21 MATLAB loader in `data/RISdata.py`, install `scipy` as an extra dependency:

```bash
pip install scipy
```

## Quick Start

The default configuration already points to the synthetic-data path.

```bash
python main.py
```

By default, the script will:

- initialize synthetic channels;
- run 50 communication rounds;
- write a log file under `log/`;
- generate a round-metric figure under `figs/` when the run finishes.

## Important Configuration Knobs

All configuration lives in `utils/config.py`.

### Data and topology

- `use_synthetic_data`
- `risdata_root`
- `risdata_subset`
- `risdata_result_key`
- `risdata_reference_key`
- `num_users`
- `num_bs_antennas`
- `num_ris_elements`
- `num_pilots`
- `link_switch`

### Channel dynamics

- `bs_position_xy`, `ris_position_xy`
- `user_cluster_ratios`
- `user_cluster_centers_xy`
- `user_cluster_position_jitter_xy`
- `user_speed_range`
- `user_motion_direction_deg`
- `user_speed_user_mask`
- `channel_time_step`
- `channel_carrier_frequency_hz`
- `alpha_direct`
- `pilot_SNR_dB`

### GRU context and prediction target

- `gru_context_mode`: `persistent_hidden` or `time_window`
- `gru_csi_target_mode`: `t`, `t+1`, or `uplink_linear`
- `window_length`
- `window_pad_value`
- `reset_hidden_on_round1`

### Federated optimization

- `meta_algorithm`: `Reptile` or `FedAvg`
- `reptile_step_size`
- `local_epochs`
- `local_lr`
- `batch_size`
- `ota_use_weighted_users`
- `user_data_partition_mode`
- `user_group_ratios`
- `user_group_data_sizes`

### AirComp and physical-layer optimization

- `use_aircomp`
- `SNR_dB`
- `ota_tx_power`
- `ota_var_floor`
- `oa_optimizer_mode`: `legacy` or `state_aware`
- `oa_ao_iters`
- `oa_theta_lr`
- `oa_theta_grad_steps`

### Optional branches and debug output

- `enable_cnn_baseline`
- `enable_cnn_arch_ablation`
- `enable_reptile_head_debug_plot`
- `enable_gru_dual_target_debug_log`
- `enable_gru_state_diff_debug_plot`

## Outputs

The run produces three main kinds of artifacts:

- `log/*.log`: per-round training and aggregation logs with a configuration fingerprint in the filename;
- `figs/*.png`: summary curves parsed from the log after the run ends;
- `debug/.../*.png`: optional Reptile head projections and GRU state-delta diagnostics when debug plotting is enabled.

The log parser extracts metrics such as:

- mean local loss per round;
- global update norm;
- AirComp aggregation NMSE;
- post-optimization proxy NMSE.

## Data Support Notes

### Synthetic data

This is the supported path today. `main.py` builds:

- a random BS-RIS channel `H_BR`;
- per-user RIS-UE channels `h_RUs`;
- optional direct BS-UE channels `h_BUs`;
- per-round pilot observations through `data/pilot_gen.py`.

### RIS-S21 loader utility

`data/RISdata.py` provides the MATLAB loader plus the adapter used by `main.py` when `use_synthetic_data = False`. The adapter:

- groups samples by geometry;
- subtracts the chosen reference trace when available;
- solves a least-squares inverse problem over RIS patterns to recover one effective reflection channel per geometry;
- returns that channel to the existing FL/OTA simulation as the static initialization.

## Known Limitations

- Configuration is hard-coded in `utils/config.py`; there is no CLI or experiment launcher.
- The supported execution path is the synthetic simulation.
- The RIS-S21 path is an adapter, not a full measurement-native pipeline: after static initialization, temporal evolution is still simulated.
- Training runs on CPU by default unless you modify the trainer/model placement.
- `requirements.txt` does not include optional dataset-loader dependencies such as `scipy`.

## License

This project is released under the MIT License. See `LICENSE`.
