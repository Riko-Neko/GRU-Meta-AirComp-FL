# GRU Meta-AirComp-FL for RIS-Assisted Channel Estimation

This repository contains a research-oriented simulation for RIS-assisted channel estimation under federated learning and over-the-air aggregation (AirComp).

The current main pipeline is centered on:

- mobility-driven synthetic or RIS-S21-initialized channels;
- a shared CNN+GRU backbone with personalized local heads;
- dual-target GRU prediction for current and uplink-time CSI;
- optional GRU path-loss factorization for semantic grouping;
- grouped OTA aggregation with per-group beamforming and RIS optimization.

## Current Status

The default and recommended path is still the end-to-end simulation driven by `main.py` and `utils/config.py`.

The current codebase has changed in two important ways:

- the GRU branch is no longer just a single global backbone update path; it can switch from single-global training to two-group semantic aggregation;
- the GRU head no longer only predicts CSI; it can also factor out RIS path-loss and use that prediction to drive grouping and weighted local loss design.

Data support is currently:

- `use_synthetic_data = True`: mobility-driven synthetic geometry and channel evolution;
- `use_synthetic_data = False`: RIS-S21 measurement data is used as a static initializer, after which the same mobility/path-loss/small-scale fading simulator continues the time evolution.

## What The Project Does

Each communication round follows this high-level loop:

1. Generate pilot observations for every user from the current channels, beamformer, and RIS phase pattern.
2. Build GRU inputs either with persistent hidden states or explicit time windows.
3. Train one local GRU model per user on multiple same-round noisy observations; in persistent mode, physical time advances once per round while repeated samples only increase local training data.
4. Use the GRU branch to predict current CSI and uplink-time CSI, and optionally predict RIS path-loss.
5. Build semantic grouping proxies from the GRU outputs:
   `beta_hat` from predicted RIS path-loss and `d_hat` from relative CSI variation between the two GRU outputs.
6. Smooth those proxies with EMA and detect a plateau; once the plateau condition is met, solve a two-group risk-aware SCA grouping problem to split users into low-risk and high-risk groups.
7. Aggregate only the shared backbone parameters:
   in single mode, there is one global GRU backbone;
   in grouped mode, each group keeps its own global backbone, AirComp aggregation, beamformer `f`, and RIS phase vector `theta`.
8. Keep all per-user GRU heads local while the server only updates shared backbone parameters.
9. Optimize beamforming/RIS variables for the next round and evolve channels with mobility-driven path-loss and small-scale fading.

## Main Components

```text
.
├── aircomp_opt/
│   ├── OTA_sim.py          # AirComp aggregation simulator
│   ├── f_theta_optim.py    # Beam/RIS optimization (OA or SCA backend)
│   └── grouping_optim.py   # Two-group semantic grouping optimizer
├── data/
│   ├── RISdata.py          # RIS-S21 measurement loader and adapter
│   ├── channel.py          # Mobility-driven channel evolution
│   └── pilot_gen.py        # Pilot generation and observation simulation
├── fl_core/
│   ├── agg.py              # FedAvg-style aggregation helpers
│   ├── lmmse.py            # Optional LMMSE baseline
│   ├── model_vector.py     # Model/state flattening utilities
│   ├── reptile_agg.py      # Reptile aggregation helper
│   └── trainer.py          # Local training loops
├── model/
│   ├── csi_cnn_arch.py     # CNN architecture ablation
│   ├── csi_cnn_baseline.py # Literature-style CNN baseline
│   └── csi_cnn_gru.py      # Main GRU model with optional path-loss head
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

- `GRU`: the main method, now supporting semantic two-group aggregation.
- `CNN-arch`: a non-recurrent architecture ablation that keeps the FL/OTA pipeline but removes the GRU state mechanism.
- `CNN-base`: a literature-style memoryless CNN baseline with full-model FedAvg aggregation.
- `LMMSE`: a formula-based pilot estimator evaluated on the same channels.
- `Oracle-true`: an upper-bound reference that optimizes the physical layer from true channel information.

The GRU branch has four details that matter for understanding current results:

- The shared part is still the backbone (`Conv1D + GRU`), while the head remains personalized per user and is cached locally.
- The head predicts two CSI outputs, `t` and `tau`, where `tau` corresponds to the uplink-time target. `gru_csi_target_mode` chooses whether evaluation uses `t`, a linear interpolation, or the direct `tau` head.
- When `enable_gru_pl_factorization=True`, the GRU adds a separate path-loss head (`head_pl`) on top of the same backbone. That scalar is predicted in a bounded log-domain range and trained with a SmoothL1 loss.
- In persistent-hidden mode, local GRU training uses multiple same-round observations conditioned on the previous round hidden state, then runs one canonical inference step to update the hidden state that will be carried into the next round.

When `meta_algorithm=Reptile`, the server only meta-updates the shared GRU backbone, so the local `t/tau` CSI heads and optional path-loss head are coupled through one shared meta-learned representation rather than through a shared prediction head. In the current GRU pipeline, that shared representation is used to jointly expose a slow semantic variable from predicted RIS path-loss (`beta_hat`) and a fast semantic variable from the `t -> tau` channel change (`d_hat`). Those long/short-range proxies are then passed to the grouping optimizer to decide when grouping should start and how user memberships should be updated over time.

The grouped aggregation mechanism is specific to the GRU branch:

- Before switching, there is one shared global backbone for all users.
- After switching, users are partitioned into two semantic groups and each group maintains its own global GRU backbone, AirComp aggregation, beamformer, and RIS phase vector.
- Grouping is driven by a risk-aware SCA optimizer in `aircomp_opt/grouping_optim.py`, not by a fixed manual partition.
- The grouping policy is configurable in both startup and update behavior: you can delay the initial split until semantic proxies stabilize, freeze the partition after the first split, or keep regrouping over time so users with changing channel dynamics can migrate between groups.

## Requirements

The repository currently declares these runtime dependencies in `requirements.txt`:

- `numpy`
- `torch`
- `matplotlib`
- `scipy`

Install them with:

```bash
pip install -r requirements.txt
```

`scipy` is required not only for the RIS-S21 MATLAB loader, but also for the SCA-based grouping optimizer.

## Quick Start

The default configuration already points to the synthetic-data path.

```bash
python main.py
```

With the current defaults, the run will:

- initialize the mobility-driven synthetic channel model;
- train the GRU branch with persistent hidden states;
- enable GRU path-loss factorization and semantic grouping logic;
- write logs under `log/` and figures under `figs/`.

## Important Configuration Knobs

All configuration lives in `utils/config.py`.

### Data and topology

- `use_synthetic_data`
- `risdata_root`, `risdata_subset`
- `num_users`, `num_bs_antennas`, `num_ris_elements`, `num_pilots`
- `link_switch`

### GRU behavior

- `gru_context_mode`
- `gru_csi_target_mode`
- `uplink_tau_ratio`
- `enable_gru_pl_factorization`
- `gru_pl_loss_weight`
- `enable_mobility_aware_loss_weighting`

### Grouped aggregation

- `enable_gru_semantic_grouping`
- `gru_group_switch_min_round`
- `gru_group_switch_patience`
- `gru_group_freeze_after_switch`
- `gru_group_lambda_d`
- `gru_group_k_min`
- `gru_group_switch_ema_lambda`

### FL and OTA

- `meta_algorithm`
- `reptile_step_size`
- `local_epochs`, `local_lr`
- `ota_use_weighted_users`
- `user_data_partition_mode`, `user_data_size_equal`, `user_group_data_sizes`
- `beam_ris_optimizer`

### Optional baselines

- `enable_cnn_arch_ablation`
- `enable_cnn_baseline`
- `enable_lmmse_baseline`

## Outputs

The run produces three main kinds of artifacts:

- `log/*.log`: per-round training, grouping, aggregation, and optimization logs;
- `figs/*.png`: summary curves parsed from the log after the run ends;
- `debug/.../*.png`: optional auxiliary analysis figures.

Compared with the older version of the project, the logs now also expose grouped-GRU diagnostics such as:

- semantic proxy summaries `B` and `D`;
- the round where single mode switches to grouped mode;
- low-risk and high-risk user assignments;
- regrouping decisions for the next round when dynamic grouping is enabled;
- per-group AirComp metrics and per-group backbone update norms;
- whether the partition is frozen or continues to update over time.

## Data Support Notes

### Synthetic data

This remains the main experimental path. `main.py` builds:

- a mobility-driven user geometry around fixed BS/RIS positions;
- direct and reflective large-scale fading from user positions;
- small-scale time evolution controlled by the channel evolver;
- per-round pilot observations for the GRU/CNN/LMMSE branches.

### RIS-S21 loader utility

`data/RISdata.py` provides the MATLAB loader plus the adapter used by `main.py` when `use_synthetic_data = False`. The adapter:

- groups samples by geometry;
- subtracts the chosen reference trace when available;
- solves a least-squares inverse problem over RIS patterns to recover one effective reflection channel per geometry;
- uses that recovered channel as the static initialization for the same grouped-FL simulation loop.

## Known Limitations

- Configuration is hard-coded in `utils/config.py`; there is no CLI or experiment launcher.
- GRU semantic grouping currently assumes exactly two groups.
- Grouped GRU mode requires reflection to be enabled, persistent-hidden context, and GRU path-loss factorization.
- The RIS-S21 path is still a static initializer; the later temporal evolution remains simulated rather than measurement-native.
- Training runs on CPU by default unless you modify the trainer/model placement.

## License

This project is released under the MIT License. See `LICENSE`.
