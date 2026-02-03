# GRU Meta-AirComp-FL for RIS-Assisted Channel Estimation

This project simulates a RIS-assisted AirComp-FL system for fast personalized cascaded channel estimation. User devices run lightweight CNN+GRU regression on few pilots and few local steps, while the server performs Reptile-style meta aggregation and jointly optimizes the BS receive beamformer `f` and RIS phase shifts `theta` to reduce over-the-air (OTA) aggregation distortion.

## Highlights

- RIS-assisted cascaded channel estimation with CNN+GRU regression.
- Meta-FL with Reptile (or FedAvg fallback) across users.
- AirComp aggregation distortion modeling with configurable SNR.
- Joint beamforming and RIS phase optimization per round.
- Optional direct (BS-UE) link and a switch to enable reflection/direct/both.
- Time-varying RU channels via AR(1), with optional user heterogeneity and dynamic alpha(t).
- Optional time-window GRU inputs and per-user local sample cache.

## Project Layout

```text
.
├── aircomp_opt/
│   ├── OTA_sim.py               # AirComp aggregation simulator (noise injection)
│   └── f_theta_optim.py          # Joint BS beamformer and RIS phase optimization
├── data/
│   ├── channel.py                # AR(1) RU channel evolution and alpha(t) schedules
│   ├── deepmimo.py               # DeepMIMO loader and format adapters
│   └── pilot_gen.py              # Pilot pattern generation and observation simulation
├── fl_core/
│   ├── agg.py                    # Meta updater (FedAvg + AirComp option)
│   ├── reptile_agg.py            # Reptile meta-aggregation
│   └── trainer.py                # Local training loop
├── model/
│   └── csi_cnn_gru.py             # CNN+GRU model for cascaded channel regression
├── utils/
│   ├── config.py                 # All simulation and training hyperparameters
│   └── logger.py                 # Simple console/file logger
├── main.py                       # End-to-end simulation entry point
└── README.md
```

## How It Works (High Level)

1. Generate pilot observations using current `H_BR`, `h_RU`, optional `h_BU`, beam `f`, and RIS phases `theta`.
2. Build per-user GRU sequences (optionally using a time window and local cache).
3. Run local training on each user using CNN+GRU regression to estimate cascaded channels.
4. Aggregate local models at the server with Reptile (or FedAvg), optionally through AirComp noise.
5. Optimize `f` and `theta` for the next round.
6. Evolve RU channels with AR(1) dynamics (optional heterogeneity and dynamic alpha).

## Requirements

- Python 3.8+ (recommended)
- `numpy`
- `torch`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (Synthetic Data)

1. Ensure `use_synthetic_data = True` in `utils/config.py`.
2. Run the simulation:

```bash
python main.py
```

## Using DeepMIMO Data

Set `use_synthetic_data = False` and provide `deepmimo_path` in `utils/config.py`.

Supported file formats:

- `.npz` archive containing `H_BR` and one of `h_RU`, `h_RUs`, or `H_RU`. Optional direct link keys: `h_BU`, `h_BUs`, or `H_BU`.
- `.npy` containing a dict with the same keys.
- `.npy` containing a 2D array `(K, N)` interpreted as `h_RU` with a fallback `H_BR` of ones.

Expected shapes:

- `H_BR`: `(N, M)` complex
- `h_RU`: `(K, N)` complex, or `(K, T, N)` for multiple snapshots (first snapshot used)
- `h_BU`: `(K, M)` complex, or `(K, T, M)` if provided (optional direct link)

## Configuration Guide

All knobs live in `utils/config.py`. Key options:

- `num_users`, `num_bs_antennas`, `num_ris_elements`, `num_pilots`
- `num_rounds`, `local_epochs`, `local_lr`, `batch_size`
- `meta_algorithm` (`Reptile` or `FedAvg`) and `reptile_step_size`
- `use_aircomp`, `SNR_dB`, `noise_std`
- `use_time_window`, `window_length`, `window_pad_value`
- `use_local_sample_cache`, `local_cache_size`
- `use_user_pilot_snr_hetero`, `pilot_snr_dB_min`, `pilot_snr_dB_max`
- `use_user_alpha_hetero`, `alpha_user_min`, `alpha_user_max`
- `use_dynamic_alpha`, `dynamic_alpha_mode`, `alpha_min`, `alpha_max`, `alpha_period_rounds`, `alpha_piecewise`
- `link_switch` `[reflect, direct]`: `[1,0]` reflection only, `[0,1]` direct only (no RIS), `[1,1]` both, `[0,0]` invalid

## Outputs and Logging

- Console logs are always printed.
- If `log_to_file = True`, logs are written under `./log/` with a config fingerprint (see `utils/logger.py`).

## Reproducibility

- `main.py` sets fixed seeds for NumPy and PyTorch to `0` by default.

## Notes and Limitations

- This is a research-style simulation, not an optimized production system.
- GPU acceleration is not enabled by default in the trainer.
- DeepMIMO datasets are not bundled with this repository.

## License

No license file is included. If you intend to reuse or distribute this code, add an appropriate license.
