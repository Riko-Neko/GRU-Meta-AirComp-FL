# Meta-AirComp-FL using GRU

This project simulates a **RIS-assisted AirComp-FL** system for **fast personalized cascaded channel estimation**.  
User devices perform lightweight **CNN+GRU-based regression** with few pilots and few inner-loop steps, while the server applies **Reptile Meta-FL** aggregation and **joint optimization of the BS receive beamformer (f) and RIS phase shifts (θ)** to suppress over-the-air (OTA) aggregation distortion.

---

## Directory Structure

```text
RIS_FL_Simulator/
├── data/
│   ├── deepmimo_loader.py       # DeepMIMO channel data loading and preprocessing
│   └── pilot_simulator.py       # Pilot transmission simulation (observations + cascaded CSI)
├── model/
│   └── csi_cnn_gru.py           # CNN+GRU model for cascaded channel estimation
├── fl_core/
│   ├── trainer.py               # Local training logic (GRUTrainer)
│   ├── aggregator.py            # Meta-FL aggregator base class
│   └── reptile_aggregator.py    # Reptile Meta-FL aggregator
├── aircomp_opt/
│   ├── aircomp_simulator.py     # Over-the-air aggregation (AirComp) simulator
│   └── beam_ris_optimizer.py    # Joint BS beamforming and RIS phase optimization
├── utils/
│   ├── config.py                # Global configuration and hyperparameters
│   └── logger.py                # Lightweight logging utility
├── main.py                      # End-to-end simulation entry point
└── README.md
```

## System Overview
model to map pilot observations to the cascaded channel, minimizing MSE.

3. **Meta-FL aggregation (outer loop)**  
   The server aggregates local models using the **Reptile** algorithm, producing a global initialization that can rapidly adapt to new users or time windows.

4. **AirComp distortion modeling**  
   Uplink model updates are aggregated via an AirComp simulator, injecting noise to emulate OTA aggregation distortion.

5. **Joint communication optimization**  
   The server updates **(f, θ)** via alternating optimization to align effective channels and reduce AirComp aggregation error in the next round.

6. **Channel evolution**  
   User channels evolve according to an **AR(1) / Jakes-like temporal model**, enabling controlled time variation and user heterogeneity.

---

## Key Components

### Data (`data/`)

- **deepmimo_loader.py**  
  Loads BS–RIS and RIS–UE channels from **DeepMIMO** (default: **O1** scenario).

- **pilot_simulator.py**  
  Simulates pilot transmission, generates received observations and cascaded channel ground truth.  
  Also includes a development sanity-pack generator for small-scale debugging (`.npz`).

### Model (`model/`)

- **csi_cnn_gru.py**  
  CNN extracts stable local structure from pilot observations; GRU captures temporal correlation;  
  the output head regresses the cascaded channel (real + imaginary).

### Federated Learning Core (`fl_core/`)

- **trainer.py**  
  Implements device-side local training for cascaded channel regression.

- **aggregator.py**  
  Base Meta-FL aggregator with optional AirComp noise injection.

- **reptile_aggregator.py**  
  Reptile outer-loop update: global parameters move toward the mean of locally adapted models.

### Communication & AirComp (`aircomp_opt/`)

- **aircomp_simulator.py**  
  Models OTA aggregation with additive noise and computes aggregation distortion.

- **beam_ris_optimizer.py**  
  Alternating optimization of BS receive beamformer **f** and RIS phases **θ** to minimize aggregation error.

### Utilities (`utils/`)

- **config.py**  
  Centralized configuration of system size, training hyperparameters, SNR, and simulation options.

- **logger.py**  
  Minimal logging utility for experiment tracking.

---

## Running the Simulation

1. **Install dependencies**
   ```bash
   pip install numpy torch
   ```
2. **Configure parameters**  
   Edit `utils/config.py` (e.g., number of users, RIS elements, SNR, number of rounds).

3. **Run**
   ```bash
   python main.py
   ```