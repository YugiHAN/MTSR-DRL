
# MTSR-DRL

This code is the implementation of the paper: **MTSR-DRL: Multi-Timescale State Representation Fusion Augmented Deep Reinforcement Learning for the Modular Hospital Fit-out Scheduling**


## Environment Setup

### 1. Create a Python environment

It is recommended to use **Python 3.8** or **Python 3.9** with a dedicated virtual environment.

Using `conda`:

```bash
conda create -n mtsr-drl python=3.8
conda activate mtsr-drl
````

Using `venv`:

```bash
python -m venv mtsr-drl
source mtsr-drl/bin/activate
```

### 2. Install dependencies

Install the required packages listed in `requirements.txt`: 

```bash
pip install -r requirements.txt
```

---

## Data Generation

Before generating data, configure the required parameters in `params.py`, including dataset settings, problem size, and related options.

After setting `params.py`, generate the data by directly running:

```bash
python data_utils.py
```
---

## Training

After configuring `params.py`, start training by directly running:

```bash
python train.py
```

---

## Testing the Trained Model

After configuring `params.py` and preparing the trained model, test the learned policy by directly running:

```bash
python test_trained_model.py
```

---

## Testing Heuristic Baselines

To evaluate heuristic baselines (FIFO+SPT, MOPNR+SPT, MWKR+SPT, FIFO+EET, MOPNR+EET, MWKR+EET), directly run:

```bash
python test_heuristic.py
```


---

## Testing OR-Tools and Gurobi in the MIC Environment

To test exact or optimization-based solvers **OR-Tools** and **Gurobi** in the MIC environment, directly run:

```bash
python ortools_mic.py
```

## Acknowledgement

The code framework is based on: https://github.com/wrqccc/fjsp-drl
