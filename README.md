
# MTSR-DRL

Implementation of the paper:

**MTSR-DRL: Multi-Timescale State Representation Fusion Augmented Deep Reinforcement Learning for the Modular Hospital Fit-out Scheduling**


## Environment Setup

Create a Python environment (recommended: Python 3.8):

```bash
conda create -n mtsr-drl python=3.8
conda activate mtsr-drl
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include PyTorch, NumPy, Pandas, OR-Tools, and tqdm. 


## Data Generation

Configure parameters in `params.py`, then generate data:

```bash
python data_utils.py
```

---

## Training

Train the model:

```bash
python train.py
```

Models and logs are saved under:

```
./trained_network/
./train_log/
```

---

## Testing (Learned Model)

Test trained models:

```bash
python test_trained_model.py
```

Default mode uses the **greedy strategy**.

---

## Testing (Heuristic Baselines)

Run heuristic methods:

```bash
python test_heuristic.py
```

Evaluated rules:

```python
['FIFO+SPT', 'MOPNR+SPT', 'MWKR+SPT',
 'FIFO+EET', 'MOPNR+EET', 'MWKR+EET']
```

---

## Testing (OR-Tools & Gurobi)

Run:

```bash
python ortools_mic.py
```

Notes:

* OR-Tools is installed via `requirements.txt`. 
* Gurobi requires separate installation and a valid license:

```bash
pip install gurobipy
```

---

## Workflow

```bash
python data_utils.py
python train.py
python test_trained_model.py
python test_heuristic.py
python ortools_mic.py
```

```
```
