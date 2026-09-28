# BOxCrete E2E

## Bayesian Optimization and Gaussian Process Framework for Concrete Strength Forecasting

BOxCrete E2E is an end-to-end machine learning framework for **concrete compressive-strength forecasting, uncertainty estimation, and future mix-design optimization**.

The project extends the original BOxCrete research workflow into a reproducible software pipeline with:

- Group-aware dataset splitting
- Data correction and validation
- Feature engineering
- Source-aware Gaussian Process modeling
- Time-aware strength prediction
- Predictive uncertainty estimation
- Block Leave-One-Out (Block-LOO) validation
- MLflow experiment tracking
- Reproducible model artifacts
- External zero-shot dataset validation
- Future model release and inference support

---

## Project Status

**Current development stage: External validation and model release preparation**

The current production-oriented strength model (V2) has been trained and evaluated using the internal BOxCrete dataset.

### Current V2 validation

| Metric | Result |
|---|---:|
| MAE | **514.38 psi** |
| RMSE | **714.46 psi** |
| R² | **0.9194** |
| Mean predictive σ | **689.08 psi** |
| 95% predictive coverage | **77.78%** |
| Training rows | 468 |
| Validation rows | 99 |
| Input dimensions | 10 |
| Device | CUDA |

The next major step is **zero-shot evaluation on an external concrete-strength dataset**, followed by model packaging and release.

---

# 1. Research Background

The original BOxCrete study developed a Gaussian Process based framework for forecasting concrete and mortar compressive strength.

The original experimental dataset contains:

- 123 unique mixtures
- 533 unique strength measurements
- Mortar and concrete mixtures
- Multiple curing ages
- Gaussian Process regression
- Bayesian optimization for mix-design exploration

The published model achieved approximately:

- **R² ≈ 0.94**
- **RMSE ≈ 0.69 ksi**

across the reported evaluation sets.

The current E2E implementation reorganizes this research workflow into a reproducible machine-learning project suitable for experimentation, external validation, MLflow tracking, and future deployment.

---

# 2. Main Objectives

BOxCrete E2E aims to provide a complete pipeline for:

1. Concrete strength prediction
2. Predictive uncertainty estimation
3. Time-dependent strength modeling
4. Material-source-aware modeling
5. External dataset validation
6. Experiment tracking
7. Reproducible model artifacts
8. Future concrete mix optimization
9. Model deployment and inference

---

# 3. Current Model: Strength V2

The current V2 model is a time-aware Gaussian Process model designed specifically for the BOxCrete dataset.

## Input Features

The raw model interface uses 10 variables:

```text
Cement
Fly Ash
Slag
Water
HRWR
Fine Aggregate
Coarse Aggregates
Material Source
Temperature
Time
```

Where:

- `Material Source` represents the experimental material/source category.
- `Temperature` represents curing/environmental temperature.
- `Time` represents curing age.

---

# 4. Feature Engineering

V2 uses engineered F5 features in addition to the raw inputs.

The engineered features are:

```text
wb_ratio
scm_frac
log_hrwr_binder
log_wc_ratio
log_coarse_fine
log_agg_paste
log_maturity_robust
```

These features provide physically meaningful relationships between:

- Water and binder
- Supplementary cementitious materials
- High-range water reducer
- Aggregate proportions
- Paste volume
- Curing maturity

The engineered features are appended to the raw input representation before kernel evaluation.

---

# 5. Gaussian Process Architecture

The V2 model combines several components.

### Multi-Matern kernel

The model uses:

```text
Blind Matern branch
        +
Source-specific Matern branch
        +
RBF time component
```

The source-aware component allows the model to learn differences between material-source categories.

### Time gating

The covariance is modulated using a time gate:

```text
h(t) = 1 - exp(-t / τ)
```

with:

```text
τ = 0.1
```

The time gate is used to model the effect of curing time on covariance and predictive uncertainty.

### Heteroscedastic uncertainty

The model uses a time-dependent noise formulation:

```text
σ²(t) = h(t)² σ²_global
```

This allows predictive noise to vary with curing time.

---

# 6. Target Transformation

The strength target is modeled in a scaled representation.

The model uses:

```text
Y_scaled = Y / Y_max
```

together with a zero-mean GP representation.

Predictions are transformed back to the original strength unit during inference.

The current internal evaluation reports strength in:

```text
psi
```

---

# 7. Training Objective

The V2 model combines:

1. Block Leave-One-Out predictive negative log likelihood
2. Gaussian Process marginal likelihood

The combined objective is:

```text
L =
(1 - λ) × Block-LOO NLL
+
λ × MLL NLL
```

with the current default:

```text
λ = 0.5
```

Optimization uses the project-specific Block-LOO training procedure.

---

# 8. Block Leave-One-Out Validation

Random row-level splitting is avoided because multiple measurements belong to the same concrete mixture.

Instead, samples are grouped according to their mixture composition.

This prevents measurements from the same mixture appearing simultaneously in training and validation.

The current frozen split contains:

```text
Training mixtures:   104
Validation mixtures:  22
Test mixtures:        23
```

Corresponding rows:

```text
Training:    468
Validation:   99
Test:        103
```

The split is performed at the **mixture level**, not at the individual measurement level.

---

# 9. Dataset Structure

The project separates raw, corrected, processed, and model-ready data.

```text
data/
├── raw/
│   └── boxcrete_data.csv
│
├── corrections/
│   └── boxcrete_corrections.csv
│
├── processed/
│   ├── mix_splits.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── model_ready/
│   ├── strength_train.csv
│   ├── strength_validation.csv
│   ├── strength_test.csv
│   │
│   ├── gwp_train.csv
│   ├── gwp_validation.csv
│   ├── gwp_test.csv
│   │
│   ├── slump_train.csv
│   ├── slump_validation.csv
│   ├── slump_test.csv
│   │
│   └── feature_importance/
│
└── external/
    └── uci_concrete_strength/
        ├── raw.csv
        ├── metadata.yaml
        ├── mapped.csv
        └── evaluation.csv
```

---

# 10. Raw Dataset

The original BOxCrete dataset contains:

```text
670 rows
149 unique mix names
```

The dataset contains measurements at:

```text
1 day
3 days
5 days
14 days
28 days
```

Material-source categories include:

```text
Source 0 → Mortar
Source 1 → Concrete
Source 2 → Concrete
```

For mortar mixtures:

```text
Coarse Aggregate = 0
```

Concrete mixtures contain non-zero coarse aggregate quantities.

---

# 11. Data Corrections

The original raw dataset is preserved without modification.

Known corrections are stored separately in:

```text
data/corrections/boxcrete_corrections.csv
```

This allows the project to maintain:

```text
Raw data
    ↓
Correction manifest
    ↓
Processed data
    ↓
Model-ready data
```

The correction workflow is intentionally separated from the raw dataset to preserve provenance.

---

# 12. Reproducibility

The project uses a frozen group-aware split.

The same mixture must not appear in multiple dataset partitions.

The split is designed to preserve material-source and curing-age distributions while preventing mixture leakage.

---

# 13. MLflow Experiment Tracking

MLflow is used to track:

- Experiments
- Training runs
- Hyperparameters
- Metrics
- Model artifacts
- Validation predictions
- Metadata

The main experiment is:

```text
BOxCrete
```

The V2 run is tracked as:

```text
strength_v2_block_loo
```

---

# 14. Starting the MLflow UI

Start the MLflow tracking server from the project root:

```bash
cd ~/Projects/boxcrete-e2e

mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --host 127.0.0.1 \
    --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

The MLflow Tracking UI allows experiment runs, metrics, parameters, and artifacts to be inspected.

For the training process, point the MLflow client to the server:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Then execute training in another terminal.

---

# 15. Training V2

From the project root:

```bash
cd ~/Projects/boxcrete-e2e

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python -m src.models.run_strength_v2
```

The training script performs the V2 strength-model training and records the relevant experiment information in MLflow.

---

# 16. V2 Artifacts

Model artifacts are stored under:

```text
artifacts/
└── strength_v2/
    ├── model_state.pt
    ├── validation_predictions.csv
    └── metadata.json
```

### `model_state.pt`

Contains the trained PyTorch/GP model state.

### `validation_predictions.csv`

Contains validation predictions and uncertainty information.

Typical fields include:

```text
true
prediction
predictive_std
lower_95
upper_95
```

### `metadata.json`

Stores information required to reproduce and understand the model, including:

- Architecture
- Training configuration
- Dataset information
- Feature configuration
- Evaluation metrics
- Model settings

---

# 17. Current V2 Results

The current internal V2 evaluation produced:

```text
MAE  : 514.38 psi
RMSE : 714.46 psi
R²   : 0.9194

Mean predictive σ : 689.08 psi
95% coverage      : 77.78%
```

The point-prediction performance is strong within the current internal evaluation.

However, the uncertainty interval currently has lower empirical 95% coverage than its nominal 95% level.

Therefore, uncertainty calibration remains an area for further investigation.

---

# 18. External Validation

The next major stage is evaluation on an external dataset.

The selected external dataset is the:

```text
UCI Concrete Compressive Strength Dataset
```

The dataset contains approximately:

```text
1030 samples
```

with eight mixture/process variables and compressive strength as the target.

The external variables include:

```text
Cement
Blast Furnace Slag
Fly Ash
Water
Superplasticizer
Coarse Aggregate
Fine Aggregate
Age
```

Target:

```text
Concrete compressive strength
```

---

# 19. Zero-Shot External Evaluation

The UCI dataset will first be used for **zero-shot evaluation**.

The model will not be retrained using UCI data before this evaluation.

The workflow is:

```text
UCI raw dataset
        │
        ▼
Schema validation
        │
        ▼
Feature mapping
        │
        ▼
V2 inference
        │
        ▼
Predictions + uncertainty
        │
        ▼
External evaluation
```

This provides a more meaningful assessment of how the released model behaves on data outside its original training dataset.

---

# 20. External Validation Metrics

The external evaluation will report:

### Point prediction

```text
MAE
RMSE
R²
```

### Uncertainty

```text
Mean predictive σ
Median predictive σ

50% coverage
80% coverage
90% coverage
95% coverage
```

The evaluation will also be separated by curing age where appropriate.

---

# 21. Age-Based External Evaluation

The UCI dataset contains ages extending beyond 28 days.

Therefore, external evaluation will distinguish:

```text
Age ≤ 28 days
```

from:

```text
Age > 28 days
```

The second category represents a more challenging temporal extrapolation relative to the BOxCrete training range.

Results will therefore not be treated as a single homogeneous validation population.

---

# 22. External Dataset Compatibility

The V2 model expects:

```text
Cement
Fly Ash
Slag
Water
HRWR
Fine Aggregate
Coarse Aggregates
Material Source
Temperature
Time
```

The UCI dataset does not directly provide:

```text
Material Source
Temperature
```

Therefore, the external evaluation requires an explicit feature-mapping strategy.

The project will **not silently assign arbitrary values and describe the resulting evaluation as unbiased validation**.

The mapping assumptions will be documented in:

```text
data/external/uci_concrete_strength/metadata.yaml
```

---

# 23. Important External Validation Principle

The external dataset will remain separate from the original training data.

The correct order is:

```text
Internal training
      ↓
Internal evaluation
      ↓
External zero-shot evaluation
      ↓
Analysis
      ↓
Only then consider external-data adaptation/retraining
```

This prevents accidental contamination of the external validation benchmark.

---

# 24. Model Release Plan

The planned release workflow is:

```text
                 BOxCrete Raw Data
                         │
                         ▼
                Data Validation
                         │
                         ▼
                 Frozen Data Split
                         │
                         ▼
                 V2 Model Training
                         │
                         ▼
                 Internal Evaluation
                         │
                         ▼
              External Zero-Shot Test
                         │
                         ▼
                Model Calibration
                         │
                         ▼
                  Model Packaging
                         │
                         ▼
                    Release
```

The release should include:

```text
Model weights
Feature specification
Input schema
Preprocessing configuration
Training metadata
Evaluation results
Uncertainty information
Inference example
License
Citation
```

---

# 25. Planned Inference Interface

A future inference interface will accept the 10 V2 input variables:

```text
Cement
Fly Ash
Slag
Water
HRWR
Fine Aggregate
Coarse Aggregates
Material Source
Temperature
Time
```

and return:

```text
Predicted compressive strength
Predictive uncertainty
Lower confidence/credible bound
Upper confidence/credible bound
```

A future API may expose the model through:

```text
FastAPI
```

with a web frontend implemented separately.

---

# 26. Planned Mix Optimization

The long-term BOxCrete objective is not only prediction.

The framework is intended to support concrete mix optimization using predicted strength together with additional objectives such as:

```text
Strength
GWP
Cost
Material constraints
```

The optimization stage will be developed after the predictive model and external validation pipeline are finalized.

---

# 27. Future Objectives

Planned development includes:

- [ ] Complete UCI zero-shot evaluation
- [ ] Evaluate uncertainty calibration
- [ ] Evaluate age-specific performance
- [ ] Compare internal and external performance
- [ ] Package V2 model
- [ ] Freeze inference preprocessing
- [ ] Create reproducible inference script
- [ ] Release model weights
- [ ] Add FastAPI inference service
- [ ] Add web interface
- [ ] Add model/version metadata
- [ ] Extend GWP prediction
- [ ] Extend slump prediction
- [ ] Implement constrained mix optimization
- [ ] Add deployment documentation

---

# 28. Technology Stack

## Programming

```text
Python
```

## Machine Learning

```text
PyTorch
GPyTorch
BoTorch
scikit-learn
NumPy
Pandas
```

## Experiment Tracking

```text
MLflow
SQLite
```

## Development

```text
Git
Linux
CUDA
NVIDIA GPU
```

## Future Deployment

```text
FastAPI
HTML
CSS
JavaScript
Docker
```

---

# 29. Repository Structure

```text
boxcrete-e2e/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   └── boxcrete_data.csv
│   │
│   ├── corrections/
│   │   └── boxcrete_corrections.csv
│   │
│   ├── processed/
│   │   ├── mix_splits.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   ├── model_ready/
│   │   ├── strength_train.csv
│   │   ├── strength_validation.csv
│   │   ├── strength_test.csv
│   │   ├── gwp_train.csv
│   │   ├── gwp_validation.csv
│   │   ├── gwp_test.csv
│   │   ├── slump_train.csv
│   │   ├── slump_validation.csv
│   │   └── slump_test.csv
│   │
│   └── external/
│       └── uci_concrete_strength/
│
├── src/
│   ├── boxcrete/
│   │   ├── strength_model.py
│   │   ├── block_loo.py
│   │   ├── features.py
│   │   ├── kernels.py
│   │   ├── likelihoods.py
│   │   └── priors.py
│   │
│   └── models/
│       ├── run_strength_v2.py
│       └── gp_strength.py
│
├── artifacts/
│   └── strength_v2/
│       ├── model_state.pt
│       ├── validation_predictions.csv
│       └── metadata.json
│
├── mlruns.db
│
└── tests/
```

---

# 30. Reproducibility Guidelines

The following principles are used throughout the project:

### Raw data is immutable

Never overwrite:

```text
data/raw/
```

### Corrections are explicit

Changes to raw observations are recorded in:

```text
data/corrections/
```

### Splits are frozen

Training, validation, and test mixture assignments should not change between experiments unless a new experiment explicitly defines a new split.

### External data is isolated

External validation data must not enter the training pipeline before zero-shot evaluation.

### Model artifacts are versioned

Released model versions should use separate artifact directories, for example:

```text
artifacts/
└── strength_v2/
    └── v1/
```

Future versions can then be stored as:

```text
v2/
v3/
...
```

---

# 31. Development Workflow

Recommended workflow:

```text
1. Modify code
       ↓
2. Run tests
       ↓
3. Train model
       ↓
4. Log experiment to MLflow
       ↓
5. Evaluate validation/test data
       ↓
6. Inspect predictions
       ↓
7. Compare MLflow runs
       ↓
8. Freeze successful model
       ↓
9. Test external dataset
       ↓
10. Package release
```

---

# 32. Model Governance

Every released model should record:

```text
Model version
Training dataset version
Feature version
Training configuration
Random seed
Software environment
Training date
Evaluation dataset
Evaluation metrics
Uncertainty metrics
Known limitations
```

This allows a released prediction to be traced back to the exact model and dataset used to produce it.

---

# 33. Limitations

The current BOxCrete research and implementation have several limitations.

The original experimental work does not model all concrete performance characteristics.

In particular, the current strength-focused workflow does not provide a complete prediction of:

```text
Workability
Durability
Long-term structural behavior
All environmental exposure conditions
```

External validation is also constrained by differences between datasets, including:

```text
Material composition
Experimental procedures
Curing conditions
Feature definitions
Measurement distributions
Material-source information
Temperature information
```

Therefore, external performance should be interpreted together with dataset compatibility and feature-mapping assumptions.

---

# 34. Citation

If you use BOxCrete or the underlying research, please cite the original work:

```bibtex
@article{boxcrete,
  title={BOxCrete: A Bayesian Optimization Open-Source AI Model
         for Concrete Strength Forecasting and Mix Optimization},
  author={...},
  journal={...},
  year={...}
}
```

Replace the placeholder bibliographic fields with the final published citation information before release.

---

# 35. License

License information will be added before the public model release.

```text
License: TBD
```

---

# 36. Acknowledgements

This project builds upon the original BOxCrete research and the broader application of Gaussian Processes, Bayesian optimization, and machine learning to concrete-material modeling.

---

# 37. Contact

**Teshan Jayasinghe**

Research interests:

```text
AI / ML
Gaussian Processes
Physics-guided Machine Learning
Computer Vision
Embedded Systems
Engineering AI
Concrete Material Modeling
```

---

## Status

**BOxCrete E2E is currently in the external-validation and model-release preparation stage.**
```
