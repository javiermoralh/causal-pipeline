# An End-to-End Pipeline for Causal ML with Continuous Treatments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive framework for Causal Machine Learning with continuous treatments, designed for real-world industrial applications.**

This repository implements the methodology described in our KDD 2025 paper: *"An end-to-end pipeline for Causal ML with continuous treatments: An application to financial decision making"*. The framework addresses critical challenges in applying causal inference to industrial-scale problems, including high dimensionality, positivity assumption violations, and continuous treatment effects.

## 🎯 Key Contributions

- **Positivity Assumption Violation Handling**: Framework to detect and quantify positivity violations in continuous treatment settings + different remediation strategies suggestions.
- **Scalable Dimensionality Reduction**: Two-stage selection framework specifically designed for causal inference with high-dimensional data.
- **Continuous Treatment Adaptations**: Extension of sensitivity analysis and estimation methods from binary to continuous treatments.
- **End-to-End Integration**: Complete modular workflow bridging the gap between academic theory and practical implementation.

## 🔧 Problem Statement

Traditional causal inference methods face significant limitations in industrial applications:

- **High dimensionality** in modern datasets creates computational and statistical challenges
- **Continuous treatments** are common in practice but poorly supported by existing tools
- **Positivity violations** occur frequently when human decision-making limits treatment exposure
- **Limited toolkits** provide comprehensive end-to-end solutions for real-world scenarios

Our paper specifically addresses these challenges in the context of **financial debt collection**, where the goal is to determine optimal write-down percentages to maximize recovery while minimizing loss.

## 🏗️ Pipeline Architecture

The framework consists of six sequential, modular steps:

### 1. **Dimensionality Reduction**
**Challenge**: High-dimensional observational data introduces computational complexity for causal discovery algorithms and iterative refinement with expert-knowledge.

**Solution**: Two-stage feature selection framework:
- **Stage 1**: Treatment-predictive feature selection using hybrid FCBF + Sequential Forward Selection
- **Stage 2**: Dual partial correlation analysis for confounder and outcome predictor identification

**Output**: Reduced adjustment set $Z = Z_T \cup Z_Y$ maintaining causal validity

### 2. **Causal Identification**
**Challenge**: Algorithmic causal discovery alone is insufficient for reliable causal graph construction.

**Solution**: Hybrid approach combining:
- **Algorithmic discovery**: Ensemble of PC, FCI, and GES algorithms
- **Domain expertise**: Iterative refinement through expert validation
- **Temporal constraints**: Enforcement of known variable orderings

**Output**: Expert-validated causal graph and final adjustment set

### 3. **Positivity Assumption Violation Handling**
**Challenge**: Systematic violations of positivity assumption in observational data.

**Solution**: Three-step model-agnostic procedure:
- **Detection**: Generalized Propensity Score (GPS) based violation quantification
- **Quantification**: Continuous severity measures across treatment space
- **Remediation**: Domain-informed data augmentation strategies

**Output**: Augmented dataset with improved treatment overlap

### 4. **Effect Estimation**
**Challenge**: Standard estimators designed for binary treatments poorly handle continuous interventions.

**Solution**: Benchmarking of three approaches:
- **Linear regression with interactions**: Interpretable baseline
- **S-learner**: Flexible single-model approach with regularization
- **Modified AIPTW**: Doubly-robust estimation adapted for continuous treatments

**Output**: Individual dose-response curves and treatment effect estimates

### 5. **Refutation and Evaluation**
**Challenge**: Validating causal estimates without ground truth.

**Solution**: Comprehensive validation framework:
- **Placebo tests**: Treatment replacement to verify effect attribution
- **Random common cause tests**: Stability assessment via synthetic confounders
- **E-value sensitivity analysis**: Potential unmeasured confounding quantification for continuous treatments
- **Qini curves**: Uplift-oriented performance evaluation

**Output**: Validated estimates with robustness metrics and final estimate selection

### 6. **Policy Optimization**
**Challenge**: Translating causal estimates into actionable business decisions.

**Solution**: Counterfactual-driven optimization:
- **Individual treatment recommendations**: Personalized optimal write-down levels
- **Expected value calculations**: Cost-benefit analysis integration
- **Policy simulation**: Risk assessment under different intervention strategies

**Output**: Optimized treatment policies and decision support tools

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/javiermoralh/causal-pipeline.git
cd causal-pipeline

# Create and activate virtual environment
python -m venv causal_env
source causal_env/bin/activate  # On Windows: causal_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Experimental Results

Our methodology demonstrates significant improvements over baseline approaches:

| Method | Mean Bias | 95% CI |
|--------|-----------|---------|
| Baseline (Full Features) | 0.387 | [0.375, 0.397] |
| Adjustment Set Only | 0.339 | [0.317, 0.359] |
| **Proposed Pipeline** | **0.231** | **[0.219, 0.244]** |

- **40.3% reduction** in estimation bias compared to baseline
- **150x faster** execution through efficient dimensionality reduction
- **Improved precision** with narrower confidence intervals

## 📁 Repository Structure

The main code to reproduce the paper results can be found in the numbered notebooks

```
causal-pipeline/
├── config/
├── data/
├── figs/
├── modules/
├── utils/
├── 01_data_generation.ipynb     # Synthetic dataset creation
├── 02_dimensionality_reduction.ipynb
├── 03_causal_identification.ipynb
├── 04_positivity_handling.ipynb
├── 05_effect_estimation.ipynb
├── 06_evaluation_refutation.ipynb
└── 07_ablation_study_identification.ipynb
└── 08_ablation_study_estimates.ipynb
```


### Custom Configuration

The specification of the method’s configuration, including correlation thresholds for dimensionality reduction, algorithmic choices for treatment-predictive feature derivation, and
parameters for positivity violation detection, can be found in the notebooks.

## 🧪 Running Experiments

### Reproduce Paper Results

```bash
# Run complete experimental results
python 01_data_generation.ipynb
python 02_dimensionality_reduction.ipynb
python 03_causal_identification.ipynb
python 04_positivity_handling.ipynb
python 05_effect_estimation.ipynb
python 06_evaluation_refutation.ipynb
python 07_ablation_study_identification.ipynb
python 08_ablation_study_estimates.ipynb
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use any of the contributions of this paper and this code in your research, please cite:

```bibtex
@inproceedings{causal-pipeline-2025,
  title={An end-to-end pipeline for Causal ML with continuous treatments: An application to financial decision making},
  author={Moral Hernández, Javier and Higuera-Cabañes, Clara and Ibraín, Álvaro},
  booktitle={KDD2025},
  series={KDD '25},
  year={2025},
  publisher={ACM},
  address={Madrid, Spain},
  url={https://github.com/javiermoralh/causal-pipeline}
}
```

## 👥 Authors

- **Javier Moral Hernández** - BBVA AI Factory - javier.moral.hernandez@bbva.com
- **Clara Higuera-Cabañes** - BBVA AI Factory - clara.higuera@bbva.com  
- **Álvaro Ibraín** - BBVA AI Factory - alvaro.ibrain@bbva.com


**For questions, issues, or collaboration opportunities, please open an issue or contact the authors directly.**