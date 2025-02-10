# Causal Pipeline for Continuous Treatments

End-to-end pipeline for causal machine learning with continuous treatments, designed for financial decision-making applications. The pipeline addresses challenges in applying causal inference to real-world scenarios, including high dimensionality, positivity violations, and continuous treatment effects.

## Pipeline Steps

1. **Dimensionality Reduction**
   - Feature selection for treatment prediction
   - Confounder identification
   - Outcome predictor selection

2. **Causal Identification**
   - Algorithmic causal discovery
   - Domain knowledge integration
   - Final adjustment set selection

3. **Positivity Violation Correction**
   - Violation detection using GPS
   - Data augmentation strategies

4. **Effect Estimation**
   - Linear regression with interactions
   - S-learner implementation
   - Modified AIPTW for continuous treatments

5. **Validation Framework**
   - Placebo treatment tests
   - Random common cause tests
   - E-value sensitivity analysis
   - QINI curves

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{causal-pipeline-2025,
  title={An end-to-end pipeline for Causal ML with continuous treatments: An application to financial decision making},
  author={Anonymous Author(s)},
  booktitle={KDD2025},
  year={2025}
}