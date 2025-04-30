# Supplementary Materials: "Statistical Analysis of USD/IDR Exchange Rate Response to the 2025 US Presidential Transition"

[![DOI](https://zenodo.org/badge/975096371.svg)](https://doi.org/10.5281/zenodo.15306342)

## Citation

```bibtex
@article{kaban2025analysis,
  title={{Statistical Analysis of USD/IDR Exchange Rate Response to the 2025 US Presidential Transition}},
  author={Kaban, S. N. and Nugraha, C. and Herho, S. H. S.},
  journal={xxxx},
  volume={XX},
  number={X},
  pages={XXX--XXX},
  year={2025}
}
```

## Research Overview

This repository contains supplementary materials for our comprehensive statistical investigation of USD/IDR exchange rate behavior surrounding the January 20, 2025 US presidential inauguration. The study employs multiple robust non-parametric tests with bootstrap resampling to quantify the immediate and sustained effects of US political transition on the Indonesian currency.

## Research Focus and Hypotheses

Our investigation addresses critical questions regarding the impact of US presidential transition on emerging market currencies:

1. **H₁**: The USD/IDR exchange rate exhibits significant distributional shifts following US presidential inauguration
2. **H₂**: Post-inauguration exchange rate volatility differs significantly from pre-inauguration patterns
3. **H₃**: The transition period contains identifiable anomalous exchange rate behaviors not explained by typical market fluctuations

To examine these hypotheses, we analyze 100 days of daily exchange rate data before and after the January 20, 2025 inauguration, focusing on distributional characteristics, variance patterns, and anomaly detection.

## Methodological Framework

Our analysis employs a robust statistical framework designed specifically for financial time-series data:

- **Non-parametric Methods**: Utilized to avoid distributional assumptions often violated in financial data
- **Bootstrap Resampling**: 10,000 iterations to generate confidence intervals and ensure statistical robustness
- **Multi-test Consensus**: Triangulation of findings through multiple statistical approaches:
  - Brown-Forsythe test for variance homogeneity
  - Cliff's Delta for non-parametric effect size measurement
  - Kolmogorov-Smirnov test for distribution comparisons
  - Mann-Whitney U test for stochastic dominance assessment
  - Comprehensive normality assessment using multiple tests
  - Ensemble anomaly detection combining machine learning and statistical methods


## Technical Requirements

- Python 3.8+
- Dependencies: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, yfinance

