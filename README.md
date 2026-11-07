# Reweighing (Kamiran & Calders, 2012) on Adult, German Credit and COMPAS

Evaluation of the pre-processing bias-mitigation algorithm **Reweighing** with IBM's
[AI Fairness 360](https://github.com/Trusted-AI/AIF360) (`aif360.algorithms.preprocessing.Reweighing`).

> Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without
> discrimination. *Knowledge and Information Systems, 33*(1), 1–33.

## Run

```bash
pip install -r requirements.txt
jupyter notebook reweighing_pipeline.ipynb      # or: jupyter nbconvert --to notebook --execute reweighing_pipeline.ipynb
```

The first cell downloads the raw data (UCI Adult and German Credit, ProPublica COMPAS) into AIF360's
`data/raw` folder if they are not there yet. A full run takes a few minutes on a laptop.

## Settings

| Setting | Value |
|---|---|
| Sensitive attribute (privileged = 1) | Adult: sex (male) · German: age (> 25) · COMPAS: race (Caucasian) |
| Favourable label | Adult: income > 50K · German: good credit · COMPAS: no recidivism within 2 years |
| Train/test split | 70 / 30, stratified on label × sensitive group, 10 random splits (seeds 0–9) |
| Mitigation | `Reweighing` fitted on the training split only; weights passed as `sample_weight` |
| Classifiers | Logistic regression (standardised features, `max_iter=2000`); random forest (200 trees, `min_samples_leaf=5`) |
| Sensitive attribute as model feature | No (main run); yes in the extra section 7 |
| Fairness measures | DP_Diff (statistical parity difference), DIR (disparate impact ratio), consistency (k = 5 neighbours on standardised features) |
| Performance measures | Accuracy, balanced accuracy |

## Outputs

* `results_per_split.csv` – every metric for every dataset, classifier, method and split
* `results_summary.csv` – mean ± sd over the 10 splits
* `fairness_before_after.png` – original labels vs. baseline vs. reweighed models

Tested with Python 3.13, aif360 0.6.1, scikit-learn 1.9, pandas 3.0 (the notebook disables pandas'
new string dtype, which AIF360 0.6 does not support).
