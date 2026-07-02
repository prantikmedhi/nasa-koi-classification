# Classifying Kepler Objects of Interest: Confirmed Planets, Candidates, and False Positives

## Problem

The Kepler space telescope flagged thousands of periodic dimming events as possible planets. Each Kepler Object of Interest (KOI) eventually gets dispositioned as CONFIRMED (a real planet), FALSE POSITIVE (usually an eclipsing binary star or an instrumental artifact), or stays a CANDIDATE. This project builds a classifier that predicts the disposition from the NASA Exoplanet Archive cumulative KOI table (9,564 objects, 140 catalog columns) and explains, in physical terms, what drives its decisions.

## Data handling and EDA

The extract deliberately excludes `koi_score`. That column is a direct output of the vetting process that produced the label, so using it would be target leakage, and we made no attempt to reconstruct it. EDA surfaced three structural issues.

Nineteen columns are entirely null (eccentricity errors, stellar age, model fit statistics, and similar). They carry no information and were dropped.

Identifier and leakage-prone columns were removed from the feature set: `kepid` and `kepoi_name` are IDs, `koi_pdisposition` is a near duplicate of the target, and `kepler_name` is only assigned to confirmed planets, so even its missingness encodes the answer.

KOI uncertainties come in symmetric ± pairs. A correlation check confirmed that most `_err2` columns are perfectly anti-correlated with their `_err1` twin, so the mirrors were pruned.

Remaining missingness (at most 16% per column, concentrated in secondary vetting statistics) was median-imputed, with medians fit on the training split only. Most physical quantities here are heavily right-skewed, and a mean would be dominated by the tails. The few low-cardinality categorical columns (fit type, catalog provenance) were one-hot encoded with "missing" as an explicit level.

Class balance is 51% FALSE POSITIVE, 29% CONFIRMED, 21% CANDIDATE, so we optimized macro F1 rather than accuracy and used balanced class weighting in every model. We preferred weighting to synthetic oversampling (SMOTE) because interpolating in a feature space of heavy-tailed physical quantities can fabricate physically impossible objects. Weighting keeps every real observation.

## Feature engineering

Eight domain-derived features were added, each tied to a piece of vetting physics. Transit depth per hour separates stellar eclipses, which are deep and sharp, from planetary transits, which are shallow and gradual. The planet-to-star radius ratio catches "planets" comparable in size to their host star, which are in fact stars. The ratio of the multiple-event to the single-event statistic captures whether evidence accumulates over repeated orbits, as it does for real planets, or arrives once, as it does for glitches. Comparing the observed transit duration against the value expected from Kepler's third law flags grazing eclipses. Log transforms of period, depth, and insolation handle quantities that vary over orders of magnitude. Finally, SNR per transit separates intrinsically strong signals from ones that merely got observed many times.

## Models and results

Three models of increasing capacity were compared with stratified 5-fold cross-validation on an 80/20 stratified split, with the random seed fixed at 42 throughout:

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.916 | 0.888 |
| Random Forest | 0.928 | 0.903 |
| XGBoost | 0.942 | 0.922 |

A 20-configuration randomized hyperparameter search on XGBoost moved cross-validation macro F1 only slightly (0.930 baseline vs. 0.928 tuned) and did not improve the held-out test score. The defaults were already near optimal, and we report that as is rather than cherry-picking a run. Per-class F1 for the best model: FALSE POSITIVE 0.992, CONFIRMED 0.913, CANDIDATE 0.863. Nearly all the residual error in the confusion matrix sits on the boundary between CONFIRMED and CANDIDATE, which makes sense: a candidate is, by definition, a planet-like signal that has not yet cleared the confirmation bar, so the two classes genuinely overlap.

One ablation deserves its own paragraph. The four `koi_fpflag_*` columns are outputs of Kepler's automated vetting pipeline. They are legitimate features, machine generated and available at prediction time, but they encode much of the FALSE POSITIVE decision, so we measured the dependence: removing them drops macro F1 from 0.922 to 0.830. The model stays reasonably strong on transit-fit and stellar features alone. We report the with-flags model as primary and keep this ablation as the honesty check.

## What the model learned

SHAP analysis ranks the vetting flags first. The stellar-eclipse, not-transit-like, and centroid-offset flags are near-decisive evidence against a planet. After the flags come genuinely physical features: the multiple-event statistic (is the signal periodic and cumulative?), the centroid offset `koi_dicco_msky` (is the dimming actually happening on the target star, or on a neighbor blended into the same pixel?), transit signal-to-noise (confirmation requires a clean signal, and low SNR is a big part of why candidates stay candidates), and planet radius (objects larger than about 2 Jupiter radii are almost always small stars). This decision logic mirrors how astronomers vet KOIs by hand: check the geometry, check the repeatability, check the signal quality. That is decent evidence the model learned physics rather than dataset quirks.

## Reproducibility

Everything runs top-to-bottom from two notebooks (`01_eda.ipynb`, `02_modeling.ipynb`) backed by four documented modules in `src/`, with fixed random seeds and pinned dependency ranges in `requirements.txt`.
