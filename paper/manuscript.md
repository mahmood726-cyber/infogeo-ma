# Information-Geometric Meta-Analysis: Geodesic Pooling and Curvature-Based Heterogeneity on Statistical Manifolds

**Mahmood Ahmad**^1

1. Royal Free Hospital, London, United Kingdom | mahmood.ahmad2@nhs.net | ORCID: 0009-0003-7781-4478

---

## Abstract

**Background:** Standard meta-analysis pools effects via Euclidean inverse-variance weighting, ignoring the natural geometry of probability distributions. On the statistical manifold — where each study is a point in the space of normal distributions — the appropriate distance is the Fisher-Rao metric, not the Euclidean norm. This distinction matters when studies differ in both effect size and precision.

**Methods:** InfoGeoMA is a browser tool implementing meta-analysis on the normal statistical manifold. Each study N(mu_i, sigma_i^2) is a point on a 2D Riemannian manifold equipped with the Fisher-Rao metric. The geodesic Fréchet mean — the point minimising the sum of squared geodesic distances — replaces the Euclidean inverse-variance mean. A novel kappa-heterogeneity metric measures non-Euclidean spread on the manifold, capturing geometric structure invisible to I-squared.

**Results:** Applied to three canonical datasets: BCG vaccine (k=13) showed kappa=0.73 (moderate curvature heterogeneity) with a geodesic shift of 0.59 from the DL estimate. The Teo magnesium dataset (k=8) showed extreme kappa (>10,000), revealing that ISIS-4 (SE=0.04) occupies a fundamentally different manifold region than the small trials (SE 0.8-1.4). SGLT2i heart failure (k=6) similarly showed high kappa, driven by the DAPA-CKD subgroup study's large SE.

**Conclusion:** Information-geometric meta-analysis reveals non-Euclidean structure in evidence that variance-based methods cannot detect. Available at https://github.com/mahmood726-cyber/infogeo-ma (MIT).

**Keywords:** information geometry, Fisher-Rao metric, statistical manifold, Fréchet mean, geodesic pooling, Riemannian heterogeneity

---

## 1. Introduction

The Fisher-Rao metric, introduced by C.R. Rao in 1945, defines the natural distance between probability distributions. For the family of normal distributions N(mu, sigma^2), the Fisher information matrix induces a Riemannian metric on the 2D (mu, sigma) parameter space, making it a curved manifold — the normal statistical manifold.

Standard meta-analysis treats each study as a point in Euclidean space (effect size on the real line) and pools via inverse-variance weighting. This ignores the second dimension (the standard error) and assumes Euclidean geometry. On the statistical manifold, the appropriate pooling is the Fréchet mean — the point minimising the sum of squared geodesic (Fisher-Rao) distances.

This distinction matters most when studies have widely varying precisions — precisely the setting where meta-analytic pooling is most sensitive to methodological choices.

## 2. Methods

### Fisher-Rao Distance
For two normal distributions N(mu_1, s_1^2) and N(mu_2, s_2^2), the Fisher-Rao distance (Atkinson & Mitchell, 1981) is:

d_FR = sqrt(2) * arccosh(1 + (mu_1-mu_2)^2/(2*s_1*s_2) + (s_1^2-s_2^2)^2/(4*s_1^2*s_2^2))

This distance accounts for both the effect size difference AND the precision difference simultaneously, in a geometrically natural way.

### Geodesic Fréchet Mean
The Fréchet mean theta* = argmin sum(w_i * d_FR(theta, theta_i)^2) is computed via gradient descent on the manifold, initialised at the Euclidean inverse-variance mean.

### Kappa-Heterogeneity
kappa = (mean geodesic distance from Fréchet mean) / (expected distance under homogeneity). Unlike I-squared, kappa captures the curvature-dependent spread of studies on the manifold.

## 3. Results

| Dataset | k | DL | Fréchet | Shift | I² | κ |
|---------|---|-----|---------|-------|-----|-----|
| BCG | 13 | -0.549 | 0.042 | 0.591 | 92% | 0.73 |
| Magnesium | 8 | -0.470 | -1.383 | 0.912 | 67% | 10,851 |
| SGLT2i | 6 | -0.227 | 0.307 | 0.534 | 28% | 2,394 |

The extreme kappa values for magnesium and SGLT2i reflect studies with vastly different SEs occupying distant regions of the manifold. I-squared, being a variance ratio, cannot distinguish "studies far apart on the manifold" from "studies close together but with high within-study noise."

## 4. Discussion

InfoGeoMA demonstrates that the choice of geometry matters for meta-analysis. The Fréchet mean can differ substantially from the DL mean, particularly when the precision range across studies spans orders of magnitude. The kappa-heterogeneity metric provides a complementary view of between-study variability that captures non-Euclidean structure invisible to I-squared.

The main limitation is computational: gradient descent on the manifold requires ~200 iterations (vs closed-form for Euclidean), though this runs in <100ms in the browser. The Fréchet mean is sensitive to the learning rate and may not converge to the global minimum for highly scattered datasets.

## References

1. Rao CR. Information and accuracy attainable in estimation. *Bull Calcutta Math Soc*. 1945;37:81-91.
2. Amari S. *Information Geometry and Its Applications*. Springer; 2016.
3. Atkinson C, Mitchell AFS. Rao's distance measure. *Sankhya A*. 1981;43:345-365.
4. Fréchet M. Les éléments aléatoires de nature quelconque dans un espace distancié. *Ann Inst H Poincaré*. 1948;10:215-310.
