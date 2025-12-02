# Bee Flight Path Reconstruction with Probabilistic Regression

This project reconstructs the 2D flight path of bumblebees from noisy directional sensor readings using probabilistic modelling and regularised linear regression with Gaussian basis functions. It was originally developed as part of the **COM4509/6509 – Machine Learning and Adaptive Intelligence** module at the **University of Sheffield**. :contentReference[oaicite:3]{index=3}

## Problem Overview

A bee flies in a 2D field while four fixed detectors periodically measure only the **bearing** (direction) of the bee, not its distance. For each flight path we have:

- **truepath**: 100 time-stamped ground-truth positions over 30 seconds  
- **observations**: a set of rows `[time, detector_x, detector_y, bearing_x, bearing_y]` describing each detector reading :contentReference[oaicite:4]{index=4}  

The goal is to **infer the bee’s continuous trajectory** from these partial, noisy observations.

---

## Methodology

### 1. Observation Likelihood

For a predicted bee position **p** and an observation **ob**, the model:

1. Computes the **unit vector** from the detector to the predicted position.
2. Compares it to the observed bearing vector.
3. Forms an **error vector** as the difference of these two unit vectors.
4. Defines the **unnormalised negative log-likelihood** as:

\[
\mathcal{L}(\text{ob} \mid \mathbf{p}) = \frac{\lVert \mathbf{u} - \mathbf{b} \rVert^2}{2\sigma^2}
\]

where \(\mathbf{u}\) is the predicted unit direction, \(\mathbf{b}\) is the observed bearing and \(\sigma\) is a noise scale hyperparameter. This is derived from a Gaussian error model over the direction mismatch. :contentReference[oaicite:5]{index=5}  

This is implemented in:

- `negloglikelihood(ob, p, noise_scale=0.1)`

### 2. Trajectory Model: Gaussian Basis Regression

The bee’s x(t) and y(t) coordinates are modelled independently with a **linear combination of Gaussian basis functions**:

\[
f(t) = \sum_{b=1}^{B} w_b \exp\left(-\frac{(t - c_b)^2}{2\alpha^2}\right)
\]

- Basis centres \(c_b \in \{-3, 1, 5, …, 29, 33\}\) seconds  
- Shared width parameter \(\alpha\) (e.g. 3 s)  
- Separate weight vectors for x and y coordinates :contentReference[oaicite:6]{index=6}  

Implemented via:

- `getpred(T, w, width=3)` → predicts positions at times `T` for a given weight vector `w`.

### 3. Total Objective: Regularised Negative Log-Likelihood

For a weight vector **w**, observations `obs` and predicted path `predpath`, the **total cost** is:

\[
\mathcal{J}(\mathbf{w}) =
\sum_{i=1}^{N} \text{NLL}(\text{ob}_i \mid \mathbf{p}_i, \sigma^2)
+ \lambda \sum_{b=1}^{B} w_b^2
\]

- First term: sums the directional negative log-likelihood over all observations.
- Second term: **L2 regularisation** (weight decay) to penalise overly large parameters and improve generalisation. :contentReference[oaicite:7]{index=7}  

Implemented via:

- `totalnegloglikelihood(w, obs, reg=0.001, noise_scale=0.1)`

### 4. Parameter Optimisation

- Initial weights are sampled from a Gaussian distribution.
- The cost function is minimised (e.g. via gradient-based optimisation or scipy’s optimisers) to obtain a **MAP estimate** of the trajectory parameters under the Gaussian prior induced by the L2 term. :contentReference[oaicite:8]{index=8}  

The learned weights define smooth trajectories for both x(t) and y(t), from which the 2D bee path is reconstructed.

---

## Analysis & Visualisation

The notebook includes:

- Plot of the **true flight path** for a selected bee, with the starting location highlighted.
- Visualisation of:
  - Detector locations
  - Bearing observations
  - Reconstructed trajectory vs ground truth
- Qualitative analysis of:
  - Posterior uncertainty at specific times (e.g. why the posterior is spread at \(t = 1.8s\))
  - Behaviour near the sequence boundaries (e.g. path curling back towards the centre due to weaker observation support)
  - Effect of Gaussian basis width and regularisation on smoothness and overfitting. :contentReference[oaicite:9]{index=9}  

---
