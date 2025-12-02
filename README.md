# Bee Flight Path Reconstruction with Probabilistic Regression

This project reconstructs the 2D flight path of bumblebees from noisy directional sensor readings using probabilistic modelling and regularised linear regression with Gaussian basis functions. It was originally developed as part of the **Machine Learning and Adaptive Intelligence** module at the **University of Sheffield**. 

## Problem Overview

A bee flies in a 2D field while four fixed detectors periodically measure only the **bearing** (direction) of the bee, not its distance. For each flight path we have:

- **truepath**: 100 time-stamped ground-truth positions over 30 seconds  
- **observations**: a set of rows `[time, detector_x, detector_y, bearing_x, bearing_y]` describing each detector reading 
The goal is to **infer the bee’s continuous trajectory** from these partial, noisy observations.

---

## Methodology

### 1. Observation Likelihood

For a predicted bee position **p** and an observation **ob**, the model:

1. Computes the **unit vector** from the detector to the predicted position.
2. Compares it to the observed bearing vector.
3. Forms an **error vector** as the difference of these two unit vectors.
4. Defines the **unnormalised negative log-likelihood** as:

L(ob | p) = (‖u − b‖²) / (2 * sigma²)

where:
- u is the predicted unit direction,
- b is the observed bearing, and
- sigma is a noise scale hyperparameter.

This is derived from a Gaussian error model over the direction mismatch.

This is implemented in:
- negloglikelihood(ob, p, noise_scale=0.1)


### 2. Trajectory Model: Gaussian Basis Regression

The bee’s x(t) and y(t) coordinates are modelled independently with a **linear combination of Gaussian basis functions**:

f(t) = Σ ( w_b * exp( − (t − c_b)² / (2 * alpha²) ) )

- Basis centres c_b ∈ {−3, 1, 5, …, 29, 33} seconds  
- Shared width parameter alpha (e.g. 3 s)  
- Separate weight vectors for x and y coordinates

Implemented via:

- `getpred(T, w, width=3)` → predicts positions at times `T` for a given weight vector `w`.

---

### 3. Total Objective: Regularised Negative Log-Likelihood

For a weight vector **w**, observations `obs` and predicted path `predpath`, the **total cost** is:

J(w) = Σ NLL(ob_i | p_i, sigma²)  +  lambda * Σ (w_b²)

- First term: sums the directional negative log-likelihood over all observations.  
- Second term: **L2 regularisation** (weight decay) to penalise overly large weights and improve generalisation.

Implemented via:

- `totalnegloglikelihood(w, obs, reg=0.001, noise_scale=0.1)`


### 4. Parameter Optimisation

- Initial weights are sampled from a Gaussian distribution.
- The cost function is minimised (e.g. via gradient-based optimisation or scipy’s optimisers) to obtain a **MAP estimate** of the trajectory parameters under the Gaussian prior induced by the L2 term.  

The learned weights define smooth trajectories for both x(t) and y(t), from which the 2D bee path is reconstructed.

---

## Analysis & Visualisation

The notebook includes:

- Plot of the true flight path for a selected bee, with the starting point highlighted.
- Illustration of the geometric relationship between detector location, bearing direction and a predicted bee position.
- Reconstruction of the bee’s trajectory using Gaussian basis regression, and a comparison of the reconstructed path against the ground-truth trajectory.
