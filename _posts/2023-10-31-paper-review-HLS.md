---
layout: post
title: Detailed Paper Review "Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting"
date: 2023-10-31 09:56:00-0400
description:
tags: Autonomous Driving, Trajectory Planning
categories: blogpost
giscus_comments: true
related_posts: false
related_publications: HLS, LookOut
toc:
  beginning: true
  sidebar: left  # or 'right'
---

## Introduction

Welcome to my in-depth review of the paper titled "Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting." I think this research paper is very important for the field of autonomous driving, especially in the trajectory forecasting.  Before we go into the paper's architecture, algorithms, and experimental results, let's first understand the challenges that the authors aim to address and the novel approaches they have introduced.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/HLS.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 1. Illustration of how the proposed Hierarchical Latent Structure (HLS) is used in the trajectory forecasting (Image source : D. Choi & K. Min [1]).
</div>

## Notations and Definitions

| Notation                             | Definition |
|--------------------------------------|------------|
| $$ N $$                              | Number of vehicles in the traffic scene |
| $$ \mathbf{Y}_{i} $$                 | Future positions of $$ V_{i} $$ for the next $$ T $$ timesteps |
| $$ \mathbf{X}_{i} $$                 | Positional history of $$ V_{i} $$ for the previous $$ H $$ timesteps at time $$ t $$ |
| $$ \mathcal{C}_{i} $$                | Additional scene information available to $$ V_{i} $$ |
| $$ \mathbf{L}^{(1: M)} $$            | Lane candidates available for $$ V_{i} $$ at time $$ t $$ |
| $$ \mathbf{L}^{m} $$                 | $$ F $$ equally spaced coordinate points on the centerline of the $$ m $$-th lane |
| $$ E_{m} $$                          | The event that $$ \mathbf{L}^{m} $$ becomes the reference lane for $$ V_{i} $$ |
| $$ \mathbf{z}_{l} $$                 | Low-level latent variable used to model the modes |
| $$ \mathbf{z}_{h} $$                 | High-level latent variable used to model the weights for the modes |
| $$ p_{\theta} $$                     | Decoder network |
| $$ p_{\gamma} $$                     | Prior network |
| $$ \mathcal{C}_{i}^{m} $$            | Scene information relevant to $$ \mathbf{L}^{m} $$ |
| $$ \mathcal{L}_{E L B O} $$          | Modified ELBO objective |
| $$ \beta $$                          | A constant |
| $$ q_{\phi} $$                       | Approximated posterior network |
| $$ f_{\varphi} $$                    | Proposed mode selection network |
| VLI                                  | Vehicle-lane interaction |
| V2I                                  | Vehicle-to-vehicle interaction |


## The Core Problem : "Mode Blur"

The paper aims to overcome a specific limitation in vehicle trajectory forecasting models that leverage Variational Autoencoders (VAEs) concept called as the "mode blur" problem. For clearer illustration, please take a look at the figure below (this corresponds to the figure 1 in the reference paper [1]) : 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/figure1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 2. Illustration of the "mode blur" problem in VAE-based generated trajectory forecasts (Image source : D. Choi & K. Min [1]).
</div>

As you can see from the figure above, the red vehicle is attempting to forecast its future trajectory represented by the branching gray paths. The challenge faced here lies in the generated forecast trajectories' that are more often on a "central" path, representing an average of all potential future paths rather than distinct possibilities. This phenomenon is what the author mean by the "mode blur" problem.  Specifically, the VAE-based model is not committing to a specific path, but rather giving a "blurred" average of possible outcomes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/modeblur-previousSOTA.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 3. Example of "mode blur" problem that exist in the previous SOTA model (Image source : Cui et al, 2021 [2]).
</div>

If you wonder why the "mode blur" problem can be very important, consider the above figure example taken from the previous SOTA model as observed by D. Choi & K. Min [1]. Before analyzing that figure in more detail, assume that the green bounding box represents the Autonomous Vehicle (AV), the light blue bounding boxes represent surrounding vehicles, and the trajectories (path predictions) of the surrounding vehicles are shown using the solid lines with light blue dots. 

In scenario 2, a clear observation here is the overlapping and intersecting trajectories, especially around the intersection. These trajectories seem to be "blurred" between the lanes rather than being clearly defined in one lane or another. While in the scenario 3, despite the more linear environment, we can still observe "mode blur" problems, especially with the trajectories of the vehicle immediately in front of the AV. The trajectories seem to be dispersed across the lane without a distinct path. This issue can lead to the Autonomous Vehicle (AV) having to make frequent adjustments to its path. This is indeed problematic as the AV might need to execute sudden brakes and make abrupt steering changes. This not only results in an uncomfortable ride for the passengers but also raises safety concerns.

The reason for this problem is the use of Variational Autoencoders (VAEs) in the trajectory forecasting models. Even though VAEs are theoretically elegant, simple to train, and can produce quite good manifold representations (meaning they can capture complex patterns and relationships in data), they have a well-known limitation: the outputs that they generate can often be "blurry", particularly in tasks involving image reconstruction and synthesis. This is the result of the VAE trying to generate an output that's an average representation of potential outcomes. Remember that the main objective of the VAE is to optimize the evidence lower bound (ELBO) on the marginal likelihood of data $$ p_\theta(\mathbf{x}) $$. This lower bound is formulated as:

$$ 
\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z})) 
$$

Two components in the ELBO:
   - The first term $$ \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] $$ is the reconstruction loss which measures how well the VAE reconstructs the original data when sampled from the approximate posterior $$ q_\phi $$.
   - The second term $$ D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z})) $$ is the Kullback-Leibler divergence between the approximate posterior $$ q_\phi $$ and the prior $$ p_\theta $$. This term acts as a regularizer, pushing the approximate posterior towards the prior.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/VAE-graphical-model.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 4. Variational Autoencoder (VAE) which uses variational bayesian principle ((Image source : <a href="https://lilianweng.github.io/posts/2018-08-12-vae">Weng's Blog</a>)).
</div>

For more detailed understanding, you can take a look at this very good blogpost [Lil'Log](https://lilianweng.github.io/posts/2018-08-12-vae/).

As you can see from the objective function above, the VAE wants to minimize reconstruction loss, while the KL divergence term encourages the VAE not to create very distinct and separate clusters for each mode in the latent space but to keep them close to the prior. This might cause different modes to be close in the latent space, instead of committing to a particular mode, which might results in generating something in-between, leading to blurry results in the case of the reference paper [1], generated trajectories that lie between adjacent lanes.

So in the context of trajectory planning, the "mode blur" problem is most likely happened due to the balance between reconstruction loss (how well the trajectory is reconstructed) and the KL divergence (pushing the trajectory's latent representation towards the prior). When generating data (like trajectories or images), VAE might be uncertain about which mode (or cluster) of the latent space a particular data point belongs to. 


## Key Contributions

Based on my understanding so far, there are 4 major contributions of this paper:

1. **Mitigating Mode Blur**: Propose a hierarchical latent structure within a VAE-based forecasting model to avoid "mode blur" problem, enabling clearer and more precise trajectory predictions.
  
2. **Context Vectors**: Two lane-level context vectors (Vehicle-Lane Interaction (VLI) and Vehichle-Vehicle Interaction (V2I)) are conditioned on the low-level latent variables for more accurate trajectory predictions.

3. **Additional Methods**: Introduce positional data preprocessing and GAN-based regularization to further enhance the performance.

4. **Benchmark Performance**: The state-of-the-art performance on two large-scale real-world datasets.


## Proposed Method

### Hierarchical Latent Structure (HLS)

#### Introduction to HLS

In complex traffic scenes with `N` vehicles, predicting the future trajectory of each vehicle can be challenging. The Hierarchical Latent Structure (HLS) proposed by D. Choi & K. Min [1] aims to generate plausible trajectory distributions, taking into consideration both individual vehicle history and the overall scene.

You may wonder how that kind of approach can avoid the "mode blur" problem that happens in the previous work. To answer that, we need to first recall that the mode blur problem arises when a model averages over possible future trajectories, resulting in a prediction that is a mix of multiple plausible outcomes.

This paper aims to generate a trajectory distribution $$p\left(\mathbf{Y}_{i} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)$$ for vehicles. This distribution is supposed to predict the future positions $$\mathbf{Y}_{i}$$ based on the past positional history $$\mathbf{X}_{i}$$ and the scene context $$\mathcal{C}_{i}$$.

The generated trajectory distribution is represented as a sum of modes, weighted by their probability or importance. Mathematically, it can be defined like below :

$$
p\left(\mathbf{Y}_{i} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)=\sum_{m=1}^{M} \underbrace{p\left(\mathbf{Y}_{i} \mid E_{m}, \mathbf{X}_{i}, \mathcal{C}_{i}\right)}_{\text {mode }} \underbrace{p\left(E_{m} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)}_{\text {weight }}
$$

As you can see from the mathematical equation above, it shows the trajectory distribution as a weighted sum of simpler distributions. Each mode represents a likely future trajectory.

#### HLS as a Method to Avoid "Mode Blur" Problem

The key intuition here is that instead of predicting a single trajectory that's an average of all possible futures, the proposed model considers each possible trajectory (mode) separately. By modeling each mode with a latent variable, the model can sample trajectories from these modes based on their weights or importance. This allows for diverse trajectory predictions rather than a blurred average.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/figure1b_mode-separately.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 5. Illustration of the trajectory forecasting distribution generated by HLS model (Image source : D. Choi & K. Min [1]).
</div>

The HLS approach consists of two latent variables, a low-level latent variable $$\mathbf{z}_{l}$$ and a high-level latent variable $$\mathbf{z}_{h}$$. The low-level latent variable, $$\mathbf{z}_{l}$$, helps the forecasting model define the mode distribution. The conditional VAE framework is employed, making the model generate realistic trajectories based on the provided past data and scene context. The high-level latent variable, $$\mathbf{z}_{h}$$, models the weights of the modes, determining which lane or trajectory is more probable.

The hierarchical latent structure allows the model to capture the different levels of variability in the data. The low-level captures the possible trajectories, and the high-level captures their probabilities. This layered approach ensures that the model doesn't simply average out all possibilities but respects the diversity and uncertainty inherent in predicting future trajectories.

The paper emphasizes the importance of scene context. The future motion of a vehicle is influenced both by its past motions and by the scene context, which includes surrounding vehicles and the geometry of the road. By incorporating this context into the model, the authors ensure that each predicted mode is feasible and respects the constraints and influences of the environment.

In short, the proposed approach addresses the mode blur problem by representing the trajectory distribution as a weighted sum of modes. Instead of blurring all possibilities together, it uses latent variables to capture the inherent variability and uncertainty in trajectory forecasting. This results in predictions that respect the diversity of potential futures while being influenced by past data and scene context.