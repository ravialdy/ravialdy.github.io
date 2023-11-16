---
layout: post
title: Paper Review "Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting"
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

<style>
h2 {
    margin-top: 1.25em; /* Increased margin-top */
    margin-bottom: 0.5em;
}
h3 {
    margin-top: 1.0em; /* Added margin-top for h3 */
    margin-bottom: 0.5em;
}
</style>

## Introduction

**Disclaimer** : This review is based on my understanding of the reference paper [1]. While I have made much effort to ensure the accuracy of this article, there may things that I have not fully captured. If you notice any misinterpretation or error, please feel free to point them out in the comments section.

I'm very excited to present a review of the paper titled "Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting" [1] authored by Dooseop Choi and KyoungWook Min. This paper is a very good work proved by its acceptance at the European Conference on Computer Vision (ECCV) 2022. 

For you who are not familiar with academia world in the AI field yet, ECCV is one of the most prestigious conferences in the domain of computer vision and having a paper accepted there indicates the importance of this work. I truly believe this research paper is crucial for the autonomous driving topic, particularly in trajectory forecasting.

<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/HLS.gif" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 1. Illustration of how the proposed Hierarchical Latent Structure (HLS) is used in the trajectory forecasting (Image source : D. Choi & K. Min [1]).
</div>

## Notations and Definitions

| Notation                             | Definition |
|--------------------------------------|------------|
| $$ N $$                              | Number of vehicles in the traffic scene |
| $$ T $$                              | Total number of timesteps for which trajectories are forecasted |
| $$ H $$                              | Number of previous timesteps considered for positional history |
| $$ V_{i} $$                          | The $$ i^{th} $$ vehicle in the traffic scene |
| $$ \mathbf{Y}_{i} $$                 | Future positions of $$ V_{i} $$ for the next $$ T $$ timesteps |
| $$ \mathbf{X}_{i} $$                 | Positional history of $$ V_{i} $$ for the previous $$ H $$ timesteps at time $$ t $$ |
| $$ \mathcal{C}_{i} $$                | Additional scene information available to $$ V_{i} $$ |
| $$ \mathbf{L}^{(1: M)} $$            | Lane candidates available for $$ V_{i} $$ at time $$ t $$ |
| $$ \mathbf{z}_{l} $$                 | Low-level latent variable used to model the modes |
| $$ \mathbf{z}_{h} $$                 | High-level latent variable used to model the weights for the modes |
| $$ p_{\theta} $$                     | Decoder network |
| $$ p_{\gamma} $$                     | Prior network |
| $$ \mathcal{L}_{E L B O} $$          | Modified ELBO objective |
| $$ q_{\phi} $$                       | Approximated posterior network |
| $$ f_{\varphi} $$                    | Proposed mode selection network |
| VLI                                  | Vehicle-Lane Interaction |
| V2I                                  | Vehicle-to-Vehicle Interaction |


## The Main Problem : "Mode Blur"

The paper aims to overcome a specific limitation in vehicle trajectory forecasting models that leverage Variational Autoencoders (VAEs) concept called as the "mode blur" problem. For clearer illustration, please take a look at the figure below (this corresponds to the figure 1 in the reference paper [1]) :

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4 img-container">
        {% include figure.html path="/assets/img/HLS_Paper/figure1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 2. Illustration of the "mode blur" problem in VAE-based generated trajectory forecasts (Image source : D. Choi & K. Min [1]).
</div>

As you can see from the figure above, the red vehicle is attempting to forecast its future trajectory represented by the branching gray paths. The challenge faced here lies in the generated forecast trajectories' that are sometimes between defined lane paths.

This phenomenon is what the author mean by the "mode blur" problem.  Specifically, the VAE-based model is not committing to a specific path, but rather giving a "blurred" average of possible outcomes.

<div class="row mt-4">
    <div class="col-12 col-lg mt-4 img-container">
        {% include figure.html path="/assets/img/HLS_Paper/modeblur-previousSOTA.png" class="img-fluid" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 3. Example of "mode blur" problem that exists in the previous SOTA model (Image source: Cui et al, 2021 [2]).
</div>

If you still wonder why the "mode blur" problem can be very important, consider the above figure example taken from the previous SOTA model as observed by D. Choi & K. Min [1]. Before analyzing that figure in more detail, assume that the green bounding box represents the Autonomous Vehicle (AV), the light blue bounding boxes represent surrounding vehicles, and the trajectories (path predictions) of the surrounding vehicles are shown using the solid lines with light blue dots.

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4">
        {% include figure.html path="/assets/img/HLS_Paper/scenario2_ModeBlur.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 4. Scenario 2 of the "mode blur" problem that exist in the previous SOTA model (Image source : Cui et al, 2021 [2]).
</div>

In scenario 2, a clear observation here is the overlapping and intersecting trajectories, especially around the intersection. These trajectories seem to be "blurred" between the lanes rather than being clearly defined in one lane or another. While in the scenario 3, despite the clearer trajectory forecasts than the previous one, we can still observe "mode blur" problems. Some predicted trajectories seem to be dispersed across the lane without a distinct path. 

This issue can lead to the Autonomous Vehicle (AV) having to make frequent adjustments to its path. This is indeed problematic as the AV might need to execute sudden brakes and make abrupt steering changes. This not only results in an uncomfortable ride for the passengers but also raises safety concerns.

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4">
        {% include figure.html path="/assets/img/HLS_Paper/scenario3_ModeBlur.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 5. Scenario 3 of the "mode blur" problem that exist in the previous SOTA model (Image source : Cui et al, 2021 [2]).
</div>

The reason for this problem is the use of Variational Autoencoders (VAEs) in the trajectory forecasting models since they have a well-known limitation: the outputs that they generate can often be "blurry". The authors of paper [1] observed that similar problem also found in the trajectory planning case, not only in the tasks involving image reconstruction and synthesis. 

VAEs aim to learn a probabilistic latent space representation of the data. When dealing with complex distributions such as future vehicle trajectories, the latent space needs to capture the multi-modal nature of the data, representing different possible future states (modes). Recall that the main objective of the VAEs is to optimize the Evidence Lower Bound Objective (ELBO) on the marginal likelihood of data $$ p_\theta(\mathbf{x}) $$. This lower bound is formulated as:

$$ 
\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z})) 
$$

Two components in the ELBO:
   - The first term $$ \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] $$ is the reconstruction loss which measures how well the VAE reconstructs the original data when sampled from the approximate posterior $$ q_\phi $$.
   - The second term $$ D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z})) $$ is the Kullback-Leibler divergence between the approximate posterior $$ q_\phi $$ and the prior $$ p_\theta $$. This term acts as a regularizer, pushing the approximate posterior towards the prior.

<div class="row mt-4">
    <div class="col-12 col-lg mt-4 img-container">
        {% include figure.html path="/assets/img/HLS_Paper/VAE_Image.png" class="img-fluid" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 6. Variational Autoencoder (VAE) which uses variational bayesian principle (Image source : <a href="https://sebastianraschka.com/teaching/stat453-ss2021/">Sebastian Raschka slide</a>).
</div>

<div class="row mt-4">
    <div class="col-12 col-lg mt-4 img-container">
        {% include figure.html path="/assets/img/HLS_Paper/AEvsVAE_Latent.png" class="img-fluid" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 7. Generated latent variable from common Autoencoder (fixed value) vs. VAE (probability distribution) (Image source : <a href="https://www.jeremyjordan.me/variational-autoencoders/">Jeremy Jordan</a>).
</div>

For more detailed understanding, you can take a look at this very good blogpost [Lil'Log](https://lilianweng.github.io/posts/2018-08-12-vae/) or excellent explanation by [Ahlad Kumar](https://www.youtube.com/watch?v=YHldNC1SZVk).

As you can see from the objective function above, the VAE wants to minimize reconstruction loss, while the KL divergence term encourages the VAE not to create very distinct and separate clusters for each mode in the latent space but to keep them close to the prior.

As far as i know, many previous works assume the prior distribution for the latent variables, $$ Z $$, to be a standard Gaussian distribution, $$ \mathcal{N}(0, I) $$, which is fixed and does not depend on the input context. The reason for using this assumption is to simplify the learning process.

This can be problematic because a standard Gaussian prior assumes that the latent space is unimodal and therefore does not capture the multi-modal nature of the future trajectories where multiple distinct future paths (modes) are possible.

When the VAE learns to represent data in the latent space, it must balance the reconstruction and KL divergence terms. It wants to spread out the representations to minimize the reconstruction loss (since the trajectory distribution is multi-modal) but it is also constrained by the KL divergence to keep these representations from getting too dispersed (since the prior is unimodal).

As a consequence, during the generation phase, when the model samples from these latent representations, it also may end up sampling from "in-between" spaces if the distinct modes are not well-separated. This results in outputs that are a blend of several possible outcomes rather than committing to a single, distinct outcome.

So in the context of trajectory planning, the "mode blur" problem is most likely happened due to the balancing act between reconstruction loss and the KL divergence done by the ELBO objective function. When generating data, the VAE may generate a predicted trajectory that doesn't clearly commit to any of the possible paths (like staying in the lane, changing lanes, turning, etc). Instead, it generates a trajectory that lies somewhere in between.


## Key Contributions

Based on my understanding so far, there are 4 major contributions of this paper [1]:

1. **Mitigating Mode Blur**: Propose a hierarchical latent structure within a VAE-based forecasting model to avoid "mode blur" problem, enabling clearer and more precise trajectory predictions.
  
2. **Context Vectors**: Two lane-level context vectors `VLI` and `V2I` are conditioned on the low-level latent variables for more accurate trajectory predictions.

3. **Additional Methods**: Introduce positional data preprocessing and GAN-based regularization to further enhance the performance.

4. **Benchmark Performance**: The state-of-the-art performance on two large-scale real-world datasets.


<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/VLI-visualization.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 8. Visualization of VLI (Image source : D. Choi & K. Min [1]).
</div>


## Hierarchical Latent Structure (HLS)

<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/HLS-Avoid-ModeBlur_Example-fotor-20231104133653.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 9. Example of how HLS avoids "mode blur" problem (Image source : D. Choi & K. Min [1]).
</div>

### Introduction to HLS

In the complex traffic scenes with `N` vehicles, predicting the future trajectory of each vehicle can be challenging. The Hierarchical Latent Structure (HLS) proposed by D. Choi & K. Min [1] aims to generate plausible trajectory distributions, taking into consideration both individual vehicle history and the overall scene.

You may wonder how that kind of approach can avoid the "mode blur" problem that happens in the previous work. The goal of the proposed method is to generate a trajectory distribution $$p\left(\mathbf{Y}_{i} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)$$ for vehicles. This distribution is supposed to predict the future positions $$\mathbf{Y}_{i}$$ based on the past positional history $$\mathbf{X}_{i}$$ and the scene context $$\mathcal{C}_{i}$$.

The generated trajectory distribution is represented as a sum of modes, weighted by their probability or importance. Mathematically, it can be defined like below :

$$
p\left(\mathbf{Y}_{i} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)=\sum_{m=1}^{M} \underbrace{p\left(\mathbf{Y}_{i} \mid E_{m}, \mathbf{X}_{i}, \mathcal{C}_{i}\right)}_{\text {mode }} \underbrace{p\left(E_{m} \mid \mathbf{X}_{i}, \mathcal{C}_{i}\right)}_{\text {weight }}
$$

The equation above indicates that the trajectory distribution $$ p(\mathbf{Y}_{i} \mid \mathbf{X}_{i}, \mathcal{C}_{i}) $$ can be expressed as a weighted sum of distributions called modes. The term "mode" represents a plausible path, and the term "weight" represents the probability of each mode occurring. 

Remember that in a standard VAE, the generation process can sometimes collapse to the mean, resulting in less diverse samples. However, in a hierarchical Conditional VAE (C-VAE) which we will discuss in more detail later, the low-level latent variable will be used to generate multiple potential trajectories and the high-level latent variable can be leveraged to assign probabilities to the generated trajectories given a certain condition (like the current state of the car and its environment). This doesn’t mean picking the "average" path but selecting from a distribution of paths where each path is weighted according to its fit to the current context.

### HLS to Avoid "Mode Blur"

The paper assumes that the trajectory distribution can be approximated as a mixture of simpler distributions. Each of these simpler distributions, or "modes", represents a distinct pattern or type of trajectory that a vehicle could follow.

The key intuition here is to consider each possible trajectory (mode) separately. It does this through a model that uses latent variables to represent different modes. By modeling each mode with a latent variable, the model can sample trajectories from these modes based on their weights or importance. This allows for diverse trajectory predictions rather than a blurred average.

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4 img-container">
        {% include figure.html path="/assets/img/HLS_Paper/figure1b_mode-separately.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 10. Illustration of the trajectory forecasting distribution generated by HLS model (Image source : D. Choi & K. Min [1]).
</div>

To capture this mixture of distributions, the HLS model employs two levels of latent variables:

1. **Low-level latent variable ($$\mathbf{z}_{l}$$)**: Used to model individual modes of the trajectory distributions. This is done through a decoder network that generates the vehicle's future positions and a prior network that defines the distribution of the latent variable given the past positions and scene context.

2. **High-level latent variable ($$\mathbf{z}_{h}$$)**: Represents the weights for different modes. This is the output of a mode selection network that determines the probabilities associated with different lanes.

Note that $$\mathbf{z}_{l}$$ is conditioned on both the vehicle's past trajectory and the contextual information, allowing the model to generate diverse and realistic trajectories within each mode. This aligns with the 'simple distributions' aspect of the assumption, as each latent variable models a distinct, simpler trajectory pattern.

The mathematical equation of the new objective function can be expressed as below :

$$
\begin{aligned}
\mathcal{L}_{ELBO} = & -\mathbb{E}_{\mathbf{z}_{l} \sim q_{\phi}}\left[\log p_{\theta}\left(\mathbf{Y}_{i} \mid \mathbf{z}_{l}, \mathbf{X}_{i}, \mathcal{C}_{i}^{m}\right)\right] \\
& + \beta KL\Big(q_{\phi}\left(\mathbf{z}_{l} \mid \mathbf{Y}_{i}, \mathbf{X}_{i}, \mathcal{C}_{i}^{m}\right)\| p_{\gamma}\left(\mathbf{z}_{l} \mid \mathbf{X}_{i}, \mathcal{C}_{i}^{m}\right)\Big),
\end{aligned}
$$

Notice that the modified ELBO objective function has two main components:

- **Reconstruction Term**: The $$ -\mathbb{E}_{\mathbf{z}_l \sim q_{\phi}}[\log p_{\theta}(\mathbf{Y}_i \mid \mathbf{z}_l, \mathbf{X}_i, \mathcal{C}_i^m)] $$ term is responsible for reconstructing the future trajectories ($$ \mathbf{Y}_i $$) from the low-level latent variables ($$ \mathbf{z}_l $$), the input ($$ \mathbf{X}_i $$), and the context ($$ \mathcal{C}_i^m $$). This term ensures that the model can generate trajectories that are consistent with the observed data and the given context.

- **Regularization Term**: The $$ +\beta KL $$ term regularizes the latent space by encouraging the distribution of the latent variables to be close to some prior distribution. This is essential to avoid overfitting and to ensure that the latent space is well-structured and interpretable.

The key aspect here is that the prior $$ p_{\gamma}(\mathbf{z}_l \mid \mathbf{X}_{i}, \mathcal{C}_{i}^{m}) $$ is conditional on the input and the context. This implies that the model isn't learning a single static prior for all data but rather a dynamic prior that adapts based on the specific input $$ \mathbf{X}_i $$ and context $$ \mathcal{C}_{i}^{m} $$.

This conditionality allows the model to learn different representations for different subsets of data, guided by the vehicle's past trajectory and additional scene information relevant to the vehicle. By doing so, the model can capture the nuances and variations in trajectory distributions that are specific to different traffic situations and lane configurations. 

By structuring the model in this way, the HLS method can generate trajectory predictions that are not just an average of all possible paths but are instead a combination of distinct, plausible paths that a vehicle might realistically take, each with its own probability. This approach effectively addresses the mode blur problem by maintaining the distinctness of each trajectory mode.

### HLS Overall Architecture

<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/HLS_Architecture-fotor-20231104133313.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 11. Diagram of HLS architecture (Image source : D. Choi & K. Min [1]).
</div>

The proposed method from the paper focuses on predicting the future trajectory of vehicles by considering the interaction with the surrounding environment, particularly lanes and other vehicles. The approach is organized into the following modules:

1. **Feature Extraction Module**: 
   - This module uses three LSTM networks to encode positional data for vehicles and lanes.
   - Data preprocessing involves calculating speed, heading for vehicles, and tangent vectors for lanes. This is to capture the motion history and lane's orientation, making predictions more accurate.


2. **Scene Context Extraction Module**:
   - It considers the interactions of a vehicle with its reference lane `VLI` and other surrounding vehicles `V2I`.
   - For the lane interaction, it uses attention mechanisms to weigh the importance of surrounding lanes relative to the reference lane.
   - For vehicle-to-vehicle interactions, a Graph Neural Network (GNN) is employed. Only vehicles within a certain distance from the reference lane are considered. The interactions are captured through multiple rounds of message passing, and the final context vector represents the interaction history.
   - There's an emphasis on the distance threshold, which is empirically set to 5 meters, representing the typical distance between two nearby lane centerlines in straight roads.


3. **Mode Selection Network**:
   - Determines the weights for different modes of trajectory distribution. Each mode corresponds to a lane, capturing the assumption that the lanes heavily influence the vehicle's motion.
   - It uses lane-level scene context vectors, which contain information about both lane and vehicle interactions.
   - A softmax operation is applied to get the final weights, representing the probability distribution over different modes.


4. **Encoder, Prior, and Decoder**:
   - **Encoder**: This is often referred to as the recognition network. It is responsible for approximating the posterior distribution and is implemented as Multi-Layer Perceptrons (MLPs) with the encoding of the future trajectory $$\tilde{\mathbf{Y}}_{i}$$ and the lane-level scene context vector $$\mathbf{c}_{i}^{m}$$ as inputs. The encoder outputs two vectors, mean $$\mu_{e}$$ and standard deviation $$\sigma_{e}$$. Notably, the encoder is used only during the training phase because the future trajectory $$\mathbf{Y}_{i}$$ is not available during inference.

   - **Prior**: This represents the prior distribution over the latent variable and is also implemented as MLPs. It takes the lane-level scene context vector $$\mathbf{c}_{i}^{m}$$ as its input and outputs mean $$\mu_{p}$$ and standard deviation $$\sigma_{p}$$ vectors.

   - **Decoder**: This network is responsible for generating predictions for the future trajectory $$\hat{\mathbf{Y}}_{i}$$. It does so via an LSTM network. The input to the LSTM consists of an embedding of the predicted position $$\mathbf{e}_{i}^{t}$$ along with the lane-level scene context vector $$\mathbf{c}_{i}^{m}$$ and the latent variable $$\mathbf{z}_{l}$$. The LSTM updates its hidden state $$\mathbf{h}_{i}^{t+1}$$ based on these inputs, and the new predicted position $$\hat{\mathbf{p}}_{i}^{t+1}$$ is generated from this hidden state.

The design of this method aims to provide a holistic understanding of the vehicle's motion by considering both lane and vehicle interactions. Lanes guide the general direction of movement, while nearby vehicles influence more immediate decisions like lane changes or speed adjustments.

## Conclusion

<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.html path="/assets/img/HLS_Paper/Example_HLS_nuScene.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 12. Example of trajectory forecasting generated by HLS on nuScenes dataset (Image source : D. Choi & K. Min [1]).
</div>

This paper proposes a novel and unique way to tackle the problem of "mode blur" predictions in trajectory forecasting. Instead of just mixing all possible paths, it uses a system of weights to represent different possible futures. This is achieved by introducing a hierarchy in latent variables which can make the model to be more accurate in representing different possible outcomes. The use of lane-level context vectors can add more precision, especially in understanding vehicle-lane and vehicle-vehicle interactions. With the additional techniques like positional data processing and GAN-based regularization, this work not only sharpens the predictions but also can outperform the previous SOTA models in terms of accuracy.