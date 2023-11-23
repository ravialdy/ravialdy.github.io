---
layout: post
title: The Magic of Variational Inference
date: 2023-11-20 09:56:00-0400
description:
tags: Variational, Inference, Bayesian
categories: blogpost
giscus_comments: true
related_posts: false
related_publications: VarInfer
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

Hi guys, welcome to my blogpost again :) Today, I want to discuss about the magical and how wonderful variational inference is. This approach is widely used in many applications, such as text-to-image generation, motion planning, Reinforcement Learning (RL), etc. 

The reason for this is that in many cases the distribution of generated output that we want is very complex, e.g., images, text, video, etc. This is where VI can help us through latent variables. I believe that once we can master this concept well, then we can understand many recent AI techniques more easily and intuitively.

I often found the explanation on the internet about this topic is not clear enough in explaining the reasons why this concept needs to exist somehow, why we need to calculate many fancy math terms, etc. Therefore, in this post I also want to focus more on the reasoning part so that all of us can understand the essence of this method beyond the derivations and usefulness of VI.

This post is based on my understanding of the topic, so if you find any mistake please let me know :)

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4 img-container">
        {% include figure.html path="/assets/img/VI/DALL·E 2023-11-23 17.07.41.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 1. Illustration of the output from text-to-image model (Image source : DALLE-3).
</div>

## Latent Variable Models

You may ask what Variational Inference (VI) really is? How it can be oftenly used in many recent AI methods? To answer those questions, let me start with latent variable models.

Let's say we want to build a regression model that can fit a simple data distribution like this, 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/VI/regression.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 2. Regression model that tries to fit simple data distribution (Image source : <a href="https://www.analyticsvidhya.com/blog/2022/01/different-types-of-regression-models/">Analytics Vidhya</a>).
</div>

What we basically try to do from the image above is to model $$ p_(\mathbf{y} \mid \mathbf{x}) $$ where $$ y $$ is our data given $$ x $$. It seems very simple right? But now let's imagine we have quite complex data dsitribution like below,

<div class="row mt-4 justify-content-center">
    <div class="col-12 col-md-8 mx-auto mt-4 img-container">
        {% include figure.html path="/assets/img/VI/mixture_gaussian.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 3. The scenario where data distribution is complex enough (Image source : <a href="https://www.youtube.com/watch?v=iL1c1KmYPM0&t=2900s">Stanford Online</a>).
</div>

You might be confused initially on how we can build a model that fits that distribution. But don't worry, I was also used to be like that too :) In reality, the distribution that we face might be much more complex than that.

Fortunately, we can approximate that distribution through multiplication of two simple distributions. How we can do that? This is where the concept of latent variable models comes into play. 

The data distribution itself $$ p_(\mathbf{x}) $$ can be expressed mathematically as,

$$
p(\mathbf{x})=\sum_z p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})
$$

where $$ \mathbf{z} $$ is the latent variable. Maybe you ask, what is that thing? Basically it is the hidden value that is not the variable $$ y $$ nor $$ x $$, but needs to be considered where we want to calculate the probability of the observed data $$ \mathbf{x} $$ in a more complex distribution. These latent variables represent underlying factors or characteristics that might not be directly observable but significantly influence the observed data.

For example, the latent variables for figure 3 is the categorical value that maps each data point into cluster blue, green, or yellow. 

You may still wonder, how latent variable models is used in this case? First, we need to know that the prior or latent variable distribution $$ p(\mathbf{z} $$ is assumed to be a simple distribution, typically chosen as a standard gaussian distribution $$ \mathcal{N}\left(0, \boldsymbol\Sigma^{2}\right) $$, with variance $$ \boldsymbol\Sigma^{2} $$.

So how about the distribution $$ p(\mathbf{x} \mid \mathbf{z}) $$? This is also assumed to be a normal distribution, but the parameters mean $$ \boldsymbol\mu_{nn} $$ and the variance $$ \boldsymbol\Sigma_{nn}^{2} $$ are generated by our neural networks. This means that even though the process of defining that distribution can be quite complex, but it is still considered to be a simple distribution since we can parameterize it.

Thus, by doing like that we basically can approximate our data distribution $$ p(\mathbf{x}) $$ as the multiplication of two simple distributions $$ p(\mathbf{x})= \sum_z p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) $$. This is why we use latent variable models.

But, there is a problem when we use this approach directly. Remember that earlier we assumed that the prior distribution $$ p(\mathbf{z} $$ is just a standard gaussian distribution which also means that it is a unimodal distribution. 

Can you imagine if we leverage that simple distribution directly without any training or learning process to approximate a very complex distribution which is oftenly has many modes? What will happen is that the approximation result will be not good since we do not incorporate any knowledge about our data into the pre-defined prior distribution. This is the where VI plays an important role :)

## Posterior Distribution

From the previous explanation, you may be curious on how to update our prior belief represented by $$ p(\mathbf{z}) $$. Actually, bayes theorem provides a way to do that. Specifically, the answer lies on what we call as posterior distribution $$ p(\mathbf{z} \mid \mathbf{x}) $$. But in many cases, we cannot compute that expression or even if we can, then we need a lot of resources.

For understanding why is that, let me briefly recap about the use of bayes theorem in this case. 

Our posterior can mathematically be expressed as,

$$
p(\mathbf{z} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z})}{p(\mathbf{x})}
$$

where $$ p(\mathbf{x}) $$ is the marginal likelihood or evidence of our data distribution.

There is also a joint distribution $$ p(\mathbf{z}, \mathbf{x}) $$ that represents the probability of both the latent variables $$ \mathbf{z} $$ and the observed data $$ \mathbf{x} $$ occurring together. It can be factored into the product of the likelihood and the prior: 

$$
p(\mathbf{z}, \mathbf{x}) = p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z})
$$

To calculate the marginal likelihood $$ p(\mathbf{x}) $$, we can view it from bayesian perspective as the probability of observing the data $$ \mathbf{x} $$ marginalized over all possible values of the latent variables $$ \mathbf{z} $$. It is obtained by integrating (or summing, in the case of discrete variables) the joint distribution over $$ \mathbf{z} $$:

$$
p(\mathbf{x}) = \int p(\mathbf{z}, \mathbf{x}) \, \mathrm{d}\mathbf{z} = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, \mathrm{d}\mathbf{z}
$$

This integral accounts for all possible configurations of the latent variables that could have generated the observed data.

So why $$ p(\mathbf{z} \mid \mathbf{x}) $$ is very difficult to calculate? The reason is that if we want to calculate it, then it means that we also need to calculate $$ p(\mathbf{x}) $$ since it is located in the denominator of the posterior math equation. This also means that the complexity will arise rapidly as the dimensionality of $$ \mathbf{z} $$ grows. 

This is because the integral used for calculating $$ p(\mathbf{x}) $$ sums over all possible values of $$ \mathbf{z} $$, and each evaluation of the joint distribution within the integral can also be computationally expensive, leading to an intractable integral.

For example, let's imagine we have a mixture model with $$ K $$ clusters and $$ n $$ data points, each data point can belong to any of the $$ K $$ clusters as illustrated in figure 3. This is similar to having $$ n $$ slots (data points), where each slot can be filled with one of $$ K $$ options (clusters). The total number of ways to assign clusters to data points is $$ K^n $$.

Let's say we just have $$ K = 3 $$ clusters and $$ n = 10 $$ data points, the total number of cluster assignments is $$ 3^{10} $$. Mathematically, $$ 3^{10} = 59,049 $$. This is the total number of ways to assign each of the 10 data points to one of the 3 clusters. For each of these 59,049 combinations, we need to compute an integral over the means of the Gaussian components. 

Recall that this integral involves calculating the likelihood of the data given a specific set of cluster assignments and means. If there are $$ K $$ clusters, then the integral is also $$ K $$-dimensional. Thus, you now can imagine how much resources that we need to solve that simple example !!

## Approximate Posterior Distribution

Now you understand why calculating the exact posterior distribution is often very difficult. Many researchers try to solve this problem by approximating that distribution in various ways. In this post, I just want to focus on the estimation method related to the VI concept. Let's go into more detail yeeyy :)

Remember that the root cause is not the posterior itself, but its requirement to calculate marginal distribution $$ p(\mathbf{x}) $$ to derive $$ p(\mathbf{z} \mid \mathbf{x}) $$ which involves integrations. Thus, the key idea here is to approximate the posterior by replacing the annoying integral operations with the optimization process of expected value $$ E_{z \sim q_i(\mathbf{z})} $$ with respect to approximate posterior $$ q_i(\mathbf{z}) $$.

Specifically, the optimization is used to find the best approximation $$ q(\mathbf{z} ; \boldsymbol{v*}) $$  where $$ \boldsymbol{v} $$ are the variational parameters from a chosen family of distributions that minimizes the difference (specifically, the Kullback-Leibler divergence) from the true posterior $$ p(\mathbf{z} \mid \mathbf{x}) $$. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/VI/VI_optimization.png" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 4. Optimization process happening in Variational Inference (VI) (Image source : <a href="https://www.cs.columbia.edu/~blei/talks/2016_NIPS_VI_tutorial.pdf">NIPS 2016 Tutorial</a>).
</div>

For the next part of this post, stay tune :) I will complete this as soon as I can :)