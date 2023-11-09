---
layout: post
title: Understanding Phenomenal REINFORCE Policy Gradient Method
date: 2023-10-15 09:56:00-0400
description:
tags: Reinforcement Learning (RL), REINFORCE, Policy Gradient
categories: blogpost
giscus_comments: true
related_posts: false
related_publications: REINFORCE
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

Welcome to my blog post! Today we are going to discuss about a very fascinating and important topic in the world of Reinforcement Learning (RL) — the Policy Gradient REINFORCE Method. This method is quite famous for solving some complex problems in RL. Don't worry if you're new to this field, I will try to keep things simple and easy to understand. First of all, I will be focusing on the background of the Policy Gradient Theorem and why it was proposed in the first place.

### Brief Recap about Reinforcement Learning

Before diving into the core method, it's important to get some basics right. Imagine we have a small robot placed at the entrance of a maze. The maze is simple: it has walls, open passages, and a cheese located at the exit. The robot's ultimate goal is to find the most efficient path to reach the cheese. In RL, an agent (a robot in this case) interacts with an environment (like a maze). At each time $$ t $$, the agent is in a state $$ s_t $$, takes an action $$ a_t $$, and receives a reward $$ r_t $$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/reinforce/DALL·E 2023-10-24 15.38.59 - Vector design of a playful scene where a cartoon robot is gearing up to enter a maze. The maze's pathways are clear, with walls separating the routes.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 1. Illustration of an agent (robot) tries to reach the cheese as soon as possible (Image source : DALLE-3).
</div>

The agent follows a "policy" $$ \pi(a \mid s) $$, which tells it what action $$ a $$ to take when in state $$ s $$. This policy is controlled by some parameters $$ \theta $$, which we adjust to make the policy better. Here are the more formal definitions of important terms in RL:

- **Environment**: The space or setting in which the agent operates.

- **State**: The condition of the environment at a given time point, often denoted as $$ s $$ or $$ s_t $$ to indicate its time-dependence.

- **Agent**: An entity that observes the state of the environment and takes actions to achieve a specific objective.

- **Action**: A specific operation that an agent can execute, typically denoted by $$ a $$ or $$ a_t $$.

- **Policy**: A policy, denoted by $$ \pi(a \mid s) $$ or $$ \pi(s, a) $$, is a mapping from states to actions, or to probabilities of selecting each action.

- **Reward**: A scalar value, often denoted by $$ r $$ or $$ r_t $$, that the environment returns in response to the agent's action.

## The Problem with Traditional Methods

In RL, we often want a computer to learn how to make decisions by itself. For instance, imagine a game where a robot must find its way out of a maze. The robot learns by trying different paths and seeing which ones get it out of the maze faster. It sounds simple right? But what if the maze is large and complicated? Then the number of decisions the robot must make becomes huge. This is where function approximators like neural networks come into play, they can help the robot to generalize from its experience to make better decisions.

For a long time, people used something called a "value-function approach" to do this. In this approach, all the effort is put into calculating a value for each decision or "action" the robot can make. The robot then chooses the action with the highest value. However, this approach has some downsides:

  - **Deterministic Policies**: The traditional method is good for making a fixed decision, but sometimes we want the robot to be a bit random. Why? Because the best decision can depend on chance or unknown factors.

  - **Sensitive Choices**: A tiny change in the calculated value can dramatically change the action taken by the robot. This is risky because we want the robot to learn stable behavior.

  - **Convergence Issues**: This simply means that using the value-function approach does not always guarantee that the robot will find the best way to act in all situations.

## The Policy Gradient Theorem

Before we delve into the details of REINFORCE algorithm, we need to understand what policy gradient really is and why it can be a game-changer in the world of RL. The reason for this is that REINFORCE itself belongs to this approach. Unlike traditional value-based methods which assess the "goodness" of states or state-action pairs, policy gradients aim to directly optimize the policy. Let's take a look at what kind of potential problems this approach has and how policy gradient methods can avoid those issues:

  - **Curse of Dimensionality**: Value-based methods require an estimated value for every possible state or state-action pair. As the number of states and actions increases, the size of the value function grows exponentially. By focusing directly on optimizing the policy, policy gradient methods can avoid this issue since it works with a much smaller set of parameters.

  - **Non-Markovian Environments**: In some cases, the environment is not following the Markov Property, where the future state depends only on the current state and action. Policy gradient methods do not rely on the Markov property because they do not predict future values; they only need to evaluate the outcomes of current actions.

  - **Exploration vs. Exploitation**: Value-based methods often cause the agent to stick to known high-value states and actions, missing out on potentially better options. By adjusting the policy parameters, policy gradient methods can encourage the agent to explore different actions with probabilities, rather than committing to the action with the highest estimated value.

In other words, the key difference is that the size of the parameter set in policy gradient methods is determined by the complexity of the policy representation (e.g., the architecture of the neural network), not by the size of the state or action space.

For example, suppose you have a neural network with 1000 parameters. It can still process thousands or even millions of different states and output actions for each of them because the same parameters are used to evaluate every state through the network's forward pass. This means that even for complex environments, the number of parameters doesn't necessarily increase with the complexity of the state space, which is often the case with value-based methods.

Thus, policy gradient methods are particularly well-suited for high-dimensional or continuous action spaces, can naturally accommodate stochastic policies, and are less sensitive to the challenges associated with value function approximation.

### The Formal Objective

The objective is to maximize the expected return $$ J(\theta) $$, defined as the average sum of rewards an agent can expect to receive while following a specific policy $$ \pi $$.

$$
\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T-1} \gamma^{t} r_{t}\right]
$$

In this equation, $$ \gamma $$ is the discount factor, $$ \theta $$ are the parameters governing the policy $$ \pi $$, and $$ T $$ is the time horizon.

### The Policy Gradient Theorem in Detail

To find the maximum of this objective function, we need its gradient w.r.t $$ \theta $$. The Policy Gradient Theorem provides this invaluable piece of information. Formally, it is expressed as:

$$
\frac{\partial J(\theta)}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \frac{\partial \pi(a \mid s)}{\partial \theta} Q^{\pi}(s, a)
$$

Here, $$ d^{\pi}(s) $$ represents the stationary distribution of states when following policy $$ \pi $$, and $$ Q^{\pi}(s, a) $$ is the expected return of taking action $$ a $$ in state $$ s $$ while following $$ \pi $$.

This equation essentially tells us how a change in $$ \theta $$ will influence the expected return $$ J(\theta) $$.

### The Log-Derivative Trick

For effective computation of the gradient, the log-derivative trick is often employed. It allows us to express the derivative of the policy with respect to its parameters $$ \theta $$ in a simpler form. The trick is based on the identity:

$$
\nabla_{\theta} \pi(a \mid s) = \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s)
$$

This identity is derived from the property of logarithms that the derivative of the log of a function is the derivative of the function divided by the function itself:

$$
\nabla_{\theta} \log \pi(a \mid s) = \frac{\nabla_{\theta} \pi(a \mid s)}{\pi(a \mid s)}
$$

Therefore, multiplying both sides by $$ \pi(a \mid s) $$, we get:

$$
\nabla_{\theta} \pi(a \mid s) = \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s)
$$

### The Role of Log-Derivative Trick

Now, let's see how this trick fits into the policy gradient equation. When we substitute $$ \nabla_{\theta} \pi(a \mid s) $$ using the log-derivative trick into the Policy Gradient Theorem, we get:

$$
\frac{\partial J(\theta)}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s) Q^{\pi}(s, a)
$$

The sum over states and actions weighted by the state distribution $$ d^{\pi}(s) $$ and the policy $$ \pi(a \mid s) $$ can be seen as an expectation. This is because the expectation of a random variable is the sum of the possible values of the random variable weighted by their probabilities which in this case is the product of $$ d^{\pi}(s) $$ and $$ \pi(a \mid s) $$. Thus, we can rewrite the above sum as:

$$
\frac{\partial J(\theta)}{\partial \theta} = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) Q^{\pi}(s_t, a_t) \right]
$$

Here, $$ \mathbb{E}_{\tau \sim \pi_{\theta}} $$ denotes the expected value when the trajectory $$ \tau $$ (a sequence of states and actions) is sampled according to the policy $$ \pi $$ parameterized by $$ \theta $$. This form is computationally more convenient because we can estimate the expectation by sampling trajectories and calculating the average over them, which become an unbiased estimator for that expectation. We will use this in the REINFORCE algorithm.


## Introducing REINFORCE Algorithm

After understanding the power and flexibility of Policy Gradient methods, it's time to delve into one of its most famous implementations: the REINFORCE algorithm which stands for REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility, this algorithm is often considered as one of the most important and fundamental building block in the world of Reinforcement Learning.

### Main Idea of REINFORCE

The core idea of REINFORCE that differentiate it with other methods is in its utilization of Monte Carlo methods to estimate the gradients needed for policy optimization. By taking sample paths through the state and action space, REINFORCE avoids the need for a model of the environment and sidesteps the computational bottleneck of calculating the true gradients. This is particularly useful when the state and/or action spaces are large or continuous, making other methods infeasible.

For those who are not familiar with Monte Carlo approach, it is basically the process of sampling and averaging for estimating expected values in situations with large or infinite state spaces. By doing this, we can get the estimates that are unbiased without incorporating all available data.

For better understanding, imagine we have a standard six-sided dice, and we want to estimate the expected value of a roll. The true expected value, $$ E(X) $$, for a fair dice roll is the average of all possible outcomes:

$$
E(X) = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5
$$

However, suppose we don't know this formula or we are dealing with a more complex system where the expected value isn't straightforward to calculate. This is where Monte Carlo estimation comes into play. 

Let's say we perform a series of dice rolls and record the results after each roll. We'll then calculate the cumulative average after each roll and see how this average converges to the true expected value of 3.5 as we increase the number of rolls.

<div class="row mt-4">
    <div class="col-12 col-lg mt-4 img-container">
        {% include figure.html path="/assets/img/reinforce/graph_montecarlo_sample.jpg" class="img-fluid" zoomable=true %}
    </div>
</div>
<div class="caption text-center mb-4">
    Figure 2. Convergence plot for dice rolls when we increase number of samples.
</div>

In the plot above, you can observe the cumulative average of the dice rolls as the number of rolls increases. This line represents the average of the rolled numbers after each roll. The red dashed line marks the true expected value of a fair die roll, which is 3,5. As you can see, if we take enough samples, then it will give us similar result as the true expected value of that roll. This example demonstrates that taking enough sampling can lead us to have results that are close enough to the true expected value.

To understand more about the role of Monte Carlo in the REINFORCE, see the explanation below.

### REINFORCE & Policy Gradient Theorem

Recall that the Policy Gradient Theorem provides an expression for the gradient of the expected return with respect to the policy parameters. REINFORCE directly employs this theorem but takes it a step further by providing a practical way to estimate this gradient through sampling. The mathematical equation for obtaining expected return $$ J(\theta) $$ using this theorem can be written as:

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) Q^{\pi}(s_t, a_t) \right]
$$

REINFORCE simplifies this expression by utilizing the Monte Carlo estimate for $$ Q^{\pi}(s_t, a_t) $$, which is the sampled return $$ G_t $$:

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) G_t \right]
$$

Here, $$ G_t $$ is the return obtained using a Monte Carlo estimate, providing a sample-based approximation of $$ Q^{\pi}(s_t, a_t) $$.

In essence, REINFORCE is a concrete implementation of the Policy Gradient method that uses Monte Carlo sampling to estimate the otherwise intractable or unknown quantities in the Policy Gradient Theorem. By doing so, it provides a computationally efficient, model-free method to optimize policies in complex environments.

### Mathematical Details of REINFORCE

The REINFORCE algorithm can be understood through a sequence of mathematical steps, which are as follows:

1. **Initialize Policy Parameters**: Randomly initialize the policy parameters $$ \theta $$.

2. **Generate Episode**: Using the current policy $$ \pi_\theta $$, generate an episode $$ S_1, A_1, R_2, \ldots, S_T $$.

3. **Compute Gradients**: For each step $$ t $$ in the episode,
    - Compute the return $$ G_t $$.
    - Compute the policy gradient $$ \Delta \theta_t = \alpha \gamma^t G_t \nabla_\theta \log \pi_{\theta}(a_t \mid s_t) $$.
    
4. **Update Policy**: Update the policy parameters $$ \theta $$ using $$ \Delta \theta $$.

The key equation that governs this update is:

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_{\theta}(a_t \mid s_t) G_t \right]
$$

### Conclusion and Limitations

While REINFORCE is oftenly used for its simplicity and directness, it's also essential to recognize its limitations. The method tends to have high variance in its gradient estimates, which could lead to unstable training. However, various techniques, like using a baseline or employing advanced variance reduction methods, can alleviate these issues to some extent.

REINFORCE is often the easy choice when you need a simple yet effective method for policy optimization, especially in high-dimensional or continuous action spaces.