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

Welcome to my blog post! Today we're going to discuss about a very fascinating topic in the world of AI and Reinforcement Learning (RL) — the Policy Gradient REINFORCE Method. This method is quite famous for solving some complex problems in RL. Don't worry if you're new to this field; I'll try to keep things simple and easy to understand. First of all, I will be focusing on the background of the REINFORCE method and why it was proposed in the first place.

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

The agent follows a "policy" $$ \pi(s, a) $$, which tells it what action $$ a $$ to take when in state $$ s $$. This policy is controlled by some parameters $$ \theta $$, which we adjust to make the policy better. Here are the more formal definitions of important terms in RL:

- **Environment**: The space or setting in which the agent operates.

- **State**: The condition of the environment at a given time point, often denoted as $$ s $$ or $$ s_t $$ to indicate its time-dependence.

- **Agent**: An entity that observes the state of the environment and takes actions to achieve a specific objective.

- **Action**: A specific operation that an agent can execute, typically denoted by $$ a $$ or $$ a_t $$.

- **Policy**: A policy, denoted by $$ \pi(a \mid s) $$ or $$ \pi(s, a) $$, is a mapping from states to actions, or to probabilities of selecting each action.

- **Reward**: A scalar value, often denoted by $$ r $$ or $$ r_t $$, that the environment returns in response to the agent's action.

### What Are We Trying to Optimize?

The ultimate goal in the RL method is to maximize the long-term reward. We often denote this as $$ \rho(\pi) $$. This is the average reward the agent expects to get over time while following policy $$ \pi $$.

## The Problem with Traditional Methods

In RL, we often want a computer to learn how to make decisions by itself. For instance, think of a game where a robot must find its way out of a maze. The robot learns by trying different paths and seeing which ones get it out of the maze faster. Sounds simple, right? But when the maze is large and complicated, the number of decisions the robot must make becomes huge. This is where function approximators like neural networks come in handy; they help the robot generalize from its experience to make better decisions.

For a long time, people used something called a "value-function approach" to do this. In this approach, all the effort is put into calculating a value for each decision or "action" the robot can make. The robot then chooses the action with the highest value. However, this approach has some downsides:

  - **Deterministic Policies**: The traditional method is good for making a fixed decision, but sometimes we want the robot to be a bit random. Why? Because the best decision can depend on chance or unknown factors.

  - **Sensitive Choices**: A tiny change in the calculated value can dramatically change the action taken by the robot. This is risky because we want the robot to learn stable behavior.

  - **Convergence Issues**: Looks like a fancy term, but it simply means that using the value-function approach does not always guarantee that the robot will find the best way to act in all situations.

## The Policy Gradient Theorem

Before we delve into the details of REINFORCE algorithm, let's clarify why policy gradients can be a game-changer in the world of RL. The reason for this is that REINFORCE itself belongs to this approach. Unlike traditional value-based methods which assess the "goodness" of states or state-action pairs, policy gradients aim to directly tweak the policy—a mapping from states to actions. This approach can avoid at least three potential problems:

  - **Curse of Dimensionality**: Value-based methods often suffer from the "curse of dimensionality." The state-action space can grow exponentially with the number of features describing the state and the range of actions available. This makes the computational cost expensive.

  - **Non-Markovian Environments**: In some cases, the environment is not following the Markov Property, where the future state depends only on the current state and action. In that case, using a value function to capture the "goodness" of a state can be misleading or incomplete.

  - **Exploration vs. Exploitation**: Value-based methods often cause the agent to stick to known high-value states and actions, missing out on potentially better options. While exploration strategies exist, they add another layer of complexity to the algorithm.

In simpler terms, by focusing directly on optimizing the policy, policy gradient methods can sidestep many of these issues. They are particularly well-suited for high-dimensional or continuous action spaces, can naturally accommodate stochastic policies, and are less sensitive to the challenges associated with value function approximation.

### The Formal Objective

The objective is to maximize the expected return $$ \rho(\pi) $$, defined as the average sum of rewards an agent can expect to receive while following a specific policy $$ \pi $$.

$$
\max_{\theta} \mathbb{E}_{\pi_{\theta}}\left[\sum_{t=0}^{T-1} \gamma^{t} r_{t}\right]
$$

In this equation, $$ \gamma $$ is the discount factor, $$ \theta $$ are the parameters governing the policy $$ \pi $$, and $$ T $$ is the time horizon.

### The Policy Gradient Theorem in Detail

To find the maximum of this objective function, we need its gradient concerning $$ \theta $$. The Policy Gradient Theorem provides this invaluable piece of information. Formally, it is expressed as:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \frac{\partial \pi(s, a)}{\partial \theta} Q^{\pi}(s, a)
$$

Here, $$ d^{\pi}(s) $$ represents the stationary distribution of states when following policy $$ \pi $$, and $$ Q^{\pi}(s, a) $$ is the expected return of taking action $$ a $$ in state $$ s $$ while following $$ \pi $$.

This equation essentially tells us how a minute change in $$ \theta $$ will influence the expected return $$ \rho(\pi) $$.

### The Log-Derivative Trick

For effective computation of the gradient, the log-derivative trick is often employed. This trick allows us to rephrase the gradient as an expectation:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) Q^{\pi}(s_t, a_t) \right]
$$

This is essentially a restatement of the gradient of a function with respect to its logarithm, which can be formally described as:

$$
\nabla_{\theta} \pi(a \mid s) = \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s)
$$

To prove this, we'll take the derivative of $$ \log \pi(a \mid s) $$ with respect to $$ \theta $$:

$$
\nabla_{\theta} \log \pi(a \mid s) = \frac{\nabla_{\theta} \pi(a \mid s)}{\pi(a \mid s)}
$$

Rearranging the terms gives:

$$
\nabla_{\theta} \pi(a \mid s) = \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s)
$$

Now, let's see how this trick fits into the policy gradient equation. The original policy gradient theorem can be expressed as:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \nabla_{\theta} \pi(s, a) Q^{\pi}(s, a)
$$

Here, $$ d^{\pi}(s) $$ represents the stationary distribution of states when following the policy $$ \pi $$.

When you apply the Log-Derivative Trick to $$ \nabla_{\theta} \pi(s, a) $$, it becomes $$ \pi(s, a) \nabla_{\theta} \log \pi(s, a) $$. Substituting this into the policy gradient theorem, and then rewriting the sum as an expectation, we obtain:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) Q^{\pi}(s_t, a_t) \right]
$$

In this expression, $$ \tau $$ symbolizes a trajectory, and $$ \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) $$ is the gradient of the log-probability of the action taken at time $$ t $$.

This brings us to the policy gradient equation that I mentioned earlier. But why is this necessary? Computing gradients directly can be computationally expensive or even infeasible, especially when you are dealing with complex policies parameterized by neural networks.

Let's say you have a term like $$ \pi(a \mid s) $$ that depends on some parameters $$ \theta $$. Taking the derivative of this term directly with respect to $$ \theta $$ might be challenging. However, the Log-Derivative Trick provides a workaround. It transforms this term into:

$$
\nabla_{\theta} \pi(a \mid s) = \pi(a \mid s) \nabla_{\theta} \log \pi(a \mid s)
$$

Notice that $$ \nabla_{\theta} \log \pi(a \mid s) $$ is usually easier to compute. Also, this trick allows us to rephrase the Policy Gradient Theorem in a more computationally friendly manner.

### Why Should We Care About Policy Gradients?

1. **Direct Optimization**: Unlike value-based methods, policy gradients directly tweak what actually matters—the policy itself.

2. **Stochasticity Handling**: Policy gradients can optimize stochastic policies, crucial for situations where the optimal action can differ due to inherent randomness.

3. **Sample Efficiency**: Because the focus is on policy improvement, fewer samples are often required to learn a good policy, making the method generally more efficient.

By understanding the Policy Gradient Theorem and its underlying principles, you'll find that it's a fundamental building block for more advanced algorithms in the RL domain. Not only does it provide a method to directly optimize the policy, but it also offers the flexibility, stability, and efficiency required for real-world applications.


## Introducing REINFORCE Algorithm

After understanding the power and flexibility of Policy Gradient methods, it's time to delve into one of its most famous implementations: the REINFORCE algorithm which stands for REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility, this algorithm is not just a fancy acronym; it's often considered as one of the fundamental building block in the world of Reinforcement Learning.

### Main Idea of REINFORCE

Remember that the Policy Gradient methods aim to optimize the policy in a way that increases the expected return from any state $$ s $$. However, calculating the true gradient of this expected return is often computationally infeasible or requires a model of the environment, which we usually don't have. REINFORCE is one of the Policy Gradient algorithms that makes us possible to directly optimizing the policy function $$ \pi(a \mid s) $$ to maximize the cumulative reward. While there are many algorithms under the Policy Gradient category, REINFORCE stands out for its simplicity and directness in estimating the gradient.

The core idea of REINFORCE that differentiate it with other methods is in its utilization of Monte Carlo methods to estimate the gradients needed for policy optimization. By taking sample paths through the state and action space, REINFORCE avoids the need for a model of the environment and sidesteps the computational bottleneck of calculating the true gradients. This is particularly useful when the state and/or action spaces are large or continuous, making other methods infeasible.

### REINFORCE & Policy Gradient Theorem

Recall that the Policy Gradient Theorem provides an expression for the gradient of the expected return with respect to the policy parameters. REINFORCE directly employs this theorem but takes it a step further by providing a practical way to estimate this gradient through sampling. The mathematical equation for obtaining expected return $$ J(\theta) $$ using this theorem can be written as:

$$
\nabla J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) Q^{\pi}(S_t, A_t) \right]
$$

REINFORCE simplifies this expression by utilizing the Monte Carlo estimate for $$ Q^{\pi}(S_t, A_t) $$, which is the sampled return $$ G_t $$:

$$
\nabla J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) G_t \right]
$$

In essence, REINFORCE is a concrete implementation of the Policy Gradient method that uses Monte Carlo sampling to estimate the otherwise intractable or unknown quantities in the Policy Gradient Theorem. By doing so, it provides a computationally efficient, model-free method to optimize policies in complex environments.

### Mathematical Details of REINFORCE

The REINFORCE algorithm can be understood through a sequence of mathematical steps, which are as follows:

1. **Initialize Policy Parameters**: Randomly initialize the policy parameters $$ \theta $$.

2. **Generate Episode**: Using the current policy $$ \pi_\theta $$, generate an episode $$ S_1, A_1, R_2, \ldots, S_T $$.

3. **Compute Gradients**: For each step $$ t $$ in the episode,
    - Compute the return $$ G_t $$.
    - Compute the policy gradient $$ \Delta \theta_t = \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t  \mid  S_t) $$.
    
4. **Update Policy**: Update the policy parameters $$ \theta $$ using $$ \Delta \theta $$.

The key equation that governs this update is:

$$
\nabla J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) G_t \right]
$$

Here, $$ G_t $$ is the return obtained using a Monte Carlo estimate, providing a sample-based approximation of $$ Q^\pi(S_t, A_t) $$.

### Conclusion and Limitations

While REINFORCE is oftenly used for its simplicity and directness, it's also essential to recognize its limitations. The method tends to have high variance in its gradient estimates, which could lead to unstable training. However, various techniques, like using a baseline or employing advanced variance reduction methods, can alleviate these issues to some extent.

REINFORCE is often the easy choice when you need a simple yet effective method for policy optimization, especially in high-dimensional or continuous action spaces.