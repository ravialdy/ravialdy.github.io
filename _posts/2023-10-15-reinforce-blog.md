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

## Introduction

Welcome to my blog post! Today we're going to discuss about a very fascinating topic in the world of AI and Reinforcement Learning (RL) — the Policy Gradient REINFORCE Method. This method is quite famous for solving some complex problems in RL. Don't worry if you're new to this field; I'll try to keep things simple and easy to understand. First of all, I will be focusing on the background of the REINFORCE method and why it was proposed in the first place.

### Brief Recap about Reinforcement Learning Framework

Before diving into the core method, it's important to get some basics right. In RL, an agent (for simplicity, you can imagine this like a robot that learns something) interacts with an environment (like a maze). At each time $$ t $$, the agent is in a state $$ s_t $$, takes an action $$ a_t $$, and receives a reward $$ r_t $$.

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


## The Math Magic: Policy Gradient Theorem

Now, the heart of REINFORCE is the Policy Gradient Theorem. It tells us how to change the policy parameters $$ \theta $$ to increase the long-term reward $$ \rho $$.

$$
\frac{\partial \rho}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \frac{\partial \pi(s, a)}{\partial \theta} Q^{\pi}(s, a)
$$

Sounds complicated? Let's break it down:

- $$ \frac{\partial \rho}{\partial \theta} $$: This is what we're interested in. It tells us how a tiny change in $$ \theta $$ will affect the long-term reward $$ \rho $$.
  
- $$ d^{\pi}(s) $$: This is the chance of landing in a state $$ s $$ while following the policy $$ \pi $$.

- $$ Q^{\pi}(s, a) $$: This represents the value of taking an action $$ a $$ in state $$ s $$ under policy $$ \pi $$.

By calculating this expression, we know how to update $$ \theta $$ to make our policy $$ \pi $$ better.

### Policy Gradient with Function Approximation

In real-world problems, calculating $$ Q^{\pi}(s, a) $$ exactly is often impossible. Imagine a simple game where the screen is 100x100 pixels, and each pixel can be either black or white. The number of possible states is $$ 2^{(100 \times 100)} $$, which is a staggeringly large number. If you had to store a $$ Q $$-value for each of these states, you'd quickly run out of memory. This is just a black-and-white example; modern games often have high-resolution, full-color frames, making the state space even larger. So, we approximate it with a function $$ f_w(s, a) $$, controlled by parameters $$ w $$.

The theorem extends to this case, and the new update rule becomes:

$$
\frac{\partial \rho}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \frac{\partial \pi(s, a)}{\partial \theta} f_{w}(s, a)
$$

This approximation allows us to handle more complex problems without knowing everything about the environment.

### Diving Deeper into the Policy Gradient Theorem

Before we delve into the mathematical details, let's clarify why policy gradients can be a game-changer in the world of RL. Unlike traditional value-based methods which assess the "goodness" of states or state-action pairs, policy gradients aim to directly tweak the policy—a mapping from states to actions. In simpler terms, we adjust the policy parameters in such a way that maximizes our expected rewards over time. This is the also one of the main part of the REINFORCE algorithm.

#### The Formal Objective

The objective is to maximize the expected return $$ \rho(\pi) $$, defined as the average sum of rewards an agent can expect to receive while following a specific policy $$ \pi $$.

$$
\max_{\theta} \mathbb{E}_{\pi_{\theta}}\left[\sum_{t=0}^{T-1} \gamma^{t} r_{t}\right]
$$

In this equation, $$ \gamma $$ is the discount factor, $$ \theta $$ are the parameters governing the policy $$ \pi $$, and $$ T $$ is the time horizon.

#### The Policy Gradient Theorem in Detail

To find the maximum of this objective function, we need its gradient concerning $$ \theta $$. The Policy Gradient Theorem provides this invaluable piece of information. Formally, it is expressed as:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \sum_{s} d^{\pi}(s) \sum_{a} \frac{\partial \pi(s, a)}{\partial \theta} Q^{\pi}(s, a)
$$

Here, $$ d^{\pi}(s) $$ represents the stationary distribution of states when following policy $$ \pi $$, and $$ Q^{\pi}(s, a) $$ is the expected return of taking action $$ a $$ in state $$ s $$ while following $$ \pi $$.

This equation essentially tells us how a minute change in $$ \theta $$ will influence the expected return $$ \rho(\pi) $$.

#### The Log-Derivative Trick

For effective computation of the gradient, the log-derivative trick is often employed. This trick allows us to rephrase the gradient as an expectation:

$$
\frac{\partial \rho(\pi)}{\partial \theta} = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) Q^{\pi}(s_t, a_t) \right]
$$

In this expression, $$ \tau $$ symbolizes a trajectory, and $$ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) $$ is the gradient of the log-probability of the action taken at time $$ t $$.

#### Why Should We Care About Policy Gradients?

1. **Direct Optimization**: Unlike value-based methods, policy gradients directly tweak what actually matters—the policy itself.

2. **Stochasticity Handling**: Policy gradients can optimize stochastic policies, crucial for situations where the optimal action can differ due to inherent randomness.

3. **Sample Efficiency**: Because the focus is on policy improvement, fewer samples are often required to learn a good policy, making the method generally more efficient.

By understanding the Policy Gradient Theorem and its underlying principles, you'll find that it's a fundamental building block for more advanced algorithms in the RL domain. Not only does it provide a method to directly optimize the policy, but it also offers the flexibility, stability, and efficiency required for real-world applications.

### Why Is This Important?

1. **Efficiency**: The theorem allows us to improve the policy without knowing how each change will affect every state. We only need to calculate the effect on the long-term reward, which is more efficient.

2. **Stability**: The method provides a stable way to improve the policy gradually. It avoids big jumps that could destabilize the learning process.

3. **Flexibility**: The approximation allows us to tackle problems where it's hard or impossible to know everything about the environment.