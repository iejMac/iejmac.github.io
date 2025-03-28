---
layout: post
title:  "exploring tensor alignments in neural networks (part 1)"
---
this document is a summary from a light exploration into neural network parametrizations. the parametrization space we’ll focus on is the abc-parametrization whose definition we borrow from [1]: 

**The Cauchy-Schwarz Inequality**\
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

## background
in numerical optimization we want our algorithms to be fast and stable. these two qualities exist in tension, creating an inevitable tradeoff. pushing for speed positions work at the stability boundary [2, 3] where small changes in experimental setup can nudge us off the edge. what functions reliably at one scale may fail at another, with instabilities often remaining hidden until deployment at scale.

we can analyze an important aspect of training multi-layer neural networks with the following representative model:


equation 1: pre-activation vector at layer l during training of an MLP
this equation captures how perturbations in weights and activations propagate through the network during training. let's examine a simple example for how to design a parametrization to combat instabilities - a single linear weight matrix acting on an input vector.



if we naively parametrize our weight matrix, the average coordinate scale is O(sqrt(n)) - this isn’t great if we want to scale model width on hardware with finite precision. lets fix this with a parameter multiplier:


with the 1/sqrt(n) multiplier, for any width we decide to go with, our matrix-vector product will be stable w.r.t. width scaling i.e. the coordinate scale is not a function of the width.
one limitation of this example is its idealistic assumption that W and x are independently sampled with zero mean, allowing us to apply the Central Limit Theorem. this is only true at initialization when both are randomly drawn from zero-mean distributions. after the first update, we must consider potential alignments between W and x.
consider an extreme case: if during the first optimizer update, all rows of W were transformed into x.T, the product would scale as O(n) rather than O(√n) - they would be fully aligned. many researchers assume this "full alignment" [5, 6] after training sufficiently warms up and design "defensive" parameterizations to ensure stability under these extreme conditions. however, in [1] they measure the log-alignment ratio and demonstrate this is often overly conservative, suggesting performance gains are possible by relaxing these alignment assumptions.

## solving for optimal parametrizations
the optimal parameterization maximizes convergence speed while maintaining training stability. to derive such parameterizations, we need clear objectives. the first is straightforward—we want to maximize learning rate. the second requires more nuance. for a comprehensive discussion on this topic, we recommend [1], from which we borrow the following definition:

equation 2: scale of change in activations

intuitively, when r_l = 0, the change in activations remains constant regardless of width scaling. Any deviation from this equilibrium pushes us toward either vanishing or exploding activation changes, resulting in poor performance or instability, respectively

with our neural network training dynamics described in equation 1 and our stability metric established in equation 2, we can now analyze alignment effects. we can examine the log-alignment ratio for each term in equation 1 (except the first term, since we have no alignment during initialization).

and derive a system of equations and inequalities which describe stable training by ensuring that our stability constraints are met during each training step (see Appendix of [1] for derivation):


figure 2: system of constraints which define stable training
## seeing the parametrization landscape
to verify our understanding and theory, we can create a visualization - let's grab an interesting point on the polyhedron defined by the above system of equations and inequalities, like muP [4], and probe around it. at each point, we can check if the system is satisfied and also train a simple neural network to measure the metrics we discussed above.

we can borrow the nice fractal visualization from [2]:
tiny MLPs with 3 layers and a hidden dimension of 64, using ReLU
synthetic data where the input dataset is sampled from an 8-dimensional standard gaussian and output is sampled from 1-dimensional standard gaussian
training with MSE, full-batch gradient descent for 500 steps of SGD

to make 2D plots, let's explore 2 slices of our parameterization space (a3 vs b3 and c1 vs c2). for each, we'll center the graph at the muP parameterization and assume full alignment.
the color of each pixel represents the mean scale of change in activations (compared to initialization) in the last 100 steps of training (darker red means more divergence, darker blue means vanishing to 0)
the best models are trained where this change is constant scale (i.e., 0), which will appear as bright blue colors
on each graph, we will overlay the boundary of stability (where rL = 0)


figure 3: training stability visualized for a grid of different parameterizations based on muP. each pixel is the mean of the change in activation scale (since initialization) for the last 100 steps of training. the theoretical boundary of stability defined by the system in figure 2 is shown in white 
these visualizations show strong overlap with our theoretical predictions and practical results. however, discrepancies appear in two regions: below and to the left of the stable training frontier in the left grid, and the right grid respectively. could our alignment assumptions be creating overly restrictive constraints? to investigate this hypothesis, let's examine one experiment from the figure above and track how its alignment variables evolve throughout the training process.



figure 4: u (left), omega (middle), and alpha (right) alignment metrics plotted for a single run from the figure 3 experiment during training. the shade of the lines denotes the layer index with darker shades denoting earlier layers

we can extract two key insights from this analysis:
alignment appears to converge during training
our initial assumptions were inaccurate - full alignment would mean 0.5 omega for all layers and 1.0 u and alpha for all layers. While our omega estimates were relatively accurate, we significantly overestimated alignment for alpha and u variables.
with these empirically measured alignment values, we can now update our visualization to reflect more accurate alignment assumptions.

figure 5: same as figure 3 but stable training frontier uses measured alignment assumptions 

fascinating! by using measured alignment assumptions, we can improve our estimates of the stable training frontier!

## maximizing update size
our analysis reveals a key opportunity: by overestimating alignment, we unnecessarily restrict learning rates. the full-alignment muP approach proves overly conservative.
rather than manually probing layer learning rates, we formulated this as a constrained optimization problem: maximize learning rates while satisfying our established stability inequalities. we developed a solver that accepts any ab parameterization and alignment settings (alpha, omega, u), then outputs maximal stable learning rate exponents.
by applying this to muP with our empirically measured alignment values, we discovered that for that experiment we can increase the learning rate exponent of the second layer by 0.404 (as shown in the figures). this translates to multiplying the middle layer learning rate by width^0.404 while maintaining stability. if we rerun this experiment in that setting we see the following substantial improvement in loss:

figure 6: effect of using maximal learning rate exponents derived from measured alignment variables

by using data-parameter alignment measurements, we minimize loss more effectively in this simple synthetic setting.

let's reproduce this in a slightly more challenging cifar-10 setting. for each experiment, we:
create a depth x width grid to ensure robustness across model scales
sweep over base learning rates and pick the best for each experiment
try 4 base parameterizations: mean-field, mup, standard, and ntk, all using full alignment assumptions
try 2 optimizers: sgd and adam (no weight decay)
run each experiment for 2000 steps with batch size 256, logging loss, per-layer learning rates, alignment metrics, and activation change scale
to test our strategy:
start with base case using full alignment assumption and derived learning rate exponents
take ab-parameterization and get converged alignment variables from lowest-width version
derive maximal learning rate exponents using measured alignments, then run with that
this method is practical—run a small-scale experiment to discover data-parameter alignments, then apply to your target run. for brevity, i'll only show mean-field parameterization results, but the pattern repeats across parameterizations. our measured-alignment-based learning rate exponents outperform base mfp for adam, but for sgd, it's the opposite.




our methodology—using converged alignment variables from smaller runs to initialize larger, separate runs—rests on the assumption that alignment is primarily determined by data and parameterization choices. since these properties remain constant across all runs in our grid, we expect the alignment patterns to generalize. to check this assumption, we can compare the converged alignment variables between full alignment initialization and measured alignment initialization across different optimizers and parameterizations used in our experiments. 


metric: abs((full - measured)[-100:].mean())

as it turns out - initial alignment assumptions influence the convergence trajectory itself, potentially invalidating our original assumptions. consequently, our calculated "maximal" learning rate exponents may no longer be truly maximal or maintain stability.
furthermore, the results clearly show that pre-initialized alignments cause significantly greater deviation in alpha alignments with SGD compared to Adam. this suggests an important phenomenon: certain optimizers exhibit more robust alignment convergence, enabling more reliable cross-model transfer of alignment assumptions.
## dynamic maximal lr schedule

to address this issue, we can ensure the property of being a maximal update parameterization remains invariant throughout training. we implement this by solving for maximal learning rate exponents in real-time at each step, integrating our solver into a learning rate schedule that updates based on current alignment measurements.

summarizing the results:
dynamic maximal learning rate schedules consistently outperform our pre-measured method or maintain better results than baseline experiments
in the worst case, across various parameterization and optimizer combinations, we match baseline performance

as an example of a significant improvement, here's SP + Adam:

and here’s an example where our pre-measured alignment method underperformed but our new dynamic scheduler was able to at least match full alignment - muP + SGD 

our hypothesis for this case concerns our definition of "maximal per-layer learning rate exponents." in our solver, we maximize the sum of learning rates, allowing tradeoffs between layers—we would reduce one layer's rate by 0.1 if it enables increasing others by a total exceeding 0.1.
this approach assumes all layers contribute equally to optimization, which may not be universally true. if this assumption explains our inconsistent improvements, we should observe a correlation between performance and tradeoff intensity. to test this, we'll plot loss improvement over the baseline against a metric measuring these tradeoffs—specifically, the average decrease in learning rate per layer (calculated as the sum of all decreases divided by the number of layers, which equals zero if all layers receive increased rates).


as we can see, there is a clear relationship showing that more tradeoffs lead to less loss improvement. the good thing is that in the vast majority of cases we see a healthy improvement in loss. in the future an interesting direction to look into might be a solver which only considers pareto improvements over full alignment.
## what impacts alignment?
its not a silly assumption that the alignment is constant and increases after training has sufficiently warmed up. to visualize this we can look at some simple math:


what we see above is that the primary mechanism through which the weights get updated during training is by adding “shades” of the data therefore its natural to assume the alignment would increase on the same sample. the issue with this assumption is that in most training workflows we rarely iterate over the same data and also we sometimes apply augmentations to it, like adding noise. once again we can look at what the math in the previous example would look like if we decided to add some noise to x:



what we see above is that even if there’s a lot of alignment developing in that first term w^{t} @ x, there’s plenty of other terms in the output which will have no alignment given that is the product of something with a noise vector. a simple example of the latter is diffusion models where at each step we will be adding noise to the data which will fundamentally limit the amount of alignment which can develop in at least the earlier layers.
to get an empirical sense of this we can add noise to our data during training and see how the alignment variables converge. below we can see a CIFAR-10 training run where we do a linear interpolation between the noise and the data according to the signal strength parameter



you can see that for some alignment variables, decreasing signal strength leads to decreased alignment.	
## conclusion
our exploration into tensor alignments reveals a critical insight: the traditional assumption that these alignments are constant and significant isn’t always correct. in reality, they're dynamic, and vary across network layers and training step. we can likely unlock faster training without sacrificing stability by measuring actual alignments instead of relying on theoretical assumptions. perhaps most intriguing is the discovery that adding noise directly manipulates alignment properties, creating a surprising connection to diffusion models. this suggests our approach could be especially valuable in training regimes where data characteristics change systematically.
stay tuned for Part 2, where we'll transform these findings into practical, ready-to-use methods that can be applied to real-world training scenarios.
future work

as we pointed out above, our notion of “maximal” per layer learning rates isn’t necessarily correct. one follow up idea could be to test out different notions of maximal f.e. ones that only allow pareto improvements over full alignment or which only allow epsilon decrease over full alignment.
what else impacts alignment? can we predict a decrease in alignment based on diffusion noise schedules and use that to our advantage? can we expect multimodal models to have lower average alignment?
cheaper ways of measuring alignment - every N steps, every N steps with decreasing frequency due to convergence, every N layers, etc. 
how to compose with training-step-wise lr schedules:
just multiply the two schedules? i.e. base warmup-stable-decay schedule multiplied by the one discovered through alignment
replace classic linear warmup with alignment discovery?

## references
[1] Scaling Exponents Across Parameterizations and Optimizers (https://arxiv.org/abs/2407.05872)
[2] The boundary of neural network trainability is fractal (https://arxiv.org/abs/2402.06184)
[3] Why Momentum Really Works (https://distill.pub/2017/momentum/)
[4] Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (https://arxiv.org/abs/2203.03466)
[5] Feature Learning in Infinite-Width Neural Networks (https://arxiv.org/abs/2011.14522)
[6] Scalable Optimization in the Modular Norm (https://arxiv.org/abs/2405.14813)
