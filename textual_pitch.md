# Lowcarb - A Carbon Aware Plugin for the Flower Framework
## The Problem: Skyrocketing Energy Demand of Machine Learning

With their ever increasing complexity, the energy demand to train state-of-the-art machine learning models is skyrocketing.
To put things into perspective, the training of famous networks like GPT-3, StyleGAN2-ada or GLaM had an estimated carbon footprint of about 154t, 611t and 217t CO<sub>2</sub>*.

To make things even more complicated, in some applications, the training data raised security and privacy concern.
Gathering all necessary data in one place is a privacy/mass surveillance nightmare in itself,
and risking to send sensitive data over possibly compromised networks is a bad idea anyway.
This is why Google introduced [Federated Machine Learning](http://) back in 20xx.
Here, the training of the neural network is distributed among many clients that train on their own local data.
The updated parameters of each individual federated learning client are then agglomerated by a centralized server into one final AI network.

The big advantage: the only thing that is sent over network and is gathered in one place are the weights and biases of the neutral net itself.
This obfuscates the actual sensitive data and mitigates many privacy and security concerns.

The catch: while traditional centralized learning happens in highly efficient, highly optimized data centers, 
federated learning often relies on power-inefficient local clients
(e.g. 600W consumer graphic cards in workstations for medical image analysis). Needlessly to say, this further worsens AI's carbon problem.

This is where our CarbonHack22's [Lowcarb](http://) project, a plugin for the federated learning framework Flower, comes into play.

## The Solution: Carbon-Aware Federated Machine Learning with [Lowcarb](http://)
[Flower](http:/https://flower.dev/) is a popular solution 
that brings the federated learning approach to established AI frameworks like PyTorch and TensorFlow. 
For CarbonHack22, we have developed an *all-batteries-included* plugin for the Flower framework 
to make it carbon-aware with less than 10 lines of code.

By including our plugin, the federated learning is split up into training rounds, 
where only clients with the least present carbon footprint participate.
This reduces the overall carbon footprint of the training by focusing on the local renewable energy sources in each client's region.

We've reached an implementation, where the scheduling of training rounds and the carbon-aware sampling of clients is totally obfuscated from the user.
This takes away responsibility from developers and opens up carbon-aware federated machine learning to everybody!

In one example we'll showcase later, including our plugin (less than 10 lines of code) reduced the training's energy-consumption by 14%.
We estimate that, depending on the use case, these 14% can materialize in millions of tons of CO<sub>2</sub>* that are saved by [Lowcarb](http://).

## Lowcarb and the Carbon-Aware-SDK
In most federated learning the individual clients are commonly distributed over many regions with different power grids and corresponding carbon intensity.
For each training round we try to select clients with the smallest expected carbon emission when performing their individual workload on their respective power grid.

To obtain the selection with the highest possible carbon savings, we rely on Carbon-Awares-SDK's forecast of the marginal carbon intensity for all client regions. 
With this, the Carbon-Aware-SDK supplies our Lowcarb plugin with all the information it needs to do its job.

**A small, but important technical sidenote:** in reality, our approach is not as naiv as we make it sound here.
When selecting clients exclusively on their marginal carbon intensity, depending on the clients' power grid, 
you might end up training only a few *chosen* ones, leaving out others completely.
In the end, this might bias the neural network towards those *chosen* few.
Consequently, our approach still enforces an unbiased, evenly distributed client selection over the whole training, 
even while some clients might be handicapped by their less-optimal local power grid's marginal carbon intensity.

(Nevertheless, if the network on its own is not susceptible towards biases, 
the Lowcarb framework can be configured to run the training with the lowest possible carbon footprint 
without caring for even client distribution.)