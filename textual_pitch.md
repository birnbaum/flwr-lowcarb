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

In one example [we showcase later](#The-Impact:-Example-Application-of-Carbon-Aware-Federated-Learning-using-Lowcarb), including our plugin (less than 10 lines of code) reduced the training's energy-consumption by 14%.
We estimate that, depending on the use case, these 14% can materialize in millions of tons of CO<sub>2</sub>* that are saved by [Lowcarb](http://).

## Lowcarb and the Carbon-Aware-SDK

In most federated learning the individual clients are commonly distributed over many regions / power grids with different carbon intensity.
For each training round our plugin selects the clients with the smallest expected carbon emission when performing the individual workload on their respective power grid.

To derive the best selection yielding the lowest carbon footprint, we rely on Carbon-Awares-SDK's forecast of the marginal carbon intensity for all client regions.
For each training round our plugin pulls the forecast for the next 12 hours from a Carbon-Awares-SDK's WebApi that is specified by the user. (They can choose if they host it locally or rely on a cloud solution)
The selection of clients is then based on this forecast. 
There is more to the actual client selection algorithm, so if you're interested in all the nitty-gritty details, [we explain it in more details further down the line](#The-Carbon-Aware-Client-Sampling-Algorithm). 
What's important: the Carbon-Aware-SDK supplies our Lowcarb plugin with all the information it needs to do its job.

## The Impact: Example Application of Carbon-Aware Federated Learning using Lowcarb

To demonstrate the impact of the Lowcarb plugin when used with Carbon-Aware-SDK, we have developed an example Flower application for the training of a neutral net on the privacy-sensitive X-ray data for medical image recognition of thorax disease using the XYZ dataset
As a toy example, it assumes only 100 clients spread over 14 regions worldwide, but it can also demonstrate what carbon savings can be expected from Lowcarb.

Let's start with the results right away: by using Lowcarb, the federated learning example saved 14% in carbon emission over chosing clients randomly, while obtaining same accuracy and precision, as well as even client distribution.
To further emphasize the impact of Lowcarb we extrapolated these 14% to other federated learning szenarios.

_Insert extrapolated usecases here_

## Feasibility? Use Lowcarb Today!

We consider our current implementation to be ready for most common federated learning use cases of Flower.
Thus, you are good to start using our Lowcarb plugin for Flower today.
The plugin is already available on pip: ``pip install flwr-lowcarb``.

In our GitHub repository, we also included a Jupyter notebook as a short tutorial on how to use our Lowcarb plugin.
It goes hand in hand with the official [Flower tutorial](https://flower.dev/docs/quickstart-pytorch.html). 
Here, it takes less than 10 lines of code to make it carbon aware. 
With this every Flower developer will feel confident to use our plugin right away.
We also made sure that the plugin is non-invasive as possible and won't get in the way of the users.
We are sure that this will drive widespread adoption in the future.

## The Vision

The client selection algorithm we have developed so far works for the most common federated learning use cases.
However, when it comes to its sophistication, the sky is the limit. 
In the future we plan to make it smarter, and more adaptive to the user application.
This agrees with our initial goal to make it as *hands-free*, and *batteries-included* as possible.
The Lowcarb plugin just works and saves carbon, without putting any responsibility on the user.

To give you a quick example what we have in mind here: 
Our current approach considers 12-hour forecasts, what we consider a good middle ground.
In the future we'll make this time window adaptive to the actual workload, learning from each round.
E.g. if the workload turns out to be 6 hours, a 12-hour forecast might be to short and 24 hours might be necessary.

Another example we have in mind is to not only pick the right clients, but, 
given their individual forecasts, also reschedule their workload to the best possible time in the future.
This might increase the overall time for the training to finish, but for non time-critical applications, the carbon savings are substantially increased. 
This is [current academic research in our group](https://dl.acm.org/doi/10.1145/3464298.3493399) and Lowcarb will directly profit from this research in the future.

In the bigger picture, with federated learning's increasing popularity and adaption, 
we think our Lowcarb plugin is a huge opportunity for the Carbon-Aware-SDK and carbon-aware software in general 
to reach a broader audience and raise awareness for this urgent climate issue.

## Technical Details and Toy Example

### The Carbon Aware Client Sampling Algorithm

_Insert Client Sampling Algorithm here_

**A small, but important technical sidenote:** in reality, our approach is not as naiv as we make it sound here.
When selecting clients exclusively on their marginal carbon intensity, depending on the clients' power grid, 
you might end up training only a few *chosen* ones, leaving out others completely.
In the end, this might bias the neural network towards those *chosen* few.
Consequently, our approach still enforces an unbiased, evenly distributed client selection over the whole training, 
even while some clients might be handicapped by their less-optimal local power grid's marginal carbon intensity.

(Nevertheless, if the network on its own is not susceptible towards biases, 
the Lowcarb framework can be configured to run the training with the lowest possible carbon footprint 
without caring for even client distribution.)