# Lowcarb - Carbon-Aware Federated Learning

A plugin for the Flower framework, which enables privacy-aware and carbon-aware machine learning.


## The Problem: Skyrocketing Energy Demand of Machine Learning

The ever-increasing complexity of machine learning (ML) models as well as growing amounts of data, is leading to new records in energy usage month by month. Training a state-of-the-art image or language model like StyleGAN2-ada, GPT-3 or GLaM on highly energy-efficient GPU clusters has an estimated carbon footprint of about 154t, 611t and 217t CO2, respectively (at the global carbon intensity average of 475 gCO2/kWh).

Unfortunately, in practice, it is often simply not possible to collect all data in once central location for privacy reasons. As governments are pushing data protection regulations, there is an ever-growing need for systems that enable data processing directly at the data source, instead of a centralized server. This is why Google introduced Federated Learning (FL) in 2016.

Here, the training of a global machine learning model is distributed among many clients that train the same model on their own local data for a specified number of epochs.
The updated model parameters of each individual client's model are then collected and agglomerated by a centralized server before being sent back to each client. Repeating this process over and over again results in a final ML model that caries the wisdom of each client, e.g. performs well on the overall data, without ever having had to share data between clients or the central server. The training of each client on its local data is refered to as local training round, wheras the server-led process of first sending the global model parameters to the clients, then collecting each client's model parameter updates, and finally agglomerating the different client paramters into a single set of paramters, is refered to as communication round.\
Since the number of participating devices in such a setup could easily add up to multiple million of devices, e.g. mobile phones, most FL setups employ a client selection scheme that considers only a subset of participating clients in each FL training round. The used selection algorithms should optimize the two following objectives: [Time-to-accuracy](https://www.usenix.org/system/files/osdi21-lai.pdf) and [Fairness](https://arxiv.org/abs/2110.15545). While time-to-accuracy aims to prioritize clients that offer the greatest utility in improving overall model accuracy and tastest overall training time, fairness is concerned with ensuring unbiasedness of the model, that is, the final model should perform equally well on each clients local data and is not prone towards a solution of a group of clients that are over-represented in terms of their data or their overall participation in the training.

To wrap-up, FL's big advantage: Intead of sharing potentially large quantities of privacy sensitive data, we only share model parameters with a central server, while still benefitting from good quality ML models.\
The catch: While traditional centralized learning happens in highly efficient, highly optimized data centers, federated learning often relies on power-inefficient local clients (e.g. 600W consumer graphical processing units (GPU) in workstations for medical image analysis) and potentially more training iterations due to non-iid. local training data across the clients. Needlessly to say, this further worsens AI's carbon problem.

This is where our CarbonHack22's [Lowcarb](http://) project, a plugin for the federated learning framework Flower, comes into play.

## The Solution: Carbon-Aware Federated Machine Learning with [Lowcarb](http://)

[Flower](http:/https://flower.dev/) is a popular solution 
that brings the federated learning approach to established AI frameworks like PyTorch and TensorFlow. 
For CarbonHack22, we have developed an *all-batteries-included* plugin for the Flower framework to make it carbon-aware with less than 10 lines of code.

By focusing on the local renewable energy sources in each client's region, our plugin reduces the overall carbon footprint of the FL training, without loss of communication efficiency and client fairness.
<!--By including our Plugin, the federated learning is split up into training rounds, where only clients with the least present carbon footprint participate.
This reduces the overall carbon footprint of the training by focusing on the local renewable energy sources in each client's region.-->
We've reached an implementation, where the scheduling of training rounds and the carbon-aware sampling of clients is totally obfuscated from the user.
This takes away responsibility from developers and opens up carbon-aware federated machine learning to everybody!

In one example [we showcase later](#The-Impact:-Example-Application-of-Carbon-Aware-Federated-Learning-using-Lowcarb), including our plugin (less than 10 lines of code) reduced the training's energy-consumption by 14%.
We estimate that, depending on the use case, these 14% can materialize in millions of tons of CO<sub>2</sub>* that are saved by [Lowcarb](http://).

## Lowcarb and the Carbon-Aware-SDK

In most federated learning the individual clients are commonly distributed over many regions / power grids with different carbon intensity.
For each training round our plugin selects the clients with the smallest expected carbon emission when performing the individual workload on their respective power grid. (TODO: ADD FAIRNESS)

To derive the best selection yielding the lowest carbon footprint, we rely on Carbon-Awares-SDK's forecast of the marginal carbon intensity for all client regions.
For each training round our plugin pulls the forecast for the next 12 hours from a Carbon-Awares-SDK's WebApi that is specified by the user. (They can choose if they host it locally or rely on a cloud solution)
The selection of clients is then based on this forecast. 
There is more to the actual client selection algorithm, so if you're interested in all the nitty-gritty details, [we explain it in more details further down the line](#The-Carbon-Aware-Client-Sampling-Algorithm). 
What's important: the Carbon-Aware-SDK supplies our Lowcarb plugin with all the information it needs to do its job.

## The Impact: Example Application of Carbon-Aware Federated Learning using Lowcarb

To demonstrate the impact of the Lowcarb plugin when used with Carbon-Aware-SDK, we have developed an example Flower application for the training of a neutral net on the privacy-sensitive X-ray data for medical image recognition of thorax diseases using the NIH chest [X-Ray dataset](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest).
As a toy example, it assumes only 100 clients spread over 14 regions worldwide, but it can also demonstrate what carbon savings can be expected from Lowcarb.

Let's start with the results right away: by using Lowcarb, the federated learning example saved 14% in carbon emission over chosing clients randomly, without sacrificing training time, training accuracy, and client fairness.
<!--
while obtaining the same test accuracy and precision, as well as even client distribution.-->
To further emphasize the impact of Lowcarb we extrapolated these 14% to other federated learning szenarios of the near future.

_Insert extrapolated usecases here_
### Case study 1: Federated, autonomous driving fleet
Tesla’s Autopilot is currently one of the most advanced autonomous driving systems out there. During [this CVPR 2021 Workshop](https://blogs.nvidia.com/blog/2021/06/22/tesla-av-training-supercomputer-nvidia-a100-gpus/), their (now Ex-)Senior Diretor of AI, Andrej Karpathy, shed some light on the inner workings of this system, that we are going to use as a starting point.\
In essence, each car in their autonomous car fleet performs inference on real time camera data, using a very large [Visual Transformer Neural Network](https://arxiv.org/abs/2010.11929), the state-of-the art architecture for Deep Learning, which lives on an *on-board* GPU. Once they have collected $\approx$ 1 million data points, each consisting of 10-second video clips, with 36 frames per second ($\approx$ 1.5 PetaBytes of data), they (centrally) finetune their model on a total of 5760 of NVIDIA’s A100 80GB flagship GPUs, before distributing the updated model back to their fleet and restarting this whole process. Although they do not provide further information on needed training time, [this NVIDIA](https://developer.nvidia.com/blog/training-a-state-of-the-art-imagenet-1k-visual-transformer-model-using-nvidia-dgx-superpod/) source states, that training a basic Visual Transformer on 1 million images on eigth A100 80GB takes roughly 1 week $=$ 168 hours.\
In the following, we assume an over-simplified scenario where training time scales linearly with the number of GPUs and the number of data points. 
Moreover, we assume that in a FL setup with 10 clients, each holding a disjoint subset of 100k data points of the aforementioned data (simplified to one frame per data point), performing one communication round with every client training locally until convergence, before averaging the model parameters, results in an equally-well performing model, compared to a model trained on all the data at once. In such a scenario, it would take 168 hours * 8 / 10  $\approx$ 134 hours for each client to converge. Since the TDP of a NVIDIA A100 80GB PCIs is 300W, and assuming an average $\text{CO}_2$ intensity of $400 \frac{\text{gCO}_2}{kWh}$, this would result in $300\text{W} \times 134\text{h} \times 400 \frac{\text{gCO}_2}{kWh} = 16 \text{kgCO}_2$ produced by each car's GPU for one FL training round, or a total of $160\text{kgCO}_2$ produced by the entire fleet (assuming we constantly hit TDP).\
According to sources [[1]](https://www.researchandmarkets.com/reports/5206354/autonomous-vehicle-market-by-automation-level-by?utm_source=CI&utm_medium=PressRelease&utm_code=txvgwg&utm_campaign=1479104+-+Global+Outlook+for+the+Autonomous+Vehicle+Market+to+2030+-+Sale+of+Autonomous+Vehicles+is+Forecast+to+Reach+58+Million+Units+by+2030&utm_exec=Rcent%20cari18prd), [[2]](https://www.statista.com/press/p/autonomous_cars_2020/) & [[3]](https://www.sciencedirect.com/science/article/pii/S0960982216303414#:~:text=Summary,Michael%20Gross%20reports.), by 2030, there could be somewhere between 60-150 million fully autonomous vehicles on the roads, worldwide. Scaling the case above on this amount of cars would result in at least $6 \text{million} * 160 \text{kgCO}_2  \approx 160.000$ $\text{tCO}_2$ emissision, and thus $160.000 \text{tCO}_2 * 0.14 \approx 22.400 \text{tCO}_2$  savings out of the box by using our approach. That corresponds to the $\text{CO}_2$ emitted from approximately 11.000 longhaul roundtrip flights from [Frankfurt to New York City](https://co2.myclimate.org/en/portfolios?calculation_id=5201579).\
Note: We are aware that many of the before made assumptions, including the overall consent of major car manufacturers to participate in such a training setup or the TDP of the used GPUs, are assumptions favoring higher $\text{CO}_2$ emissions and thus savings. \
On the other hand, however, we excluded the pre-training phase of these models, the size and complexity of the used data set, the longer training runs due to non-identically distributed data over clients and local model divergence, the hyperparameter optimization which results in many more copies of the same training runs, and other factors in our calculation, which easily compensate and even favor lower emissions and thus savings.

### Case study 2: Federated Learning in Hospitals?

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

In the bigger picture, with federated learning's increasing popularity and adoption, 
we think our Lowcarb plugin is a huge opportunity for the Carbon-Aware-SDK and carbon-aware software in general 
to reach a broader audience and raise awareness for this urgent climate issue.

## Technical Details and Toy Example
### Problem description
We examine the problem of training a machine learning (ML) model for thorax disease classification on chest X-Ray images, that are collected from patients of 100 hospitals, in 14 regions, around the world. Since this data cannot be shared across the different entities because of data protection and bandwidth reasons (large amount of images produced by each hospital), we resolved to train our model using the framework of Federated Learning (FL). In essence, it includes thefollowing three steps:

1. Train the same copy of a ML model on each entities's data set for a set number of iterations
2. Send the locally updated weights to a central server and average them
3. Send the updated model back to each entity

Due to the possibly larger number of participating clients, most FL applications use a <span style="color:red">random selection</span> of clients for each training round, irrespective of their access to clean energy wrt. $\text{CO}_2$ emissions.

Including our Plugin, Lowcarb, into the FL training pipeline, results in a <span style="color:lightgreen">carbon-aware selection</span> of clients that significantly reduces the Carbon emissions produced during the FL training, while maintining test performance, training speed, and client fairness. On this small test example we were able to reduce the $\text{CO}_2$ emissions by 14%. Plots of test losses for the random vs. carbon-aware client selection based FL training are included below.
(ToDo: Include plots of both test loss curves vs iteration, and training times)

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
