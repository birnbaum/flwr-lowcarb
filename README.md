# Carbon Hack 22

Links:
- [Our project](https://taikai.network/en/gsf/hackathons/carbonhack22/projects/cl8wyghoi75233101wux9vigobp/idea) 
- [Carbon Hack Discord](https://discord.com/channels/1009739251761565696/1016261144803016744)


## 2 Minute Video

- Explain why FL great use case for Carbon-Aware SDK
  - SDK usage on two dimensions: Space and time
  - Motivate scale of problem (Impact)
- Idee und Scheduler vorstellen (Evtl 2 verschiedene oder parametrisierbare)
  - Kurz und abstrakt, ok wenn man Leute mal für ne Sekunde abhängt, weil krass
- Evaluation that results in crazy CO2 reduction
  - Impact betonen (2 Unternehmen benutzen Lowcarb=xyz g CO2 Einsparung  vs. alle Big Tech Unternehmen trainieren nurnoch mit lowcarb=xyz 10^100 g CO2 Einsparung)
- Mention ready-to-use Flower plugin (Feasibility)
- Vision (inklusiv welche weiteren Schritte wir nach dem PoC gehen würden etc.)


## Concept

- Custom client who communicates its location
- Client strategy/manager that queries the Carbon-Aware SDK


## Installation

```
python -m venv venv              # create venv
. venv/bin/activate               # activate venv on Linux/Mac
.\venv\Scripts\activate           # activate venv on Windows
pip install -r requirements.txt  # install dependencies
```


# API Feedback

- Python client didn't build 
  - Exception in thread "main" java.lang.RuntimeException: Could not generate model 'CarbonIntensityBatchParametersDTO'
  - Caused by: com.github.jknack.handlebars.HandlebarsException: model.handlebars:3:3: java.lang.reflect.InaccessibleObjectException: Unable to make field transient java.util.HashMap$Node[] java.util.HashMap.table accessible: module java.base does not "opens java.util" to unnamed module @25fb0107
  - Caused by: java.lang.reflect.InaccessibleObjectException: Unable to make field transient java.util.HashMap$Node[] java.util.HashMap.table accessible: module java.base does not "opens java.util" to unnamed module @25fb0107
- api_instance.get_emissions_data_for_locations_by_time returns watttime locations, so it's impossible to infer the original locations# Pitch


# Pitch

### Intro (3s)

> Slide 1
> Carbon-Aware Federated Learning
> Photos, Names, (Affiliation?)

### ML Energy usage (10s)

> Slide 2
> - StyleGAN2-ADA [1]: 325 MWh
> - GPT-3 [2]: 1287 MWh -> 1287 tons of CO2
> - GLaM [2]: 456 MWh

The energy consumption of AI is going through the roof

- The training of modern machine learning models consumes hundreds of megawatt-hours of energy resulting in hundreds of tons of carbon emissions

### FL Overview 30s

> Slide 3
> FL overview
> Figure of datacenter top and many data sources at the bottom that send their data


- ... although such models are usually trained in highly efficient data centers
- Unfortunately, this problem will become more serious soon, since in practice it is often not possible to collect all data one central location for privacy reasons
- This is why Google introduced Federated Learning in 2016 where a machine learning model is being trained directly on end devices without ever transmitting any data.

The problem is, these trainings take longer and end devices are usually less efficient, so energy usage is expected to further increase


### Vision/Scale/Impact (20s) (26s)

- Use cases for federated learning range from recommender systems on smartphones, over IoT applications, healthcare, banking, satelites in space, till autonomous vehicles.
- Imagine millions of cars with lots of computing power required for autonomous driving that generate Terrabytes of data every day
- When parked and charging, they can improve a common model in a privacy-aware manner
- ... while consuming huge amounts of energy


### Solution 20s (20s)

> dramatic pause

What if we could make these trainings carbon-aware?
> type: pip install flwr-lowcarbon

We developed a plugin for the popular federated learning framework Flower,
which does exactly that:
- Instead of randomly selecting clients on every training round, it picks clients, that are currently - according to forecasts provided by the Carbon Aware SDK - in a time window of low carbon intensity.


### Evaluation 30s (35s)

We evaluated our approach on a concrete use case:
- 100 hospitals distributed over 14 locations want to train a common model for detecting thorax diseases.
- however, for data privacy reasons they cannot share their patients xrays so they go for a federated learning approach.

We performed this training twice on real data during the same 4-day period:
- Once with the default random client selection that comes with Flower and
- once with our plugin activated

The plugin managed to reduce the carbon emissions out-of-the-box by 13%!
Without any sacrifices on
- training duration
- model accuracy
- and without introducing unfairness in client selection


### Future 6s (check)

Our prototype is publicly available, open source, and ready to use.
Try it out now.

> Last Slide
> GreenFL - Privacy-aware and carbon aware AI


[1] https://arxiv.org/pdf/2006.06676.pdf
[2] https://arxiv.org/pdf/2112.06905.pdf
