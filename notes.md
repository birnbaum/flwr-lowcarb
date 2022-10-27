

# API Feedback

- Python client didn't build 
  - Exception in thread "main" java.lang.RuntimeException: Could not generate model 'CarbonIntensityBatchParametersDTO'
  - Caused by: com.github.jknack.handlebars.HandlebarsException: model.handlebars:3:3: java.lang.reflect.InaccessibleObjectException: Unable to make field transient java.util.HashMap$Node[] java.util.HashMap.table accessible: module java.base does not "opens java.util" to unnamed module @25fb0107
  - Caused by: java.lang.reflect.InaccessibleObjectException: Unable to make field transient java.util.HashMap$Node[] java.util.HashMap.table accessible: module java.base does not "opens java.util" to unnamed module @25fb0107
- api_instance.get_emissions_data_for_locations_by_time returns watttime locations, so it's impossible to infer the original locations
- 


---

The energy consumption of training machine learning models is going through the roof:
- Recent machine learning models consume hundreds of megawatt-hours for training and the number of parameters per model are currently doubling every couple of months.
- For example, GPT-3, one of most popular models for natural language tasks today,
  took around 1300 MWh of energy to train.
- Assuming this energy came from coal, that's more than 1000 tons of carbon emissions

Note that we still want to perform more or less the same kind of work at every individual client to not introduces biases towards clients with low carbon intensity. We simply choose time windows in which it is low for the client to train.


- Large ML models are usually trained in highly energy-efficient data centers with hundreds of GPUs and all data in one place.
- However, in practice, for many applications we cannot collect all data into one centralized location for data privacy reasons.
- For this, Federated Learning was introduced by Google in 2016:
  - instead of collecting data to a centralized location, the machine learning model is distributed to a subset of all available clients, they locally train on their data, and simply sent back the updated model
  - All model updates are aggregated on the server, a new subset of clients is selected and the next round starts.

The big benefit of this appraoch is, that no data ever leaves the devices.
The big problem is, we are training on infrastructure that is a lot less energy-efficient than modern GPUs, so energy use will rise further

 based on a public dataset consisting of 100000 images of chest x-rays

Imagine training a huge language model, such as GPT-3,
but instead of collecting all data from users
you train decentralized to respect their privacy.

In Federated Learning, we do exactly that: On every training round,
we select a subset of clients in the system, train the model on their local
data, and only send back the updated model. No data ever leaves the device    
and we are training decentralized on millions of smartphones, cars, etc.

https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest


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

We tested our approach on a concrete use case:
- 100 hospitals distributed over 14 locations want to train a common model for detecting thorax diseases.
- however, for data privacy reasons they cannot share their patients xrays so they go for a federated learning approach.

We performed this training twice on real data during the same 4-day period:
- Once with the default random client selection that comes with Flower and
- once with our plugin activated

which managed to reduce the carbon emissions out-of-the-box by 13%!
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


### Slide sources

- StyleGAN-ADA: https://arxiv.org/pdf/2006.06676.pdf
- GPT-3: https://arxiv.org/pdf/2005.14165.pdf
- GLaM: https://arxiv.org/pdf/2112.06905.pdf
- Gboard: https://arxiv.org/pdf/1811.03604.pdf
- fedspace: https://arxiv.org/pdf/2202.01267.pdf
