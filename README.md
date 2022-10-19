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
- api_instance.get_emissions_data_for_locations_by_time returns watttime locations, so it's impossible to infer the original locations