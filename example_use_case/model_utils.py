from typing import Dict, List
from collections import OrderedDict
import numpy as np
import torch
from torchvision import models
import flwr as fl
import train_eval_utils

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        client_xray_df = config["client_xray_df"]

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train_eval_utils.train(self.net, self.trainloader, epochs=local_epochs, client_xray_df=client_xray_df)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        client_xray_df = config["client_xray_df"]
        set_parameters(self.net, parameters)
        loss, accuracy = train_eval_utils.test(self.net, self.valloader, client_xray_df=client_xray_df)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def init_net():
  net = models.vgg11(
    weights="IMAGENET1K_V1"
  )
  net.classifier[-1] = torch.nn.Linear(
      net.classifier[-1].in_features,
      13
  )
  return net

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)