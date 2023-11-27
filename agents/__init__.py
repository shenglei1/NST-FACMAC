REGISTRY = {}

from .mlp_agent import MLPAgent
from .mlp_agent1 import MLPAgent1
from .rnn_agent import RNNAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["mlp1"] = MLPAgent1
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
