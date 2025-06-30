import copy
import torch
import torch.nn as nn
from Methods.utils.meta_methods import FederatedMethod
import numpy as np

class ShanFL(FederatedMethod):
  NAME = 'ShanFL'
  COMPATIBILITY = ['homogeneity', 'heterogeneity']

  def __init__(self, nets_lit, client_domain_list, args, cfg):
    super().__init__(nets_list, client_domain_list, args, cfg)
    # initialize trust scores for all potential clients 
    self.trust_score = {i: cfg.FEDTRUST.initial_trust_score for i in range(self.parti_num)}
    # store the flattened model difference of the last participation for each client
    # (client_model_at_round_X - global_model_at_start_of_round_X)
    self.previous_client_model_diffs = {i: None for i in range(self.parti_num)} # initialize at None
    # hyperparameters for trust score adjustment, loaded from config
    self.consistency_threshold = cfg.FEDTRUST.consistency_threshold
    self.trust_increase_factor = cfg.FEDTRUST.trust_increase_factor
    self.trust_decrease_factor = cfg.FEDTRUST.trust_decrease_factor
    self.min_trust_score = 0.0
    self.max_trust_score = 1.0
    # store global model state at the start of the current round
    # needed to calculate the current model difference for each client
    self.global_model_at_round_start = None

  # method called from training.py before local_update
  def ini(self):
    super().ini()
    self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())

  def _flatten_parameters(self, params_dict):
      """Flattens a dictionary of model parameters into a single tensor."""
      tensors = [v.flatten() for v in params_dict.values() if isinstance(v, torch.Tensor)]
      if not tensors:
          return torch.tensor([])
      return torch.cat(tensors)

  def _perform_simple_average_aggregation(self):
      """Fallback aggregation method if trust scores are problematic or no online clients."""
      if not self.online_clients_list:
          return

      global_state = self.global_net.state_dict()
      for k in global_state:
          global_state[k] = torch.zeros_like(global_state[k])

      num_online_clients = len(self.online_clients_list)
      if num_online_clients == 0: return

      for client_id in self.online_clients_list:
          client_net_para = self.nets_list[client_id].state_dict()
          for k in global_state:
              global_state[k] += client_net_para[k] / num_online_clients

      self.global_net.load_state_dict(global_state)
