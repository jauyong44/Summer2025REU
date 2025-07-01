import copy
import torch
import torch.nn as nn
from Methods.utils.meta_methods import FederatedMethod
import numpy as np

class ShanFL(FederatedMethod):
  NAME = 'ShanFL'
  COMPATIBILITY = ['homogeneity', 'heterogeneity']

  def __init__(self, nets_list, client_domain_list, args, cfg):
    super(ShanFL, self).__init__(nets_list, client_domain_list, args, cfg)
    pass

  # method called from training.py before local_update
  def ini(self):
    super().ini()
    self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())
