import copy
import torch
import torch.nn as nn
from Methods.utils.meta_methods import FederatedMethod
import numpy as np
import random

class TRUFL(FederatedMethod):
  NAME = 'TRUFL'
  COMPATIBILITY = ['homogeneity', 'heterogeneity']

  def __init__(self, nets_list, client_domain_list, args, cfg):
    super(TRUFL, self).__init__(nets_list, client_domain_list, args, cfg)

  def ini(self):
    super().ini()
    # after global_net is initializaed, set it on the sever_model for diff calculation
    if self.sever_model is not None:
      self.sever_model.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())

  def local_update(self, priloader_list):
    total_clients = list(range(self.cfg.DATASET.parti_num))
    self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

    self.local_model.loc_update(online_clients_list=self.online_clients_list, nets_list=self.nets_list, global_net=self.global_net, 
                                priloader_list=priloader_list, epoch_index=self.epoch_index)

  def sever_update(self, priloader_list):
    self.sever_model.sever_update(nets_list=self.nets_list, online_clients_list=self.online_clients_list,
                                  epoch_index=self.epoch_index)
