import copy
import torch
import torch.nn as nn
import numpy as np
from Sever.utils.sever_methods import SeverMethod

class TRUFLSever(SeverMethod):
    NAME = 'TRUFLSever'

    def __init__(self, args, cfg):
        super(TRUFLSever, self).__init__(args, cfg)

        # Initialize TRUFL-specific server-side state from config
        # Parameters defined under CFG.TRUFL in cfg.py
        self.parti_num = cfg.DATASET.parti_num # Total number of participants
        self.trust_scores = {i: cfg.TRUFL.initial_trust_score for i in range(self.parti_num)}
        self.previous_client_model_diffs = {i: None for i in range(self.parti_num)}
        self.consistency_threshold = cfg.TRUFL.consistency_threshold
        self.trust_increase_factor = cfg.TRUFL.trust_increase_factor
        self.trust_decrease_factor = cfg.TRUFL.trust_decrease_factor
        self.min_trust_score = cfg.TRUFL.min_trust_score
        self.max_trust_score = cfg.TRUFL.max_trust_score

        # Attributes that will be passed via kwargs to sever_update
        self.nets_list = None
        self.online_clients_list = None
        self.aggregation_weight_list = None
        self.epoch_index = 0 # Will be updated by the orchestrator (training.py)

        # global_model_at_round_start is crucial for diff calculation
        self.global_model_at_round_start = None 

    def sever_update(self, **kwargs):
        # Retrieve necessary data from kwargs
        self.online_clients_list = kwargs.get('online_clients_list', self.online_clients_list)
        self.nets_list = kwargs.get('nets_list', self.nets_list)
        self.epoch_index = kwargs.get('epoch_index', self.epoch_index)
        # train_loaders might be passed but not directly used in aggregation logic here
        # train_loaders = kwargs.get('train_loaders') 
        
        # Ensure global_model_at_round_start is set
        if self.global_model_at_round_start is None:
            print("Warning: global_model_at_round_start not set. Initializing with current global_net state.")
            self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())

        if not self.online_clients_list:
            print("Warning: online_clients_list is empty in sever_update. Falling back to equal weighting.")
            if self.parti_num > 0:
                self.aggregation_weight_list = [1.0 / self.parti_num] * self.parti_num
            else:
                self.aggregation_weight_list = []
            self._perform_simple_average_aggregation() # Fallback
            # Reset global model at round start for next epoch
            self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())
            return

        # calculate current local model differences (using global_model_at_round_start)
        local_model_diffs_current_round = {}
        for idx in self.online_clients_list:
            local_net_para = self.nets_list[idx].state_dict()
            diff = {name: local_net_para[name] - self.global_model_at_round_start[name]
                    for name in self.global_model_at_round_start}
            local_model_diffs_current_round[idx] = diff

        # update trust scores based on consistency (cosine similarity)
        for client_id in self.online_clients_list:
            current_update_flat = self._flatten_parameters(local_model_diffs_current_round[client_id])
            if self.previous_client_model_diffs[client_id] is not None:
                previous_update_flat = self.previous_client_model_diffs[client_id]
                if torch.norm(current_update_flat) == 0 or torch.norm(previous_update_flat) == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = nn.functional.cosine_similarity(current_update_flat, previous_update_flat, dim=0).item()
                if cos_sim > self.consistency_threshold:
                    self.trust_scores[client_id] *= self.trust_increase_factor
                else:
                    self.trust_scores[client_id] *= self.trust_decrease_factor
            else:
                pass # Trust score remains at initial value if no previous update
            self.trust_scores[client_id] = np.clip(self.trust_scores[client_id], self.min_trust_score, self.max_trust_score)
            self.previous_client_model_diffs[client_id] = current_update_flat.clone().detach()

        # calculate weighted average based on trust scores
        active_trust_scores = [self.trust_scores[idx] for idx in self.online_clients_list]
        total_active_trust = sum(active_trust_scores)
        if total_active_trust == 0:
            print("Warning: Total trust score of online clients is zero. Falling back to equal weighting for aggregation.")
            normalized_weights = [1.0 / len(self.online_clients_list)] * len(self.online_clients_list)
        else:
            normalized_weights = [score / total_active_trust for score in active_trust_scores]
        trust_log_str = ", ".join([f"Client {idx}: {self.trust_scores[idx]:.4f}" for idx in self.online_clients_list])
        weight_log_str = ", ".join([f"Client {self.online_clients_list[i]}: {normalized_weights[i]:.4f}" for i in range(len(normalized_weights))])
        print(f"Epoch {self.epoch_index} Online Trust Scores: {trust_log_str}")
        print(f"Epoch {self.epoch_index} Aggregation Weights: {weight_log_str}")

        self.aggregation_weight_list = normalized_weights

        # weighted aggregation to update global model
        global_state = self.global_net.state_dict()
        for k in global_state:
            global_state[k] = torch.zeros_like(global_state[k])

        for i, client_id in enumerate(self.online_clients_list):
            weight = normalized_weights[i]
            client_net_para = self.nets_list[client_id].state_dict()
            for k in global_state:
                global_state[k] += weight * client_net_para[k]

        self.global_net.load_state_dict(global_state)

        # reset global model at round start for next epoch
        self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())
    
    def _flatten_parameters(self, params_dict):
        """Flattens a dictionary of model parameters into a single tensor"""
        tensors = [v.flatten() for v in params_dict.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            return torch.tensor([])
        return torch.cat(tensors)

    def _perform_simple_average_aggregation(self):
        """Fallback aggregation method if trust scores are problematic or no online clients"""
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
