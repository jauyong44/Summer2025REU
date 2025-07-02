import copy
import torch
import torch.nn as nn
import numpy as np
from Methods.utils.meta_methods import SeverMethod

class ShanFLSever(SeverMethod):
    NAME = 'ShanFLSever'

    def __init__(self, **kwargs):
        global_model = kwargs.get('global_model')
        cfg = kwargs.get('cfg')
        if global_model is None or cfg is None:
            raise ValueError("ShanFLSever.__init__ requires 'global_model' and 'cfg' in kwargs.")
        super().__init__(global_model, cfg) # Pass them to SeverMethod.__init__
        self.parti_num = cfg.DATASET.parti_num # Get participant number from global config
        self.trust_scores = {i: cfg.ShanFL.initial_trust_score for i in range(self.parti_num)}
        self.previous_client_model_diffs = {i: None for i in range(self.parti_num)}
        self.consistency_threshold = cfg.ShanFL.consistency_threshold
        self.trust_increase_factor = cfg.ShanFL.trust_increase_factor
        self.trust_decrease_factor = cfg.ShanFL.trust_decrease_factor
        self.min_trust_score = cfg.ShanFL.min_trust_score
        self.max_trust_score = cfg.ShanFL.max_trust_score
        self.global_model_at_round_start = None # This will be updated by the orchestrator (e.g., FederatedMethod.ini or before sever_update call)
        
        # These will be passed into the sever_update method directly
        self.nets_list = None
        self.online_clients_list = None
        self.aggregation_weight_list = None
        self.epoch_index = 0 # Or passed via kwargs to sever_update

    def sever_update(self, **kwargs):
        self.online_clients_list = kwargs.get('online_clients_list', self.online_clients_list)
        self.nets_list = kwargs.get('nets_list', self.nets_list)
        self.epoch_index = kwargs.get('epoch_index', self.epoch_index) # Assume epoch_index is updated externally or incremented
        
        if self.global_model_at_round_start is None:
            # Fallback or error if not set, or initialize to current global_net state
            self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())

        if not self.online_clients_list: # Simplified check
            print("Warning: online_clients_list is empty in sever_update. Skipping trust score update and using equal weighting.")
            # self.aggregation_weight_list = [1.0 / self.parti_num] * self.parti_num if self.parti_num > 0 else [] # This line already uses self.parti_num
            self._perform_simple_average_aggregation()
            # Reset global model at round start for next epoch
            self.global_model_at_round_start = copy.deepcopy(self.global_net.state_dict())
            return

        # 1. Calculate Current Local Model Differences (using global_model_at_round_start)
        local_model_diffs_current_round = {}
        for idx in self.online_clients_list:
            local_net_para = self.nets_list[idx].state_dict()
            diff = {name: local_net_para[name] - self.global_model_at_round_start[name]
                    for name in self.global_model_at_round_start}
            local_model_diffs_current_round[idx] = diff

        # 2. Update Trust Scores based on Consistency (Cosine Similarity of Diffs)
        for client_id in self.online_clients_list:
            current_update_flat = self._flatten_parameters(local_model_diffs_current_round[client_id])

            # Only compare if there's a previous update diff available
            if self.previous_client_model_diffs[client_id] is not None:
                previous_update_flat = self.previous_client_model_diffs[client_id]

                # Handle cases where update vectors might be zero (if model didn't change much)
                if torch.norm(current_update_flat) == 0 or torch.norm(previous_update_flat) == 0:
                    cos_sim = 0.0 # Treat as inconsistent if no movement
                else:
                    cos_sim = nn.functional.cosine_similarity(current_update_flat, previous_update_flat, dim=0).item()

                # Adjust trust score using multiplicative factors
                if cos_sim > self.consistency_threshold:
                    self.trust_scores[client_id] *= self.trust_increase_factor
                else:
                    self.trust_scores[client_id] *= self.trust_decrease_factor
            else:
                # First time seeing this client's update or no previous update stored
                # Trust score remains at initial value
                pass

            # Clamp trust score within [0, 1] bounds
            self.trust_scores[client_id] = np.clip(self.trust_scores[client_id], self.min_trust_score, self.max_trust_score)

            # Store current update diff for next epoch's comparison
            self.previous_client_model_diffs[client_id] = current_update_flat.clone().detach()

        # 3. Calculate Weighted Average based on Trust Scores
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

        # 4. Apply Weighted Aggregation to Update Global Model
        global_state = self.global_net.state_dict()

        for k in global_state:
            global_state[k] = torch.zeros_like(global_state[k])

        for i, client_id in enumerate(self.online_clients_list):
            weight = normalized_weights[i]
            client_net_para = self.nets_list[client_id].state_dict()
            for k in global_state:
                global_state[k] += weight * client_net_para[k]

        self.global_net.load_state_dict(global_state)

        # Reset global model at round start for next epoch
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
        if num_online_clients == 0: return # This will cause an error if used as num_online_clients / 0.0

        for client_id in self.online_clients_list:
            client_net_para = self.nets_list[client_id].state_dict()
            for k in global_state:
                global_state[k] += client_net_para[k] / num_online_clients

        self.global_net.load_state_dict(global_state)
