from Methods.utils.meta_methods import SeverMethod

class ShanFLSever(SeverMethod):
    NAME = 'ShanFLSever'

    def __init__(self, global_model, cfg):
        super().__init__(global_model, cfg)

    def sever_update(self, **kwargs):
        if not hasattr(self, 'online_clients_list') or not self.online_clients_list:
            print("Warning: online_clients_list is not set or empty in sever_update. Skipping trust score update and using equal weighting.")
            self.aggregation_weight_list = [1.0 / self.parti_num] * self.parti_num if self.parti_num > 0 else []
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
