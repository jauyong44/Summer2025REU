import copy
import torch
# This function calculates the parameter difference of the online clients to the global model.
# Used for gradient inversion attacks
def get_parameter_difference(fed_method, global_net, selected_client_indices):
    """
    Calculates the parameter difference (local_model - global_model_before_update)
    for a subset of clients.
    """
    
    global_net_copy = copy.deepcopy(global_net)
    all_clients_param_differences = []
    with torch.no_grad():
        for client_idx in selected_client_indices:
            client_net = fed_method.nets_list[client_idx]
            current_client_diffs_for_all_params = []
            for name, global_param in global_net_copy.state_dict().items():
                local_param = client_net.state_dict()[name]
                diff = local_param.detach() - global_param.detach()
                current_client_diffs_for_all_params.append(diff)
            
            # Append the list of multi-dimensional tensors for the current client
            all_clients_param_differences.append(current_client_diffs_for_all_params)
    return all_clients_param_differences
