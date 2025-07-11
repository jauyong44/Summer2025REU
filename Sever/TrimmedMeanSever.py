import copy

import numpy as np
import torch

from Backbones import get_private_backbones
from Sever.utils.sever_methods import SeverMethod
from Sever.utils.utils import trimmed_mean
from utils.utils import row_into_parameters, log_msg 


class TrimmedMeanSever(SeverMethod):
    NAME = 'TrimmedMeanSever'

    def __init__(self, args, cfg):
        super(TrimmedMeanSever, self).__init__(args, cfg)

        nets_list = get_private_backbones(cfg)

        self.momentum = 0.9
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            self.current_weights.append(param)
        self.current_weights = torch.cat(self.current_weights, dim=0).cpu().numpy()
        self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
        self.n = 5
        self.k_value_override = getattr(cfg.Sever[self.NAME], 'k_value', None)

    def sever_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():
            all_grads = []
            for i in online_clients_list:
                grads = {}
                net_all_grads = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    grads[name] = (param0.detach() - param1.detach()) / self.learning_rate
                    net_all_grads.append(copy.deepcopy(grads[name].view(-1)))

                net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
                all_grads.append(net_all_grads)
            all_grads = np.array(all_grads)

        # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
        if self.k_value_override is not None:
            k = self.k_value_override
            if (k>=len(online_clients_list)):
                raise ValueError(f"Given a k value of {k} but we only have a online client size of {len(online_clients_list)}")
            print(log_msg(f"We are using the override and have a value of {k}"))
        else:
            f = len(online_clients_list) // 2  # worse case 50% malicious points
            k = len(online_clients_list) - f - 1
            print(log_msg(f"We are using default k calculated from F and have a value of {k}"))

        current_grads = trimmed_mean(all_grads, len(online_clients_list), k)

        self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
        self.current_weights += self.velocity

        row_into_parameters(self.current_weights, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

