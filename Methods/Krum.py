from Methods.utils.meta_methods import FederatedMethod

class Krum(FederatedMethod):
    NAME = 'Krum'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(Krum, self).__init__(nets_list, client_domain_list, args, cfg)

    def ini(self):
        super().ini()

    def local_update(self, priloader_list, losses=None):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        for idx in self.online_clients_list:
            self.nets_list[idx].train()

            kwargs_for_loc_update = {
                'online_clients_list': self.online_clients_list,
                'nets_list': self.nets_list,
                'global_net': self.global_net,
                'priloader_list': priloader_list,
            }
            if losses is not None:
                kwargs_for_loc_update['losses'] = losses

            self.local_model.loc_update(**kwargs_for_loc_update)

    def sever_update(self, **kwargs):
        # Pass all relevant data to KrumServer's sever_update
        self.aggregation_weight_list = self.sever_model.sever_update(
            nets_list=self.nets_list,
            online_clients_list=self.online_clients_list,
            global_net=self.global_net,
            epoch_index=self.epoch_index
        )
