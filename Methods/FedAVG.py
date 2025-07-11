from Methods.utils.meta_methods import FederatedMethod

class FedAVG(FederatedMethod):
    NAME = 'FedAVG'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedAVG, self).__init__(nets_list, client_domain_list, args, cfg)

    def ini(self):
        super().ini()

    def local_update(self, priloader_list, losses=None):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        # Loop through only the online clients to perform their local updates
        for idx in self.online_clients_list:
            self.nets_list[idx].train() # Set the client model to training mode

            kwargs_for_loc_update = {
                'online_clients_list': self.online_clients_list,
                'nets_list': self.nets_list,
                'priloader_list': priloader_list, # Pass the full list of loaders
            }

            # Only add 'losses' if it's not None
            if losses is not None:
                kwargs_for_loc_update['losses'] = losses

            self.local_model.loc_update(**kwargs_for_loc_update)

    def sever_update(self, **kwargs):
        global_net = kwargs.get('global_net', self.global_net) # Get global_net, with fallback
        priloader_list = kwargs.get('train_loaders') # training.py passes it as 'train_loaders'
        online_clients_list = kwargs.get('online_clients_list', self.online_clients_list)
        nets_list = kwargs.get('nets_list', self.nets_list) 

        self.aggregation_weight_list = self.sever_model.sever_update(
            fed_aggregation=self.fed_aggregation,
            online_clients_list=online_clients_list,
            priloader_list=priloader_list,
            client_domain_list=self.client_domain_list,
            global_net=global_net,
            nets_list=nets_list
        )
