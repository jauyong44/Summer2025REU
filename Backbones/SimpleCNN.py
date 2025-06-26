import torch # https://pytorch.org/docs/stable/torch.html
import torch.nn as nn # https://pytorch.org/docs/stable/nn.html
import torch.nn.functional as F # https://pytorch.org/docs/stable/nn.functional.html
from torch import distributions # https://pytorch.org/docs/stable/distributions.html
from utils.utils import log_msg
from Backbones.utils import ACTIVATION_FUNCTIONS

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, cfg, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        act_type = cfg.MODEL.activation_type
        self.activation = ACTIVATION_FUNCTIONS[act_type]()
        

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


class SimpleCNN(nn.Module):

    def __init__(self, cfg):
        super(SimpleCNN, self).__init__()
        self.name = 'SimpleCNN'
        self.act_type = cfg.MODEL.activation_type
        self.feats = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], cfg=cfg, output_dim=cfg.DATASET.n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, 256)

        # last layer
        self.cls = nn.Linear(256, cfg.DATASET.n_classes)

        self.activation = ACTIVATION_FUNCTIONS[self.act_type]()
        if cfg.MODEL.CNN.init_weights:
            self.apply(self._init_weights)

    def _init_weights(self,m):
        nonlinearity_arg = self.act_type.lower()
        if isinstance(m, nn.Conv2d):
            if self.act_type in ['ReLU', 'Leaky_ReLU']:
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity_arg)
            else:
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity_arg))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if self.act_type in ['ReLU', 'Leaky_ReLU']:
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity_arg)
            else:
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity_arg))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        h = h.squeeze()
        h = self.l1(h)
        h = self.activation(h)
        h = self.l2(h)
        return h

    def classifier(self, h: torch.Tensor) -> torch.Tensor:
        y = self.cls(h)
        return y

    def forward(self, x):
        # h = self.feats(x)
        # h = h.squeeze()
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        x = self.features(x)

        y = self.cls(x)
        return y

class SimpleCNN_sr(nn.Module):

    def __init__(self, cfg):
        super(SimpleCNN_sr, self).__init__()
        self.name = 'SimpleCNN'
        self.feats = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], cfg=cfg, output_dim=cfg.DATASET.n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, 256)

        self.mlp = nn.Linear(256, 512)

        # last layer
        self.cls = nn.Linear(256, cfg.DATASET.n_classes)

        self.act_type = cfg.MODEL.activation_type
        self.activation = ACTIVATION_FUNCTIONS[act_type]()
        if cfg.MODEL.CNN.init_weights:
            self.apply(self._init_weights)

    def _init_weights(self,m):
        nonlinearity_arg = self.act_type.lower()
        if isinstance(m, nn.Conv2d):
            if self.act_type in ['ReLU', 'Leaky_ReLU']:
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity_arg)
            else:
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity_arg))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if self.act_type in ['ReLU', 'Leaky_ReLU']:
                nn.init.kaiming_normal_(m.weight,nonlinearity=nonlinearity_arg)
            else:
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity_arg))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        h = h.squeeze()
        h = self.l1(h)
        h = self.activation(h)
        h = self.l2(h)
        return h

    def classifier(self, h: torch.Tensor) -> torch.Tensor:
        y = self.cls(h)
        return y

    def featurize(self, x, num_samples=1, return_dist=False):

        features = self.features(x)
        z_params = self.mlp(features)

        z_mu = z_params[:, :self.cls.in_features]
        z_sigma = F.softplus(z_params[:, self.cls.in_features:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        # torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims, validate_args=None)
        # Independent: reinterprets some of the batch dims of a distribution as event dims.

        z = z_dist.rsample([num_samples]).view([-1, self.cls.in_features])
        # rsample(sample_shape=torch.Size([]))
        
        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        if self.training:

            x = self.featurize(x, return_dist=False)
            y = self.cls(x)
        else:
            x = self.featurize(x, num_samples=self.num_samples, return_dist=False)
            preds = torch.softmax(self.cls(x), dim=1)
            '''
            # torch.softmax(input, dim, *, dtype=None) → Tensor
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax
            # softmax function ^
            # It is applied to all slices along dim, and will re-scale 
            them so that the elements lie in the range [0, 1] and sum to 1.
            # Note: This function doesn’t work directly with NLLLoss, 
            which expects the Log to be computed between the Softmax 
            and itself. Use log_softmax instead (it’s faster and has 
            better numerical properties).
            '''

            preds = preds.view([self.num_samples, -1, self.cls.out_features]).mean(0)
            # preds: are these the predecessors?
            y = torch.log(preds)
            # https://pytorch.org/docs/stable/generated/torch.log.html#torch.log
            # Returns a new tensor with the natural logarithm of the elements of input.

        return y
