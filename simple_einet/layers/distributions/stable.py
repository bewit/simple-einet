from simple_einet.sampling_utils import SamplingContext
import torch
from torch import nn
from torch.distributions.distribution import Distribution

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from stable_distribution import TorchStable

class Stable(AbstractLeaf):
    """Alphastable layer"""

    def __init__(
            self,
            num_features: int, 
            num_channels: int, 
            num_leaves: int,
            num_repetitions: int
    ):
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # these are defined in the logspace
        _alpha = torch.rand(1, num_channels, num_features, num_leaves, num_repetitions) + 1.0
        _beta = torch.rand(1, num_channels, num_features, num_leaves, num_repetitions)
        _loc = torch.randn(1, num_channels, num_features, num_leaves, num_repetitions)
        _scale = torch.rand(1, num_channels, num_features, num_leaves, num_repetitions)
        self.alpha = nn.Parameter(_alpha)
        self.beta = nn.Parameter(_beta)
        self.loc = nn.Parameter(_loc)
        self.scale = nn.Parameter(_scale)


    def _get_base_distribution(self, ctx: SamplingContext = None) -> Distribution:
        # transformation to euclidian space
        _alpha = 2. / (1. + torch.exp(-self.alpha))
        _beta = 2. / (1. + torch.exp(-self.beta))
        _loc = self.loc
        _scale = torch.exp(self.scale)
        return TorchStable(alpha=_alpha, beta=_beta, loc=_loc, scale=_scale)
    


if __name__ == "__main__":
    from simple_einet.einet import Einet, EinetConfig
    from simple_einet.einet_mixture import EinetMixture
    from sklearn.datasets import make_blobs

    device = "cpu"

    n = 1000
    n_vars = 2
    centers = [[-5, -5], [5, 5]]
    data, _ = make_blobs(n_samples=n, n_features=2, centers=centers)
    data = torch.tensor(data).to(device)


    num_features = 4
    num_channels = data.shape[0]
    depth = 2
    num_sums = 2
    num_channels = 1
    num_leaves = 3
    num_repetitions = 3
    num_classes = 1
    dropout = 0.0
    leaf_type = Stable
    leaf_kwargs = {}

    config = EinetConfig(
        num_features=num_features,
        num_channels=num_channels,
        depth=depth,
        num_sums=num_sums,
        num_leaves=num_leaves,
        num_repetitions=num_repetitions,
        num_classes=num_classes,
        leaf_type=leaf_type,
        leaf_kwargs=leaf_kwargs,
        layer_type="linsum",
        dropout=0.0,
    )

    model = Einet(config)
    
    lls = model(data)

    print(lls)