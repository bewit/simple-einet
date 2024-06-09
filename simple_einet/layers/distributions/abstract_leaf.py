import logging
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import distributions as dist, nn

from simple_einet.abstract_layers import AbstractLayer
from simple_einet.sampling_utils import SamplingContext, index_one_hot
from simple_einet.type_checks import check_valid
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def dist_forward(distribution, x: torch.Tensor):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [n, d].

    Returns:
        torch.Tensor: Log probabilities for each feature.
    """
    # Make room for out_channels and num_repetitions of layer

    if x.dim() == 3:  # [N, C, D]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [N, C, D, 1, 1]

    # Compute log-likelihodd
    try:
        x = distribution.log_prob(x)  # Shape: [n, d, oc, r]
    except ValueError as e:
        print("min:", x.min())
        print("max:", x.max())
        raise e

    return x


def dist_cdf(distribution, x: torch.Tensor):
    """
    Computation of the CDF w.r.t. x of an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the probability of x w.r.t. the CDF.
        x: Input to compute the probability of.
           Shape [n, d].

    Returns:
        torch.Tensor: Probabilities for each feature.
    """
    if x.dim() == 3:  # [N, C, D]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [N, C, D, 1, 1]

    try:
        x = distribution.cdf(x)  # Shape: [n, d, oc, r]
    except ValueError as e:
        print("min:", x.min())
        print("max:", x.max())
        raise e

    return x


def dist_mode(distribution: dist.Distribution, ctx: SamplingContext = None) -> torch.Tensor:
    """
    Get the mode of a given distribution.

    Args:
        distribution: Leaf distribution from which to choose the mode from.
        ctx: Sampling context.
    Returns:
        torch.Tensor: Mode of the given distribution.
    """
    # TODO: Implement more torch distributions
    if isinstance(distribution, dist.Normal):
        # Repeat the mode along the batch axis
        return distribution.mean.repeat(ctx.num_samples, 1, 1, 1, 1)

    from simple_einet.layers.distributions.multivariate_normal import CustomMultivariateNormalDist

    if isinstance(distribution, CustomMultivariateNormalDist):
        return distribution.mpe(num_samples=ctx.num_samples)

    from simple_einet.layers.distributions.normal import CustomNormal
    from simple_einet.layers.distributions.binomial import DifferentiableBinomial

    if isinstance(distribution, CustomNormal):
        # Repeat the mode along the batch axis
        return distribution.mpe(num_samples=ctx.num_samples)
    elif isinstance(distribution, dist.Bernoulli):
        mode = distribution.probs.clone()
        mode[mode >= 0.5] = 1.0
        mode[mode < 0.5] = 0.0
        return mode.repeat(ctx.num_samples, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Binomial) or isinstance(distribution, DifferentiableBinomial):
        mode = distribution.probs.clone()
        total_count = distribution.total_count
        mode = torch.floor(mode * (total_count + 1))
        if mode.shape[0] == 1:
            return mode.repeat(ctx.num_samples, 1, 1, 1, 1)
        else:
            return mode
    elif isinstance(distribution, dist.Categorical):
        probs = distribution.probs.clone()
        mode = torch.argmax(probs, dim=-1)
        return mode.repeat(ctx.num_samples, 1, 1, 1, 1)
    else:
        raise Exception(f"MPE not yet implemented for type {type(distribution)}")


def dist_sample(distribution: dist.Distribution, ctx: SamplingContext = None) -> torch.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        distribution: Leaf distribution from which to sample from.
        ctx: Sampling context.

    Returns:
        torch.Tensor: Samples from the given distribution.
    """

    # Sample from the specified distribution
    if ctx.is_mpe or ctx.mpe_at_leaves:
        samples = dist_mode(distribution, ctx).float()
        samples = samples.unsqueeze(1)
    else:
        from simple_einet.layers.distributions.normal import CustomNormal

        if type(distribution) == dist.Normal:
            distribution = dist.Normal(loc=distribution.loc, scale=distribution.scale / ctx.temperature_leaves)
        elif type(distribution) == CustomNormal:
            distribution = CustomNormal(mu=distribution.mu, sigma=distribution.sigma / ctx.temperature_leaves)
        elif type(distribution) == dist.Categorical:
            distribution = dist.Categorical(logits=F.log_softmax(distribution.probs / ctx.temperature_leaves))
        samples = distribution.sample(sample_shape=(ctx.num_samples,)).float()

    assert (
        samples.shape[1] == 1
    ), "Something went wrong. First sample size dimension should be size 1 due to the distribution parameter dimensions. Please report this issue."

    # if not context.is_differentiable:  # This happens only in the non-differentiable context
    samples.squeeze_(1)
    num_samples, num_channels, num_features, num_leaves, num_repetitions = samples.shape

    if ctx.is_differentiable:
        r_idxs = ctx.indices_repetition.view(num_samples, 1, 1, 1, num_repetitions)
        samples = index_one_hot(samples, index=r_idxs, dim=-1)
    else:
        r_idxs = ctx.indices_repetition.view(-1, 1, 1, 1, 1)
        r_idxs = r_idxs.expand(-1, num_channels, num_features, num_leaves, -1)
        samples = samples.gather(dim=-1, index=r_idxs)
        samples = samples.squeeze(-1)

    # If parent index into out_channels are given
    if ctx.indices_out is not None:
        # Choose only specific samples for each feature/scope
        samples = torch.gather(samples, dim=2, index=ctx.indices_out.unsqueeze(-1)).squeeze(-1)

    return samples


class AbstractLeaf(AbstractLayer, ABC):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    Attributes:
        num_features: Number of input features.
        num_channels: Number of input features.
        num_leaves: Number of parallel representations for each input feature.
        num_repetitions: Number of parallel repetitions of this layer.
        cardinality: Number of random variables covered by a single leaf.
    """

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int = 1,
        cardinality=1,
    ):
        """
        Create the leaf layer.

        Args:
            num_features: Number of input features.
            num_channels: Number of input features.
            num_leaves: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of random variables covered by a single leaf.
        """
        super().__init__(num_features=num_features, num_repetitions=num_repetitions)
        self.num_channels = check_valid(num_channels, int, 1)
        self.num_leaves = check_valid(num_leaves, int, 1)
        self.cardinality = check_valid(cardinality, int, 1)

        self.out_features = num_features
        self.out_shape = f"(N, {num_features}, {num_leaves})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Placeholder to replace nan values for the forward pass to circument errors in the torch distributions
        # This value is distribution specific since it needs to be inside of the distribution support and might need to
        # be adjusted
        self.nan_placeholder = 0

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies dropout to the input tensor `x` according to the dropout probability
        `self.dropout`. Dropout is only applied during training (when `model.train()`
        has been called).

        Args:
            x (torch.Tensor): The input tensor to apply dropout to.

        Returns:
            torch.Tensor: The input tensor with dropout applied.
        """
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(
                x.shape,
            ).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        """
        Marginalizes the input tensor `x` along the dimensions specified in `marginalized_scopes`.

        Args:
            x (torch.Tensor): The input tensor to be marginalized.
            marginalized_scopes (List[int]): A list of dimensions to be marginalized.

        Returns:
            torch.Tensor: The marginalized tensor.
        """
        # Marginalize nans set by user
        if marginalized_scopes is not None:
            # Transform to tensor
            if type(marginalized_scopes) != torch.Tensor:
                s = torch.tensor(marginalized_scopes)
            else:
                s = marginalized_scopes

            # Adjust for leaf cardinality
            if self.cardinality > 1:
                s = marginalized_scopes.div(self.cardinality, rounding_mode="floor")

            x[:, :, s] = self.marginalization_constant
        return x

    def forward(self, x, marginalized_scopes: List[int]):
        """
        Forward pass through the distribution.

        Args:
            x (torch.Tensor): Input tensor.
            marginalized_scopes (List[int]): List of scopes to marginalize.

        Returns:
            torch.Tensor: Output tensor after marginalization.
        """
        # Forward through base distribution
        d = self._get_base_distribution()
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            # Replace nans with some valid value
            x = torch.where(torch.isnan(x), self.nan_placeholder, x)

        # Perform forward pass
        x = dist_forward(d, x)

        # Set back to nan
        if nan_mask.any():
            x[nan_mask] = torch.nan

        x = self._marginalize_input(x, marginalized_scopes)

        return x
    

    def log_cdf(self, x, marginalized_scopes: List[int]):
        """
        Computation of the log of the cumulative distribution function.

        Args:
            x (torch.Tensor): Input tensor.
            marginalized_scopes (List[int]): List of scopes to marginalize.

        Returns:
            torch.Tensor: Output tensor after marginalization.
        """
        # Forward through base distribution
        d = self._get_base_distribution()
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            # Replace nans with some valid value
            x = torch.where(torch.isnan(x), self.nan_placeholder, x)

        # Perform forward pass
        x = dist_cdf(d, x)

        # Set back to nan
        if nan_mask.any():
            x[nan_mask] = torch.nan

        x = self._marginalize_input(x, marginalized_scopes)

        x = torch.log(x)

        return x
    

    def log_characteristic_function(self, t, marginalized_scopes: List[int]):
        """
        Computation of the log of the characteristic function.

        Args:
            t (torch.Tensor): Input tensor.
            marginalized_scopes (List[int]): List of scopes to marginalize.

        Returns:
            torch.Tensor: Output tensor after marginalization.
        """
        # Forward through base distribution
        nan_mask = torch.isnan(t)
        if nan_mask.any():
            # Replace nans with some valid value
            t = torch.where(torch.isnan(t), self.nan_placeholder, t)

        # Perform forward pass
        if t.dim() == 3:  # [N, C, D]
            t = t.unsqueeze(-1).unsqueeze(-1)  # [N, C, D, 1, 1]
        try:
            t = torch.real(self._log_characteristic_function(t))  # Shape: [n, d, oc, r]
        except ValueError as e:
            print("min:", t.min())
            print("max:", t.max())
            raise e

        # Set back to nan
        if nan_mask.any():
            t[nan_mask] = torch.nan

        t = self._marginalize_input(t, marginalized_scopes)

        return t


    @abstractmethod
    def _get_base_distribution(self, ctx: SamplingContext = None) -> dist.Distribution:
        """Get the underlying torch distribution."""
        pass


    @abstractmethod
    def _log_characteristic_function(self, t: torch.Tensor) -> torch.Tensor:
        pass


    def sample(self, ctx: SamplingContext) -> torch.Tensor:
        """
        Sample from the distribution represented by this leaf node.

        Args:
            ctx (SamplingContext, optional): The sampling context to use when drawing samples.

        Returns:
            torch.Tensor: A tensor of shape (context.num_samples,) or (1,) containing the drawn samples.
        """
        d = self._get_base_distribution(ctx)
        samples = dist_sample(distribution=d, ctx=ctx)
        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_leaves={self.num_leaves}, out_shape={self.out_shape}"
