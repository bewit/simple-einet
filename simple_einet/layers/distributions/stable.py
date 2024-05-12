import math
from numbers import Number, Real
from functools import partial

import torch
from torch import distributions as dist
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all
from torchquad import MonteCarlo
from c_levyst import Nolan

# I think this can be improved a lot
# TODO: vectorize this to evaluate multiple pdf/cdf computations of the same distribution
# TODO: precompute pdf/cdf for (alpha, beta)-grid (with 0.01 spacing?) and transform and map RV for lookup tables

__all__ = ["Stable"]
torch.set_default_dtype(torch.double)
# see doc of scipy.stats.levy_stable
# I tried to stay as close to the original numpy implementation as possible (and kind of sensible)
_QUAD_EPS = 1e-10

integrator = MonteCarlo()
integrator_params = {"N": 1000}
LBFGS_epochs = 20
LBFGS_lr = 0.1
PI = torch.tensor(math.pi)


def transform_half_real_line_to_unit_interval(fct, t):
    return fct(t/(1-t)) / torch.pow(1-t, 2)


def _Phi_Z0(alpha, t):
    return (
        -torch.tan(PI * alpha / 2) * (torch.pow(torch.abs(t), (1 - alpha)) - 1)
        if alpha != 1
        else -2.0 * torch.log(torch.abs(t)) / PI
    )


def _Phi_Z1(alpha, t):
    return (
        torch.tan(PI * alpha / 2)
        if alpha != 1
        else -2.0 * torch.log(torch.abs(t)) / PI
    )


def _cf(Phi, t, alpha, beta):
    return torch.exp(
        -(torch.pow(torch.abs(t), alpha) * (1 - 1j * beta * torch.sign(t) * Phi(alpha, t)))
    )


_cf_Z0 = partial(_cf, _Phi_Z0)
_cf_Z1 = partial(_cf, _Phi_Z1)


def _pdf_single_value_cf_integrate(Phi, x, alpha, beta, **kwds):
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)

    # split up integral and use change of variables to project to unit interval

    def integrand1(t):
        if torch.all(t == 0):
            return torch.zeros_like(t)
        return torch.exp(-(torch.pow(t, alpha))) * (
            torch.cos(beta * torch.pow(t, alpha) * Phi(alpha, t))
        )
    
    def integrand2(t):
        if torch.all(t == 0):
            return torch.zeros_like(t)
        return torch.exp(-(torch.pow(t, alpha))) * (
            torch.sin(beta * torch.pow(t, alpha) * Phi(alpha, t))
        )

    transformed_integrand1 = partial(transform_half_real_line_to_unit_interval, integrand1)
    transformed_integrand2 = partial(transform_half_real_line_to_unit_interval, integrand2)

    int1 = integrator.integrate(transformed_integrand1, dim=1, integration_domain=[[0, 1]], **integrator_params)
    int2 = integrator.integrate(transformed_integrand2, dim=1, integration_domain=[[0, 1]], **integrator_params)

    return (int1 + int2) / PI

_pdf_single_value_cf_integrate_Z0 = partial(_pdf_single_value_cf_integrate, _Phi_Z0)
_pdf_single_value_cf_integrate_Z1 = partial(_pdf_single_value_cf_integrate, _Phi_Z1)


def _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta):
    if torch.abs(x0 - zeta) < x_tol_near_zeta * torch.pow(alpha, (1/alpha)):
        x0 = zeta
    return x0


def _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one):
    if torch.abs(alpha - 1) < alpha_tol_near_one:
        alpha = torch.tensor(1.0)
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    return x0, alpha, beta


def _pdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    zeta = -beta * torch.tan(PI * alpha / 2.0)
    x0 = x + zeta if alpha != 1 else x
    
    return _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)


def _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", torch.tensor(0.005))
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", torch.tensor(0.005))

    zeta = -beta * torch.tan(PI * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one)

    if alpha == 2.0:
        def _norm_pdf(x):
            return torch.exp(-torch.pow(x, 2) / 2.0) / torch.sqrt(2. * PI)
        return _norm_pdf(x0 / torch.sqrt(torch.tensor(2.0))) / torch.sqrt(torch.tensor(2.0))
    elif alpha == 0.5 and beta == 1.0:
        _x = x0 + 1.
        if _x <= 0:
            return torch.tensor(0.0)
        return torch.tensor(1.0 / torch.sqrt(2.0 * PI * _x) / _x * torch.exp(-1.0 / (2 * _x)))
    # elif alpha == 0.5 and beta == 0.0 and x0 != 0:
        # S, C = sc.fresnel([1 / np.sqrt(2 * np.pi * np.abs(x0))])
        # arg = 1 / (4 * np.abs(x0))
        # return (
        #     np.sin(arg) * (0.5 - S[0]) + np.cos(arg) * (0.5 - C[0])
        # ) / np.sqrt(2 * np.pi * np.abs(x0) ** 3)
        # cannot implement this as fresnel integrals are not available in torch
    elif alpha == 1.0 and beta == 0.0:
        return 1.0 / (1.0 + torch.pow(x0, 2)) / PI
    
    return _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta)


def _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta):
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c2 = _nolan.c2
    g = _nolan.g

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    if torch.isclose(x0, zeta):
        return(
            torch.exp(torch.lgamma(1. + 1./alpha))
            * torch.cos(xi)
            / PI
            / (torch.pow(1. + torch.pow(zeta, 2), (1./alpha/2.)))
        )
    elif x0 < zeta:
        return _pdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, quad_eps, x_tol_near_zeta
        )
    
    if torch.isclose(-xi, PI/2.):
        return torch.tensor(0.)
    

    def integrand(theta):
        g_1 = g(theta)
        if not torch.all(torch.isfinite(g_1)):
            g_1 = torch.zeros_like(g_1)
        if not torch.all(g_1 >= 0):
            g_1 = torch.zeros_like(g_1)
        return g_1 * torch.exp(-g_1)
    
    # bisect and np.quad (which is quasi Fortran's quadpack) are not available
    # instead integrate with MonteCarlo and hope for the best
    intg = integrator.integrate(integrand, dim=1, **integrator_params, integration_domain=[[-xi, PI/2.]])

    return c2 * intg


def _cdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    zeta = -beta * torch.tan(PI * alpha / 2.)
    x0 = x + zeta if alpha != 1. else x

    return _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)
    

def _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", torch.tensor(0.005))
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", torch.tensor(0.005))

    zeta = -beta * torch.tan(PI * alpha / 2.)
    x0, alpha, beta = _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one)

    if alpha == 2.0:
        dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])).cdf(x0 / torch.sqrt(torch.tensor(2.)))
    elif alpha == 0.5 and beta == 1.0:
        _x = x0 + 1.
        if _x <= 0:
            return torch.tensor(0.)
        return (1. - torch.erf(torch.sqrt(0.5 / _x)))
    elif alpha == 1.0 and beta == 0.0:
        return 0.5 + torch.arctan(x0) / PI
    
    return _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta)


def _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta):
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c1 = _nolan.c1
    c3 = _nolan.c3
    g = _nolan.g

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    if (alpha == 1. and beta < 0.) or x0 < zeta:
        return 1 - _cdf_single_value_piecewise_post_rounding_Z0(-x0, alpha, -beta, quad_eps, x_tol_near_zeta)
    elif torch.isclose(x0, zeta):
        return 0.5 - xi / PI
    
    if torch.isclose(-xi, PI/torch.tensor(2.)):
        return c1
    
    def integrand(theta):
        g_1 = g(theta)
        if not torch.all(torch.isfinite(g_1)):
            g_1 = torch.zeros_like(g_1)
        if not torch.all(g_1 >= 0):
            g_1 = torch.zeros_like(g_1)
        return torch.exp(-g_1)
    
    left_support = -xi
    right_support = PI / 2.

    if alpha > 1.:
        # TODO: i'm not really sure if the purpose of the following optimization is just to reduce the interval of integration lateron. if so, we could probably skip this. needs evaluation
        if not torch.isclose(integrand(-xi), torch.tensor(0.)):
            param = torch.clone(-xi).detach().requires_grad_(True)
            opt = torch.optim.LBFGS([param], lr=LBFGS_lr)
            def closure():
                opt.zero_grad()
                loss = integrand(param)
                loss.requires_grad_(True)
                loss.backward()
                return loss
            for epoch in range(LBFGS_epochs):
                opt.step(closure=closure)
            param.clamp(-xi, PI/2.)
            left_support = param
    else: # alpha < 1.
        if not torch.isclose(integrand(PI/2.), torch.tensor(0.)):
            param = torch.tensor(PI/2.).requires_grad_(True)
            opt = torch.optim.LBFGS([param], lr=LBFGS_lr)
            def closure():
                opt.zero_grad()
                loss = integrand(param)
                loss.requires_grad_(True)
                loss.backward()
                return loss
            for epoch in range(LBFGS_epochs):
                opt.step(closure=closure)
            param.clamp(-xi, PI/2.)
            right_support = param

    intg = integrator.integrate(integrand, dim=1, **integrator_params, integration_domain=[[left_support, right_support]])

    return c1 + c3 * intg


def _rvs_Z1(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
    raise NotImplementedError("Sampling not implemented")


def _fitstart_S0(data):
    raise NotImplementedError("Estimation not implemented yet")


def _fitstart_S1(data):
    raise NotImplementedError("Estimation not implemented yet")



class Stable(Distribution):
    r"""
    Creates a stable distribution (also called alpha-stable, levy-stable) parametrized by 
    :attr:`alpha`, :attr:`beta`, :attr:`scale`, and :attr:`loc`.
    """
    arg_constraints = {"alpha": constraints.interval(0.0, 2.0), "beta": constraints.interval(-1.0, 1.0), "scale": constraints.positive, "loc": constraints.real}
    support = constraints.real # TODO: actually the support depends on beta (and alpha) e.g. see Levy distribution
    has_rsample = False # TODO: implement if necessary

    # levy_stable parameters
    parametrization = "S1"
    pdf_default_method = "piecewise"
    cdf_default_method = "piecewise"
    quad_eps = _QUAD_EPS
    piecewise_x_tol_near_zeta = torch.tensor(0.005)
    piecewise_alpha_tol_near_one = torch.tensor(0.005)

    def __init__(self, alpha, beta, loc, scale, validate_args=None):
        self.alpha, self.beta, self.loc, self.scale = broadcast_all(alpha, beta, loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _parametrization(self):
        allowed = ("S0", "S1")
        pz = self.parametrization
        if pz not in allowed:
            raise RuntimeError(f"Parametrization '{pz}' not in supported list: {allowed}")
        return pz
    
    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("I don't know where is this relevant yet")
    

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented")
    

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented")
    

    def pdf(self, value: torch.Tensor) -> torch.Tensor:
        if self.parametrization == "S0":
            return self._pdf(value, self.alpha, self.beta, self.loc, self.scale)
        elif self.parametrization == "S1":
            alpha = self.alpha
            beta = self.beta
            loc = self.loc
            scale = self.scale
            if torch.all(torch.reshape(self.alpha, (1, -1))[0, :] != 1.):
                return self._pdf(value, alpha, beta, loc, scale)
            else:
                value = torch.reshape(value, (1, -1))[0, :]
                value, alpha, beta = broadcast_all(value, alpha, beta)

                data_in = torch.dstack((value, alpha, beta))[0]
                data_out = torch.empty(size=(len(data_in), 1))

                uniq_param_pairs = torch.unique(data_in[:, 1:], dim = 0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = (loc + 2*_beta*scale * torch.log(scale) / PI if _alpha==1.0 else loc)
                    data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = self._pdf(_x, _alpha, _beta, loc=_delta, scale=scale).reshape(len(_x), 1)

                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output
    

    def _pdf(self, value, alpha, beta, loc, scale) -> torch.Tensor:
        if self.parametrization == "S0":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z0
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z0
            _cf = _cf_Z0
        elif self.parametrization == "S1":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z1
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z1
            _cf = _cf_Z1
        
        value = torch.reshape(value, (1, -1))[0, :]
        # standardize
        value = (value - loc) / scale

        value, alpha, beta = broadcast_all(value, alpha, beta)

        data_in = torch.dstack((value, alpha, beta))[0]
        data_out = torch.empty(size=(len(data_in), 1))

        pdf_default_method_name = self.pdf_default_method
        if pdf_default_method_name in ("piecewise", "best", "zolotarev"):
            pdf_single_value_method = _pdf_single_value_piecewise
        elif pdf_default_method_name in ("dni", "quadrature"):
            pdf_single_value_method = _pdf_single_value_cf_integrate
        else:
            raise ValueError(f"PDF method '{self.pdf_default_method}' not supported")
        
        pdf_single_value_kwds = {
            "quad_eps": self.quad_eps,
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        # no fft support (by now)

        uniq_param_pairs = torch.unique(data_in[:, 1:], dim=0)
        for pair in uniq_param_pairs:
            data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
            data_subset = data_in[data_mask]
            data_out[data_mask] = torch.tensor(
                [
                    pdf_single_value_method(_value, _alpha, _beta, **pdf_single_value_kwds)
                    for _value, _alpha, _beta in data_subset
                ]
            ).reshape(len(data_subset), 1)

        # account for standardization
        data_out = data_out / scale

        return data_out.T[0]


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(value))
    

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self.parametrization == "S0":
            return self._cdf(value, self.alpha, self.beta, self.loc, self.scale)
        elif self.parametrization == "S1":
            alpha = self.alpha
            beta = self.beta
            loc = self.loc
            scale = self.scale
            if torch.all(torch.reshape(self.alpha, (1, -1))[0, :] != 1.):
                return self._cdf(value, alpha, beta, loc, scale)
            else:
                value = torch.reshape(value, (1, -1))[0, :]
                value, alpha, beta = broadcast_all(value, alpha, beta)

                data_in = torch.dstack((value, alpha, beta))[0]
                data_out = torch.empty(size=(len(data_in), 1))

                uniq_param_pairs = torch.unique(data_in[:, 1:], dim = 0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = (loc + 2*_beta*scale * torch.log(scale) / PI if _alpha==1.0 else loc)
                    data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = self._cdf(_x, _alpha, _beta, loc=_delta, scale=scale).reshape(len(_x), 1)

                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output
    

    def _cdf(self, value, alpha, beta, loc, scale):
        if self.parametrization == "S0":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z0
            _cf = _cf_Z0
        elif self.parametrization == "S1":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z1
            _cf = _cf_Z1

        value = torch.reshape(value, (1, -1))[0, :]
        # standardize
        value = (value - loc) / scale

        value, alpha, beta = broadcast_all(value, alpha, beta)

        data_in = torch.dstack((value, alpha, beta))[0]
        data_out = torch.empty(size=(len(data_in), 1))

        cdf_default_method_name = self.cdf_default_method
        if cdf_default_method_name in ("piecewise", "best", "zolotarev"):
            cdf_single_value_method = _cdf_single_value_piecewise
        else:
            raise ValueError(f"PDF method '{self.pdf_default_method}' not supported")
        
        cdf_single_value_kwds = {
            "quad_eps": self.quad_eps,
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        # fft not supported (by now)
        uniq_param_pairs = torch.unique(data_in[:, 1:], dim=0)
        for pair in uniq_param_pairs:
            data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
            data_subset = data_in[data_mask]
            data_out[data_mask] = torch.tensor(
                [
                    cdf_single_value_method(_value, _alpha, _beta, **cdf_single_value_kwds)
                    for _value, _alpha, _beta in data_subset
                ]
            ).reshape(len(data_subset), 1)

        return data_out.T[0]


    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

    def entropy(self):
        raise NotImplementedError
    


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import levy_stable
    from tabulate import tabulate

    params_set = [
        {"alpha": 2.0, "beta": 0.0, "loc": 0.0, "scale": 1./np.sqrt(2.)},
        {"alpha": 1.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
        {"alpha": 1.5, "beta": 0.5, "loc": 1000.0, "scale": 100.0},
        {"alpha": 1.0, "beta": 0.0, "loc": 0.0, "scale": 1.0},
        {"alpha": 1.0, "beta": 1.0, "loc": 4.2, "scale": 1.0},
        {"alpha": 1.0, "beta": -1.0, "loc": 4.2, "scale": 1.0},
        {"alpha": 0.5, "beta": 0.5, "loc": 0.0, "scale": 0.5},
        {"alpha": 0.5, "beta": 1.0, "loc": 0.0, "scale": 0.5},
        {"alpha": 0.5, "beta": -1.0, "loc": 0.0, "scale": 0.5},
    ]

    for params in params_set:
        alpha = params["alpha"]
        beta = params["beta"]
        loc = params["loc"]
        scale = params["scale"]

        torch_stable = Stable(alpha=torch.tensor(alpha), beta=torch.tensor(beta), loc=torch.tensor(loc), scale=torch.tensor(scale))
        scipy_stable = levy_stable(alpha=alpha, beta=beta, loc=loc, scale=scale)


        logvals = np.logspace(-5, 5, 11)
        x = list((-1) * logvals) + list(logvals)
        x.sort()
        data = torch.tensor(x)

        results = {"data": data}

        torch_densities = torch_stable.pdf(data)
        results["t-PDF"] = torch_densities
        scipy_densities = scipy_stable.pdf(data)
        results["s-PDF"] = scipy_densities
        torch_probs = torch_stable.cdf(data)
        results["t-CDF"] = torch_probs
        scipy_probs = scipy_stable.cdf(data)
        results["s-CDF"] = scipy_probs

        print("Params: ")
        print(params)
        print(tabulate(results, headers="keys"))
