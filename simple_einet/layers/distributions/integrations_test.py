import torch
from torchquad import Trapezoid, MonteCarlo, Simpson
from functools import partial
from tabulate import tabulate


def stdnorm(x):
    return (torch.tensor(1.0)/torch.sqrt(torch.tensor(2.0)*torch.pi) * torch.exp((torch.tensor(-1/2) * torch.pow(x, 2))))


def transform_half_01(fct, t):
    return fct(t/(1-t)) / torch.pow(1-t, 2)


def transform_full_11(fct, t):
    return fct(t/(1 - torch.pow(t, 2))) * (1 + torch.pow(t, 2)) / torch.pow(1 - torch.pow(t, 2), 2)



def stable_cf(t: torch.Tensor, alpha=torch.tensor(1.5), beta=torch.tensor(0.0), gamma=torch.tensor(1/torch.sqrt(torch.tensor(2.0))), mu=torch.tensor(0.0)):
    if alpha == 1.0:
        phi = - (2./torch.pi) * torch.log(torch.abs(t))
    else:
        phi = torch.tan((torch.pi * alpha) / 2.0)
    return torch.real(torch.exp(1j * t * mu - torch.pow(torch.abs(gamma * t), alpha) * (1.0 - 1j * beta * torch.sign(t) * phi)) )    


def stable_cdf(x: torch.Tensor, n_mc=10000, alpha=torch.tensor(1.5), beta=torch.tensor(0.0), gamma=torch.tensor(1/torch.sqrt(torch.tensor(2.0))), mu=torch.tensor(0.0)):
    def integrand(t):
        return torch.real(stable_cf(t) / (torch.tensor(2.0) * torch.pi * 1j * t) * torch.exp(-1j * t * x))
    
    transformed_integrand = partial(transform_full_11, integrand)
    integrator = MonteCarlo()
    integral = integrator.integrate(transformed_integrand, dim=1, N=n_mc)
    result = torch.tensor(1/2) - integral
    return result


import numpy as np
logvals = np.logspace(-1, 5, 20)
x = list((-1) * logvals) + list(logvals)
x.append(0)
x.sort()
x = torch.tensor(x)
results = {1000: [], 10000: [], 100000: [], 1000000: [], 10000000: []}
for n in results.keys():
    cdfs = stable_cdf(x = x, n_mc=n)
    results[n] = cdfs
results["x"] = x
print(tabulate(results, headers="keys"))


# mc_nodes = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
# repeats = range(1000)
# integrator = MonteCarlo()
# transformed_stdnorm = partial(transform_half_01, stdnorm)
# for n in mc_nodes:
#     ys = torch.tensor([integrator.integrate(transformed_stdnorm, dim=1, N=n, integration_domain=[[0.0, 1.0]]) for i in repeats])
#     print(n, torch.mean(ys), torch.std(ys))