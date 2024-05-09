import torch

M_PI_2 = torch.tensor(1.57079632679489661923)
M_1_PI = torch.tensor(0.31830988618379067154)
M_2_PI = torch.tensor(0.63661977236758134308)  

class Nolan:
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, x0: torch.Tensor):
        self.alpha = alpha
        self.zeta = -beta * torch.tan(M_PI_2 * alpha)

        if alpha != 1.:
            self.xi = torch.atan(-self.zeta) / alpha
            self.zeta_prefactor = torch.pow(torch.pow(self.zeta, 2) + 1., -1. / (2. * (alpha - 1.)))
            self.alpha_exp = alpha / (alpha - 1.)
            self.alpha_xi = torch.atan(-self.zeta)
            self.zeta_offset = x0 - self.zeta

            if alpha < 1.:
                self.c1 = 0.5 - self.xi * M_1_PI
                self.c3 = M_1_PI
            else: # alpha > 1.
                self.c1 = torch.tensor(1.)
                self.c3 = -M_1_PI

            self.c2 = alpha * M_1_PI / torch.abs(alpha - 1.) / (x0 - self.zeta)
            self.g = self.g_alpha_ne_one

        else: # alpha == 1.
            self.xi = M_PI_2
            self.two_beta_div_pi = beta * M_2_PI
            self.pi_div_two_beta = M_PI_2 / beta
            self.x0_div_term = x0 / self.two_beta_div_pi
            self.c1 = torch.tensor(0.)
            self.c2 = 0.5 / torch.abs(beta)
            self.c3 = M_1_PI
            self.g = self.g_alpha_eq_one

    
    def g_alpha_ne_one(self, theta: torch.Tensor) -> torch.Tensor:
        if theta == -self.xi:
            if self.alpha < 1.:
                return torch.tensor(0.)
            else:
                return torch.inf
        if theta == M_PI_2:
            if self.alpha < 1.:
                return torch.inf
            else:
                return 0
            
        cos_theta = torch.cos(theta)
        return (
            self.zeta_prefactor
            * torch.pow(
                cos_theta
                / torch.sin(self.alpha_xi + self.alpha * theta)
                * self.zeta_offset, self.alpha_exp)
            * torch.cos(self.alpha_xi + (self.alpha - 1.) * theta)
            / cos_theta
        )
    
    
    def g_alpha_eq_one(self, theta: torch.Tensor) -> torch.Tensor: 
        if theta == -self.xi:
            return 0
        if theta == M_PI_2:
            return torch.inf
        
        return (
            (1. + theta * self.two_beta_div_pi)
            * torch.exp((self.pi_div_two_beta + theta) * torch.tan(theta) - self.x0_div_term)
            / torch.cos(theta)
        )





    