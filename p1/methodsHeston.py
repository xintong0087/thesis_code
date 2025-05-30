from scipy.optimize import fsolve
from scipy.integrate import quad

from cmath import log, exp, sqrt, pi


def alphahat(u):

    return -0.5 * u * complex(u, 1)


def beta(u, rho, sigma, kappa):

    return kappa - rho * sigma * u * complex(0, 1)


def gamma(sigma):

    return 0.5 * sigma ** 2


def D(u, rho, sigma, kappa):

    return sqrt(beta(u, rho, sigma, kappa) ** 2 - 4 * alphahat(u) * gamma(sigma))


def G(u, rho, sigma, kappa):

    return (beta(u, rho, sigma, kappa) - D(u, rho, sigma, kappa)) / (beta(u, rho, sigma, kappa) + D(u, rho, sigma, kappa))


def phi2(u, rho, sigma, kappa, tau):

    return (G(u, rho, sigma, kappa) * exp(-D(u, rho, sigma, kappa) * tau) - 1) / (G(u, rho, sigma, kappa) - 1)


def A(u, kappa, theta, rho, sigma, tau):

    return kappa*theta*((beta(u, rho, sigma, kappa) - D(u, rho, sigma, kappa))*tau - 2*log(phi2(u, rho, sigma, kappa, tau))) / (sigma**2)


def Bv(u, rho, sigma, kappa, tau):

    return (beta(u, rho, sigma, kappa) - D(u, rho, sigma, kappa)) * ((1 - exp(-D(u, rho, sigma, kappa)*tau)) / ((sigma**2) * (1 - G(u, rho, sigma, kappa) * exp(-D(u, rho, sigma, kappa)*tau))))


def cf(u, F, kappa, theta, rho, sigma, tau, v0):
    f = log(F)

    return exp(complex(0, 1) * u * f 
               + A(u, kappa, theta, rho, sigma, tau)
               + Bv(u, rho, sigma, kappa, tau) * v0)


def phi(v, K, alpha, F, kappa, theta, rho, sigma, tau, v0):
    k = log(K)
    y = exp(complex(0, -1)*(complex(v, -alpha) * k)) \
            * cf(complex(v, -(alpha+1)), F, kappa, theta, rho, sigma, tau, v0) \
            / (-complex(v, -(alpha+1)) * complex(v, -alpha))

    return y.real

def psi(alpha, K, F, kappa, theta, rho, sigma, tau, v0):
    k = log(K)
    temp = -alpha*k + 0.5*log(phi(-complex(0, alpha+1), K, alpha, F, kappa, theta, rho, sigma, tau, v0)**2)
    
    return temp.real

def Ralpha(F, K, alpha):
    
    return F * (alpha<=0) - K * (alpha<=-1) - 0.5 * (F * (alpha==0) - K * (alpha==-1))


def priceHeston(S, K, T, r, kappa, theta, rho, sigma, v0, optionType, alpha=None):

    F = S * exp(r * T)

    alpha0 = 0.75
    if alpha is None:
        try:
            alpha = fsolve(lambda a: psi(a, K, F, kappa, theta, rho, sigma, T, v0), alpha0)[0]
        except:
            alpha = alpha0
        
    price = Ralpha(F, K, alpha) + 1/pi * quad(lambda x: phi(x, K, alpha, F, kappa, theta, rho, sigma, T, v0), 0.0, float("inf"))[0]

    if optionType == "P":
        price += K * exp(-r * T) - S 

    return price.real
