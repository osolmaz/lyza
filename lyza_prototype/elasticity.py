
def mu_from_E_nu(E, nu):
    return E/(1.+nu)/2.

def lambda_from_E_nu(E, nu):
    return E*nu/(1.+nu)/(1.-2.*nu)

def E_from_lambda_mu(lambda_, mu):
    return mu*(3.*lambda_+2.*mu)/(lambda_+mu)

def nu_from_lambda_mu(lambda_, mu):
    return lambda_/2./(lambda_+mu)

def kappa_from_lambda_mu(lambda_, mu):
    return lambda_ + 2./3.*mu

def kappa_from_E_nu(E, nu):
    return E/3./(1.-2.*nu)

