import numpy as np

# Constante da velocidade da luz em km/s
CLIGHT = 299792.458

def mu_luthoor_wrapper(z_cmb, z_hel, params):
    """
    Implementa a Geometria Efetiva Consciente.
    Se o modelo for empurrado para o silêncio matemático (argumento <= 0),
    ele retorna NaNs, sinalizando ao MCMC que aquela região não existe.
    """
    B, C, M = params

    z_cmb = np.atleast_1d(z_cmb)
    # Máscara para evitar singularidade trivial em z=0
    mask = z_cmb > 0.001
    mu_model = np.zeros_like(z_cmb)

    # Termo base Hubble
    log_cz = np.zeros_like(z_cmb)
    log_cz[mask] = 5.0 * np.log10(CLIGHT * z_cmb[mask])
    
    # --- RESTRIÇÃO EXPLÍCITA DE DOMÍNIO ---
    # Calculamos o argumento "nu" antes de aplicar o log
    arg_log = 1.0 + B * z_cmb[mask] + C * (z_cmb[mask]**2)
    
    # Verificação de Segurança:
    # Se qualquer ponto cair no "silêncio matemático" (<= 0), rejeitamos tudo.
    if np.any(arg_log <= 0):
        return np.full_like(z_cmb, np.nan)
    
    # Se passou na verificação, calculamos o log seguramente
    correction = 5.0 * np.log10(arg_log)
    
    mu_model[mask] = log_cz[mask] + correction + 25.0
    
    return mu_model
