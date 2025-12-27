#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import emcee
import pandas as pd
from multiprocessing import Pool
import time
import importlib

def log_likelihood(theta, z_cmb, z_hel, mu_obs, inv_cov, mu_func):
    B, C, M = theta
    
    # Chama a função de física
    mu_theory_dist = mu_func(z_cmb, z_hel, [B, C, M])
    
    # --- FILTRO DE REALIDADE ---
    # Se a função retornou NaNs (silêncio matemático), rejeitamos imediatamente.
    if np.any(np.isnan(mu_theory_dist)):
        return -np.inf
        
    model = mu_theory_dist + M 
    diff = mu_obs - model
    chi2 = np.dot(diff, np.dot(inv_cov, diff))
    return -0.5 * chi2

def log_prior(theta, bounds):
    B, C, M = theta
    b_min, b_max, c_min, c_max, m_min, m_max = bounds
    
    # Prior Uniforme (Boxcar) dentro dos limites definidos
    if (b_min < B < b_max) and (c_min < C < c_max) and (m_min < M < m_max):
        return 0.0
    return -np.inf

def log_probability(theta, z_cmb, z_hel, mu_obs, inv_cov, mu_func, bounds):
    # 1. Verifica limites rígidos (Prior)
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    
    # 2. Verifica realidade física (Likelihood)
    return lp + log_likelihood(theta, z_cmb, z_hel, mu_obs, inv_cov, mu_func)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--cov', required=True)
    parser.add_argument('--mu-module', required=True)
    parser.add_argument('--mu-func', required=True)
    parser.add_argument('--alpha', type=float, default=0.14)
    parser.add_argument('--beta', type=float, default=3.1)
    parser.add_argument('--param-names', nargs='+', required=True)
    parser.add_argument('--init', nargs='+', type=float, required=True)
    parser.add_argument('--bounds', nargs='+', type=float, required=True)
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument('--burn', type=int, default=200)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='chain_result')
    
    args = parser.parse_args()
    
    print(f"--- MCMC SETUP V10: Geometria Consciente ---")
    
    # 1. LEITURA DOS DADOS
    print("Lendo Supernovas...")
    try:
        # Tenta ler colunas padrão Pantheon+
        data_arr = np.loadtxt(args.data, skiprows=1, usecols=(1, 2, 4, 5))
        z_cmb = data_arr[:, 0]
        z_hel = data_arr[:, 1]
        mb = data_arr[:, 2]
        dmb = data_arr[:, 3]
        print(f"✅ {len(z_cmb)} observáveis carregados.")
    except:
        # Fallback Pandas
        df = pd.read_csv(args.data, sep=r'\s+')
        z_cmb = df['zcmb'].values
        z_hel = df['zhel'].values
        mb = df['mb'].values
        dmb = df['dmb'].values
        print(f"✅ {len(z_cmb)} observáveis carregados via Pandas.")

    # 2. COVARIÂNCIA + ERRO ESTATÍSTICO
    print("Configurando Matriz de Covariância...")
    cov_raw = np.loadtxt(args.cov)
    N = len(z_cmb)
    
    # Ajuste de forma (Linear -> Quadrada)
    if cov_raw.size == N*N + 1:
        cov = cov_raw[1:].reshape((N, N))
    elif cov_raw.size == N*N:
        cov = cov_raw.reshape((N, N))
    else:
        cov = cov_raw
        
    # Soma erro estatístico na diagonal (Evita singularidade)
    cov_total = cov + np.diag(dmb**2)
    inv_cov = np.linalg.inv(cov_total)
    print("✅ Matriz pronta e invertida.")
    
    # 3. SETUP E RUN
    sys.path.append('.')
    mod_name = args.mu_module.replace('.py', '')
    mod = importlib.import_module(mod_name)
    mu_func = getattr(mod, args.mu_func)
    
    ndim = len(args.init)
    nwalkers = args.nwalkers
    p0 = np.array(args.init) + 1e-4 * np.random.randn(nwalkers, ndim)
    bounds_flat = args.bounds 
    
    print(f"🚀 Iniciando a busca pelo singular ({args.nsteps} passos)...")
    with Pool(processes=args.threads) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, 
            args=[z_cmb, z_hel, mb, inv_cov, mu_func, bounds_flat],
            pool=pool
        )
        
        start = time.time()
        sampler.run_mcmc(p0, args.nsteps, progress=True)
        end = time.time()
        
    print(f"🏁 Concluído em {(end-start)/60:.2f} minutos.")
    
    # Salvar
    flat_samples = sampler.get_chain(discard=args.burn, flat=True)
    np.save(args.out + "_chain.npy", flat_samples)
    
    for i, name in enumerate(args.param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{name} = {mcmc[1]:.3f} (+{q[1]:.3f} / -{q[0]:.3f})")

if __name__ == "__main__":
    main()
