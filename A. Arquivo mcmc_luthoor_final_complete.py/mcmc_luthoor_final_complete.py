#!/usr/bin/env python3
"""
mcmc_luthoor_final_complete.py
Versão final pronta para execução — leitor robusto + cov compacto (Pantheon-style) +
fallback diagonal + regularização + MCMC (emcee) + plots e summary.

Exemplo de uso:
python3 mcmc_luthoor_final_complete.py --data lcparam.txt --cov cov.txt \
    --mu-module my_mu.py --mu-func mu_theory --param-names B C M \
    --init 0.0 0.0 -19.344 --bounds -2 2 -2 2 -30 -10 --nwalkers 64 --nsteps 2000
"""
from __future__ import annotations
import argparse
import os
import time
import json
import importlib
import importlib.util
import multiprocessing as mp
from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import emcee
import matplotlib.pyplot as plt

try:
    import corner
    _HAS_CORNER = True
except Exception:
    _HAS_CORNER = False

# -------------------------
# Utility: read file and normalize decimal commas -> dots (safe)
# -------------------------
def _read_text_replace_decimal(filename: str) -> str:
    """
    Retorna o conteúdo do arquivo com vírgulas trocadas por pontos.
    É uma substituição global, útil se seus arquivos usam ',' como separador decimal.
    Se você tiver vírgulas como separador de campo (CSV), não use esta função.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        txt = f.read()
    # Substitui vírgula decimal (assumimos que seu arquivo usa espaços como separador de colunas)
    # Substituição simples: troca todas as vírgulas por pontos — geralmente seguro para seus dados.
    txt_fixed = txt.replace(',', '.')
    return txt_fixed

# -------------------------
# Loader for Hubble diagram
# -------------------------
def load_hubble_diagram(filename: str, alpha: float = 0.14, beta: float = 3.1, dropna: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    read_txt = _read_text_replace_decimal(filename)
    # read via pandas from string
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(read_txt), delim_whitespace=True, comment='#', header=0)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler '{filename}': {e}")

    # candidate names (case insensitive)
    z_candidates = ['zcmb', 'z', 'zhd', 'z_hd']
    mB_candidates = ['mb', 'm_b', 'mb_obs', 'magnitude', 'm_bcorr', 'm_b_corr', 'm_b_corr']
    x1_candidates = ['x1', 'stretch']
    c_candidates = ['c', 'color', 'cor']
    mu_candidates = ['mu', 'mu_obs', 'muobs', 'distance_modulus']

    col_map = {col.lower(): col for col in df.columns}

    def find_col(cands: List[str]):
        for cand in cands:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    z_col = find_col(z_candidates)
    if z_col is None:
        raise RuntimeError(f"Coluna de redshift não encontrada. Colunas disponíveis: {list(df.columns)}")

    mu_col = find_col(mu_candidates)
    if mu_col is not None:
        z = pd.to_numeric(df[z_col], errors='coerce').astype(float)
        mu_obs = pd.to_numeric(df[mu_col], errors='coerce').astype(float)
        print(f"[Loader] Usando mu direto '{mu_col}' e redshift '{z_col}'.")
    else:
        mB_col = find_col(mB_candidates)
        x1_col = find_col(x1_candidates)
        c_col = find_col(c_candidates)
        if mB_col is None or x1_col is None or c_col is None:
            raise RuntimeError(
                "Arquivo não contém nem coluna 'mu' nem (mB, x1, c).\n"
                f"Procurados mB em {mB_candidates}, x1 em {x1_candidates}, c em {c_candidates}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )
        z = pd.to_numeric(df[z_col], errors='coerce').astype(float)
        mB = pd.to_numeric(df[mB_col], errors='coerce').astype(float)
        x1 = pd.to_numeric(df[x1_col], errors='coerce').astype(float)
        c = pd.to_numeric(df[c_col], errors='coerce').astype(float)
        mu_obs = mB + alpha * x1 - beta * c
        print(f"[Loader] Construindo mu_obs a partir de ({mB_col}, {x1_col}, {c_col}) com alpha={alpha}, beta={beta}.")

    df_used = df.copy()
    df_used['_z_'] = z
    df_used['_mu_obs_'] = mu_obs

    mask_valid = np.isfinite(df_used['_z_']) & np.isfinite(df_used['_mu_obs_'])
    n_total = len(df_used)
    n_valid = int(mask_valid.sum())
    if n_valid < n_total:
        msg = f"[Loader] {n_total - n_valid} linhas com valores inválidos detectadas."
        if dropna:
            print(msg + " Serão removidas.")
            df_used = df_used.loc[mask_valid].copy()
        else:
            raise RuntimeError(msg)

    z_final = df_used['_z_'].to_numpy(dtype=float)
    mu_final = df_used['_mu_obs_'].to_numpy(dtype=float)
    if z_final.size == 0:
        raise RuntimeError("Nenhuma linha válida após filtragem.")
    print(f"[Loader] OK. N = {z_final.size} supernovas usadas.")
    return z_final, mu_final, df_used

# -------------------------
# Covariance compact/full reader
# -------------------------
def load_covariance_compact(filename: str) -> np.ndarray:
    txt = _read_text_replace_decimal(filename)
    toks: List[float] = []
    for ln in txt.splitlines():
        s = ln.strip()
        if s == '' or s.startswith('#'):
            continue
        parts = s.split()
        for p in parts:
            try:
                toks.append(float(p))
            except Exception:
                # ignore
                continue
    arr = np.array(toks, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Arquivo de covariância não contém números.")

    # Perfect square -> full matrix
    s = int(np.sqrt(arr.size))
    if s * s == arr.size:
        C = arr.reshape((s, s))
        return C

    # compact format with leading N
    N = int(arr[0])
    expected = 1 + N * (N + 1) // 2
    if arr.size == expected:
        elems = arr[1:]
        C = np.zeros((N, N), dtype=float)
        k = 0
        for i in range(N):
            for j in range(i + 1):
                C[i, j] = elems[k]
                C[j, i] = elems[k]
                k += 1
        return C

    raise RuntimeError(f"Formato de covariância inesperado. Tokens: {arr.size}, esperado full N^2 ou compact 1+N*(N+1)/2.")

# -------------------------
# Prepare covariance (return cov, cho, eps, logdet)
# -------------------------
def prepare_covariance(cov_raw: Optional[np.ndarray], df: pd.DataFrame, max_tries: int = 12, initial_eps: Optional[float] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, bool], float, float]:
    N = len(df)
    # find stat error column
    candidates = ['dmb', 'sigma_mB', 'sigma_mb', 'err_mB', 'mberr', 'm_b_err', 'm_berr']
    col_map = {c.lower(): c for c in df.columns}
    stat_col = None
    for cand in candidates:
        if cand.lower() in col_map:
            stat_col = col_map[cand.lower()]
            break

    if cov_raw is None:
        if stat_col is None:
            # fallback weak: use uniform 0.1 mag if nothing exists
            sigma_int_fallback = 0.10
            print("[Cov] Nenhuma cov raw e nenhuma coluna de erro estatístico encontrada. Usando sigma=0.10 mag como fallback.")
            cov = np.diag(np.full(N, sigma_int_fallback**2))
        else:
            sigma = pd.to_numeric(df[stat_col], errors='coerce').astype(float)
            if np.any(~np.isfinite(sigma)):
                raise RuntimeError(f"Coluna de erro '{stat_col}' contém valores inválidos.")
            cov = np.diag(sigma**2)
            print(f"[Cov] Construído fallback diagonal usando '{stat_col}'.")
    else:
        cov = np.array(cov_raw, dtype=float)
        if cov.shape != (N, N):
            raise RuntimeError(f"Dimensão da covariância ({cov.shape}) diferente de N_data ({N}).")
        if stat_col is not None:
            sigma = pd.to_numeric(df[stat_col], errors='coerce').astype(float)
            if np.any(~np.isfinite(sigma)):
                raise RuntimeError(f"Coluna de erro '{stat_col}' contém valores inválidos.")
            cov = cov + np.diag(sigma**2)
            print(f"[Cov] Adicionando erro estatístico da coluna '{stat_col}' à covariância sistêmica.")
        else:
            print("[Cov] Usando cov raw (sem adicionar erro estatístico).")

    # try cholesky
    try:
        cho = cho_factor(cov, overwrite_a=False, lower=False)
        c_factor = cho[0]
        diag = np.diag(c_factor)
        logdet = 2.0 * np.sum(np.log(np.abs(diag)))
        return cov, cho, 0.0, logdet
    except (LinAlgError, ValueError):
        # regularize
        diag = np.diag(cov)
        diag_mean = float(np.mean(np.abs(diag))) if diag.size > 0 else 0.0
        eps = initial_eps if initial_eps is not None else (diag_mean * 1e-10 if diag_mean > 0 else 1e-12)
        eps = max(eps, 1e-16)
        for attempt in range(max_tries):
            C_reg = cov.copy()
            C_reg[np.diag_indices(N)] += eps
            try:
                cho = cho_factor(C_reg, overwrite_a=False, lower=False)
                c_factor = cho[0]
                diag = np.diag(c_factor)
                logdet = 2.0 * np.sum(np.log(np.abs(diag)))
                print(f"[Cov] Regularização bem sucedida com eps = {eps:.3e}")
                return C_reg, cho, eps, logdet
            except (LinAlgError, ValueError):
                eps *= 10.0
        raise RuntimeError(f"Não foi possível regularizar a matriz após {max_tries} tentativas.")

# -------------------------
# Import mu function (cached)
# -------------------------
_MU_CACHE = {}

def get_mu_raw(mu_module_path: str, mu_func_name: str) -> Callable:
    key = f"{mu_module_path}::{mu_func_name}"
    if key in _MU_CACHE:
        return _MU_CACHE[key]

    if mu_module_path.endswith('.py') or os.path.exists(mu_module_path):
        spec = importlib.util.spec_from_file_location("mu_module", mu_module_path)
        if spec is None:
            raise ImportError(f"Não foi possível carregar spec de {mu_module_path}")
        mod = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(mod)
    else:
        mod = importlib.import_module(mu_module_path)

    if not hasattr(mod, mu_func_name):
        raise AttributeError(f"O módulo importado não possui a função '{mu_func_name}'")
    func = getattr(mod, mu_func_name)
    _MU_CACHE[key] = func
    return func

def make_mu_adapter_if_needed(mu_raw: Callable) -> Callable:
    import inspect as _inspect
    try:
        sig = _inspect.signature(mu_raw)
        params_keys = list(sig.parameters.keys())
    except Exception:
        params_keys = []

    if len(params_keys) >= 3 and params_keys[1] != 'params':
        # assume signature (z, B, C, ...)
        def mu_adapter(z_array, params_dict):
            B = params_dict.get('B')
            C = params_dict.get('C')
            M = params_dict.get('M', 0.0)
            if B is None or C is None:
                raise ValueError("B e C obrigatórios em params.")
            mu_th = mu_raw(z_array, B, C)
            return np.asarray(mu_th, dtype=float) + M
        return mu_adapter
    else:
        # assume (z, params_dict)
        def mu_adapter2(z_array, params_dict):
            mu_th = mu_raw(z_array, params_dict)
            return np.asarray(mu_th, dtype=float)
        return mu_adapter2

# -------------------------
# Log-probability
# -------------------------
def log_probability(theta: np.ndarray, z: np.ndarray, mu_obs: np.ndarray, cho: Tuple[np.ndarray, bool], logdet: float,
                    bounds: Dict[str, Tuple[float, float]], mu_func: Callable, param_names: Tuple[str, ...]) -> float:
    # uniform prior in bounds
    for i, name in enumerate(param_names):
        mn, mx = bounds[name]
        if not (mn < theta[i] < mx):
            return -np.inf

    params = {param_names[i]: float(theta[i]) for i in range(len(param_names))}
    try:
        mu_model = mu_func(z, params)
        mu_model = np.asarray(mu_model, dtype=float)
    except Exception:
        return -np.inf

    if mu_model.shape != mu_obs.shape:
        return -np.inf

    try:
        diff = mu_obs - mu_model
        x = cho_solve(cho, diff)
        chi2 = float(np.dot(diff, x))
        N = z.size
        ll = -0.5 * (chi2 + logdet + N * np.log(2.0 * np.pi))
        return ll
    except Exception:
        return -np.inf

# -------------------------
# MCMC runner
# -------------------------
def run_mcmc(z: np.ndarray, mu_obs: np.ndarray, cho: Tuple[np.ndarray, bool], logdet: float,
             initial: np.ndarray, bounds: Dict[str, Tuple[float, float]], param_names: Tuple[str, ...],
             mu_func: Callable, nwalkers: int = 64, nsteps: int = 2000, burn: int = 500,
             threads: int = 1, outdir: str = 'mcmc_out', init_disp: float = 1e-2, seed: Optional[int] = None):
    ndim = len(initial)
    if nwalkers <= ndim:
        raise ValueError("nwalkers deve ser > ndim")

    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % (2**31 - 1))
    p0 = np.array(initial, dtype=float)
    scale = init_disp * (np.abs(p0) + 1e-6)
    pos = p0 + rng.normal(scale=scale, size=(nwalkers, ndim))

    os.makedirs(outdir, exist_ok=True)

    pool = None
    if threads is not None and threads > 1:
        pool = mp.Pool(processes=threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(z, mu_obs, cho, logdet, bounds, mu_func, param_names),
                                        pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(z, mu_obs, cho, logdet, bounds, mu_func, param_names))

    print(f"[MCMC] começando: nwalkers={nwalkers}, nsteps={nsteps}, ndim={ndim}, threads={threads}, seed={seed}")
    sampler.run_mcmc(pos, nsteps, progress=True)
    print("[MCMC] finalizado.")

    if pool is not None:
        pool.close()
        pool.join()

    samples = sampler.get_chain(discard=burn, flat=False)
    flat_samples = sampler.get_chain(discard=burn, flat=True)
    logprobs = sampler.get_log_prob(discard=burn, flat=True)

    np.save(os.path.join(outdir, 'chain.npy'), samples)
    np.save(os.path.join(outdir, 'flat_samples.npy'), flat_samples)
    np.save(os.path.join(outdir, 'logprob.npy'), logprobs)

    med = np.median(flat_samples, axis=0)
    p16 = np.percentile(flat_samples, 16, axis=0)
    p84 = np.percentile(flat_samples, 84, axis=0)
    summary = {param_names[i]: {'median': float(med[i]), 'p16': float(p16[i]), 'p84': float(p84[i])}
               for i in range(ndim)}

    try:
        tau = sampler.get_autocorr_time(discard=burn)
        tau_list = [float(t) for t in np.atleast_1d(tau)]
    except Exception:
        tau_list = [float('nan')] * ndim
        tau = None

    if tau is not None and np.all(np.isfinite(tau)):
        n_eff = float(flat_samples.shape[0]) / np.array(tau)
        ess_list = [float(x) for x in np.atleast_1d(n_eff)]
    else:
        ess_list = [float('nan')] * ndim

    try:
        acc_frac = sampler.acceptance_fraction.tolist()
        acc_mean = float(np.mean(sampler.acceptance_fraction))
    except Exception:
        acc_frac = None
        acc_mean = float('nan')

    diagnostics = {
        'autocorr_time': tau_list,
        'ESS': ess_list,
        'acceptance_fraction_mean': acc_mean,
        'acceptance_fraction_per_walker': acc_frac,
        'n_flat_samples': int(flat_samples.shape[0])
    }

    out_summary = {
        'parameters': summary,
        'diagnostics': diagnostics,
        'run_metadata': {
            'nwalkers': int(nwalkers), 'nsteps': int(nsteps), 'burn': int(burn), 'ndim': int(ndim),
            'threads': int(threads), 'seed': seed
        }
    }

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(out_summary, f, indent=2)

    # walkers plot
    try:
        plt.figure(figsize=(10, 2.5 * ndim))
        for i in range(ndim):
            ax = plt.subplot(ndim, 1, i + 1)
            ax.plot(sampler.get_chain()[:, :, i], alpha=0.4)
            ax.set_ylabel(param_names[i])
        plt.xlabel("step")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'walkers.png'), dpi=150)
        plt.close()
    except Exception as e:
        print("Erro ao salvar walkers plot:", e)

    # corner
    if _HAS_CORNER and flat_samples.size > 0:
        try:
            fig = corner.corner(flat_samples, labels=param_names, truths=med, show_titles=True)
            fig.savefig(os.path.join(outdir, 'corner.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print("Erro ao gerar corner plot:", e)

    print(f"[MCMC] resultados salvos em {outdir}")
    return out_summary, flat_samples, sampler

# -------------------------
# CLI
# -------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MCMC para Modelo Luthoor (Final completo)")
    p.add_argument("--data", required=True, help="Arquivo da tabela Hubble (z, mu_obs ou mB/x1/c)")
    p.add_argument("--cov", required=True, help="Arquivo da matriz de covariância (compact/full)")
    p.add_argument("--mu-module", type=str, required=True, help="Módulo Python (path .py ou package) com a função mu")
    p.add_argument("--mu-func", type=str, required=True, help="Nome da função no módulo que calcula mu")
    p.add_argument("--alpha", type=float, default=0.14, help="Parâmetro alpha (Tripp)")
    p.add_argument("--beta", type=float, default=3.1, help="Parâmetro beta (Tripp)")
    p.add_argument("--param-names", nargs='+', required=True, help="Nomes dos parâmetros a ajustar (ex: B C M)")
    p.add_argument("--init", nargs='+', required=True, help="Valores iniciais (ex: 0.0 0.0 -19.344)")
    p.add_argument("--bounds", nargs='+', required=True, help="Limites (min max ...) por parâmetro")
    p.add_argument("--nwalkers", type=int, default=64)
    p.add_argument("--nsteps", type=int, default=3000)
    p.add_argument("--burn", type=int, default=800)
    p.add_argument("--init-disp", type=float, default=1e-2)
    p.add_argument("--threads", type=int, default=1, help="Número de processos (multiprocessing.Pool)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default="mcmc_out")
    p.add_argument("--cov-max-tries", type=int, default=12)
    p.add_argument("--cov-init-eps", type=float, default=None)
    return p.parse_args(argv)

# -------------------------
# main
# -------------------------
def main(argv=None):
    args = parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"[Main] Carregando dados de {args.data} ...")
    z, mu_obs, df = load_hubble_diagram(args.data, args.alpha, args.beta)
    N = len(z)
    print(f"[Main] N = {N} supernovas.")

    print(f"[Main] Carregando covariância de {args.cov} ...")
    try:
        cov_raw = load_covariance_compact(args.cov)
    except Exception as e:
        print(f"[Main] Falha ao ler cov raw: {e}")
        cov_raw = None

    try:
        cov_matrix, cho, eps, logdet = prepare_covariance(cov_raw, df, max_tries=args.cov_max_tries, initial_eps=args.cov_init_eps)
        if eps > 0.0:
            print(f"[Main] Aviso: cov regularizada (eps={eps:.3e})")
    except Exception as e:
        print(f"[Main] Erro crítico na covariância: {e}")
        print("[Main] Tentando fallback diagonal (procura dmb / sigma_mB / ...).")
        candidates = ['dmb', 'sigma_mB', 'sigma_mb', 'err_mB', 'mberr', 'm_b_err']
        col_map = {c.lower(): c for c in df.columns}
        stat_col = None
        for cand in candidates:
            if cand.lower() in col_map:
                stat_col = col_map[cand.lower()]
                break
        if stat_col is None:
            # fallback hard: uniform sigma=0.10
            sigma_int = 0.10
            print("[Main] Nenhuma coluna de erro encontrada. Usando sigma=0.10 mag como fallback.")
            cov_matrix = np.diag(np.full(N, sigma_int**2))
        else:
            sigma = pd.to_numeric(df[stat_col], errors='coerce').astype(float)
            if np.any(~np.isfinite(sigma)):
                raise RuntimeError("Coluna de erro estatístico contém valores inválidos.")
            cov_matrix = np.diag(sigma**2)
            print(f"[Main] Fallback diagonal construído usando coluna '{stat_col}'.")
        cho = cho_factor(cov_matrix, overwrite_a=False, lower=False)
        c_factor = cho[0]
        diag = np.diag(c_factor)
        logdet = 2.0 * np.sum(np.log(np.abs(diag)))
        eps = 0.0

    if cov_matrix.shape != (N, N):
        raise RuntimeError(f"cov matrix shape {cov_matrix.shape} incompatible with data length {N}")

    # params, init, bounds
    param_names = tuple(args.param_names)
    init = np.array([float(x) for x in args.init], dtype=float)
    if init.size != len(param_names):
        raise RuntimeError("Número de valores em --init deve ser igual a número de --param-names")
    if len(args.bounds) != 2 * len(param_names):
        raise RuntimeError("Quando informado, --bounds precisa de 2 valores por parâmetro (min max ...).")
    bounds = {param_names[i]: (float(args.bounds[2 * i]), float(args.bounds[2 * i + 1])) for i in range(len(param_names))}

    # import mu
    mu_raw = get_mu_raw(args.mu_module, args.mu_func)
    mu_func = make_mu_adapter_if_needed(mu_raw)

    # smoke test
    try:
        test_params = {param_names[i]: float(init[i]) for i in range(len(param_names))}
        mu_try = mu_func(z[:3], test_params)
        mu_try = np.asarray(mu_try)
        if mu_try.shape[0] != 3:
            raise RuntimeError("mu_func smoke test retornou shape inesperado.")
    except Exception as e:
        raise RuntimeError(f"Erro no smoke test da mu_func: {e}")

    # run MCMC
    out_summary, flat_samples, sampler = run_mcmc(z, mu_obs, cho, logdet,
                                                  initial=init,
                                                  bounds=bounds,
                                                  param_names=param_names,
                                                  mu_func=mu_func,
                                                  nwalkers=args.nwalkers,
                                                  nsteps=args.nsteps,
                                                  burn=args.burn,
                                                  threads=args.threads,
                                                  outdir=args.out,
                                                  init_disp=args.init_disp,
                                                  seed=args.seed)

    print("Final summary (saved em summary.json):")
    print(json.dumps(out_summary, indent=2))

if __name__ == '__main__':
    main()