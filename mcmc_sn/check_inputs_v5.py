#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Dry Run - Check Inputs")
    parser.add_argument('--data', required=True)
    parser.add_argument('--cov', required=True)
    parser.add_argument('--mu-module', required=True)
    parser.add_argument('--mu-func', required=True)
    parser.add_argument('--param-names', nargs='+', required=True)

    args = parser.parse_args()

    print("✔ DRY RUN OK (Arquivo limpo criado via Colab!)")
    print("Data:", args.data)
    print("Cov :", args.cov)
    print("Mu module:", args.mu_module)
    print("Mu func  :", args.mu_func)
    print("Params   :", args.param_names)

if __name__ == "__main__":
    main()
