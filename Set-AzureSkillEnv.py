# Set_AzureSkillEnv.py
import os, json, argparse, subprocess, sys

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_DEPLOYMENT"):
        if not cfg.get(k):
            print(f"Missing {k} in {path}", file=sys.stderr); sys.exit(1)
    cfg.setdefault("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    cfg.setdefault("LLM_BACKEND", "azure")
    cfg.setdefault("USE_LLM_OPEN", "1")
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".azure_config.json")
    ap.add_argument("--run", nargs=argparse.REMAINDER,
                    help="Optional command to run with env set. Example: --run python autoplay.py --run long --profile perfect --llm azure")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    env = os.environ.copy(); env.update(cfg)

    if not args.run:
        # Just show what would be used
        print("Loaded Azure config:")
        for k in ("AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_DEPLOYMENT","AZURE_OPENAI_API_VERSION","LLM_BACKEND","USE_LLM_OPEN"):
            print(f"{k}={env.get(k)}")
        print("Note: Python cannot persist env to parent shell. Use the PowerShell script for persistence.")
        return

    print("Launching:", " ".join(args.run))
    subprocess.run(args.run, env=env, check=True)

if __name__ == "__main__":
    main()
