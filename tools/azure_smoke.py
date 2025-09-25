# tools/azure_smoke.py
from __future__ import annotations
from openai import NotFoundError
from skill_core.azure_cfg import client, settings

def main():
    s = settings()
    print("Endpoint :", s.endpoint)
    print("Deploy   :", s.deployment, "(deployment name passed as model=)")
    print("API ver  :", s.api_version)
    cli = client()
    try:
        r = cli.chat.completions.create(
            model=s.deployment,                       # deployment name, not model family
            messages=[{"role":"user","content":"Say 'pong' only."}],
            temperature=0.0,
            max_tokens=5,
        )
        print("Reply    :", r.choices[0].message.content)
    except NotFoundError as e:
        print("ERROR 404: Azure cannot find this deployment for this API version.")
        print("→ Verify the deployment name EXACTLY as in the portal.")
        print("→ Ensure api_version matches the portal’s ‘Target URI’ (you set 2025-01-01-preview).")
        raise
    except Exception as e:
        print("Azure call failed:", e)
        raise

if __name__ == "__main__":
    main()
