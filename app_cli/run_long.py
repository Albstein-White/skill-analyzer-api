
from __future__ import annotations
import os, datetime, time
from skill_core.types import Answer
from skill_core.engine import AdaptiveSession
from skill_core.report_html import export_report_html
from skill_core.config import load_config, get_backend
def choose_backend():
    cfg = load_config(); b = get_backend(cfg)
    if b: print(f"Using backend: {b}"); return
    print("Choose LLM backend for OPEN items: [0] None  [1] Ollama  [2] Azure")
    c = input("Your choice: ").strip()
    if c == "1": os.environ["USE_LLM_OPEN"]="1"; os.environ["LLM_BACKEND"]="ollama"
    elif c == "2": os.environ["USE_LLM_OPEN"]="1"; os.environ["LLM_BACKEND"]="azure"
    else: os.environ["USE_LLM_OPEN"]="0"
def ask(prompt: str, options=None) -> str:
    if options:
        print(prompt)
        for i,opt in enumerate(options): print(f"  [{i}] {opt}")
        while True:
            v = input("Your choice (index): ").strip()
            if v.isdigit(): return v
            print("Enter a number index.")
    else:
        return input(prompt + " ").strip()
def main():
    print("Long Skill Test v5")
    choose_backend()
    session = AdaptiveSession(run_type="long")
    while True:
        item = session.next_item()
        if item is None: break
        qtxt = f"(1-5) {item.text}  [1=strongly disagree, 5=strongly agree]" if item.type=='SR' else item.text
        t0 = time.perf_counter(); v = ask(qtxt, item.options) if item.options else ask(qtxt); rt = time.perf_counter() - t0
        session.answer_current(Answer(item_id=item.id, value=v, rt_sec=rt))
    res = session.finalize()
    from skill_core.calibration import apply_to_result
    res = apply_to_result(res, backend=(os.getenv("LLM_BACKEND") or "none"))
    os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_report_html(res, os.path.join("reports", f"report_long_{ts}.html"))
    print(f"Done. Report saved to: {path}")
if __name__ == "__main__": main()
