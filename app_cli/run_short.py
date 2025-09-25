
from __future__ import annotations
import os, datetime, time
from skill_core.types import Answer
from skill_core.engine import AdaptiveSession
from skill_core.report_html import export_report_html
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
    print("Short Skill Test v5")
    session = AdaptiveSession(run_type="short")
    while True:
        item = session.next_item()
        if item is None: break
        qtxt = f"(1-5) {item.text}  [1=strongly disagree, 5=strongly agree]" if item.type=='SR' else item.text
        t0 = time.perf_counter(); v = ask(qtxt, item.options) if item.options else ask(qtxt); rt = time.perf_counter() - t0
        session.answer_current(Answer(item_id=item.id, value=v, rt_sec=rt))
    res = session.finalize(); os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_report_html(res, os.path.join("reports", f"report_short_{ts}.html"))
    print(f"Done. Report saved to: {path}")
if __name__ == "__main__": main()
