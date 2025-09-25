# skill_core/rarity.py
def tier(score: float) -> str:
    s = float(score)
    if s >= 90: return "S"     # top
    if s >= 75: return "A"
    if s >= 60: return "B"
    if s >= 45: return "C"
    if s >= 30: return "D"
    return "F"

def rarity_label(score: float) -> str:
    s = float(score)
    if s >= 90: return "Legendary"
    if s >= 75: return "Epic"
    if s >= 60: return "Rare"
    if s >= 45: return "Uncommon"
    return "Common"
