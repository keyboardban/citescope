"""Built-in topic question packs + a paste-block parser.

Three focus topics, each a list of {id, intent, prompt}. Used by Topic Studies mode
to throw many prompts at the pipeline and compare cited vs non-cited patterns by
topic and by intent. Users can also paste their own questions.
"""

from __future__ import annotations

import re

TOPIC_EMOJI = {"Healthcare / Skincare": "🧴", "Automotive": "🚗", "Real Estate": "🏠"}

TOPIC_SETS: dict[str, list[dict]] = {
    "Healthcare / Skincare": [
        {"id": "H01", "intent": "Informational", "prompt": "What ingredients should I look for in a moisturizer for dry and sensitive skin?"},
        {"id": "H02", "intent": "Comparison", "prompt": "Ceramide cream vs hyaluronic acid cream: which is better for repairing the skin barrier?"},
        {"id": "H03", "intent": "Product/Recommendation", "prompt": "What are the best dermatologist-recommended moisturizers for sensitive skin?"},
        {"id": "H04", "intent": "Safety/Regulatory", "prompt": "How can I check whether a skincare cream is safe and properly registered in Thailand?"},
        {"id": "H05", "intent": "Ingredient Analysis", "prompt": "Is niacinamide cream good for acne-prone skin, and what concentration is commonly recommended?"},
        {"id": "H06", "intent": "Consumer Decision", "prompt": "How do I choose between a whitening cream, brightening cream, and barrier repair cream?"},
        {"id": "H07", "intent": "Myth/Fact", "prompt": "Do collagen creams actually improve skin elasticity, or is it mostly marketing?"},
        {"id": "H08", "intent": "Local/Market", "prompt": "What are popular skincare cream brands in Thailand for sensitive skin?"},
        {"id": "H09", "intent": "Problem-Solution", "prompt": "What type of cream is suitable for irritated skin after using acne treatment products?"},
        {"id": "H10", "intent": "Freshness/Trend", "prompt": "What skincare ingredients are currently trending for barrier repair and hydration?"},
        {"id": "H11", "intent": "Source Authority Test", "prompt": "What do dermatologists say about using steroid creams without prescription?"},
        {"id": "H12", "intent": "Brand/Official Source Test", "prompt": "How can consumers verify product claims made by a skincare cream brand?"},
    ],
    "Automotive": [
        {"id": "A01", "intent": "Informational", "prompt": "What factors should I consider before buying an electric vehicle in Thailand?"},
        {"id": "A02", "intent": "Comparison", "prompt": "Hybrid vs electric cars: which is more cost-effective for daily driving in Bangkok?"},
        {"id": "A03", "intent": "Product/Recommendation", "prompt": "What are the best family SUVs available in Thailand right now?"},
        {"id": "A04", "intent": "Cost Analysis", "prompt": "What is the total cost of ownership for an EV compared with a gasoline car?"},
        {"id": "A05", "intent": "Maintenance", "prompt": "Are electric vehicles cheaper to maintain than internal combustion engine cars?"},
        {"id": "A06", "intent": "Charging Infrastructure", "prompt": "How available are EV charging stations in Thailand, and what should buyers check?"},
        {"id": "A07", "intent": "Safety", "prompt": "What car safety features are most important when buying a new car?"},
        {"id": "A08", "intent": "Used Car", "prompt": "What should I check before buying a used car in Thailand?"},
        {"id": "A09", "intent": "Brand Comparison", "prompt": "Toyota hybrid vs BYD electric car: which is better for city driving?"},
        {"id": "A10", "intent": "Regulation/Policy", "prompt": "What government incentives or tax benefits exist for electric vehicles in Thailand?"},
        {"id": "A11", "intent": "Market Trend", "prompt": "What are the current automotive market trends in Thailand?"},
        {"id": "A12", "intent": "Official Source Test", "prompt": "Where should buyers verify official car specifications and warranty information?"},
    ],
    "Real Estate": [
        {"id": "R01", "intent": "Informational", "prompt": "What factors should I consider before buying a condominium in Bangkok?"},
        {"id": "R02", "intent": "Location Comparison", "prompt": "Asoke vs Thonglor vs Langsuan: which area is better for condo investment?"},
        {"id": "R03", "intent": "Investment", "prompt": "Is buying a condo in Bangkok still a good investment?"},
        {"id": "R04", "intent": "Price/Market", "prompt": "What affects condominium prices in central Bangkok?"},
        {"id": "R05", "intent": "Rental Yield", "prompt": "Which Bangkok areas usually have strong rental demand for condos?"},
        {"id": "R06", "intent": "Buyer Guide", "prompt": "What documents should I check before buying a condominium in Thailand?"},
        {"id": "R07", "intent": "Legal/Regulatory", "prompt": "What should foreigners know before buying property in Thailand?"},
        {"id": "R08", "intent": "Developer Comparison", "prompt": "How should buyers compare real estate developers before purchasing a condo?"},
        {"id": "R09", "intent": "Project Evaluation", "prompt": "What makes a condominium project high quality?"},
        {"id": "R10", "intent": "Risk Analysis", "prompt": "What are the main risks of investing in off-plan condominium projects?"},
        {"id": "R11", "intent": "Current Market", "prompt": "What is the current trend of Bangkok condo prices and demand?"},
        {"id": "R12", "intent": "Official Source Test", "prompt": "Where can buyers verify official project information, land details, or legal documents?"},
    ],
}

_ID_RE = re.compile(r"^[A-Za-z]{1,4}\d{1,3}$")


def items_for(topics: list[str]) -> list[dict]:
    """Flatten selected topic packs into tagged items (topic added to each)."""
    out: list[dict] = []
    for t in topics:
        for it in TOPIC_SETS.get(t, []):
            out.append({**it, "topic": t})
    return out


def all_items() -> list[dict]:
    return items_for(list(TOPIC_SETS))


def simple_prompts(text: str, default_topic: str = "Custom") -> list[dict]:
    """Treat every non-empty line as a full prompt — no ID/intent parsing.

    Use this for the common case: paste many prompts, one per line. Special
    characters like '|' or extra spaces are kept verbatim (never split).
    """
    items: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("id ") and "intent" in low and "prompt" in low:
            continue  # skip an obvious header row if pasted by accident
        items.append({"id": "", "intent": "Custom", "prompt": line, "topic": default_topic})
    return items


def parse_prompt_block(text: str, default_topic: str = "Custom") -> list[dict]:
    """Parse pasted questions.

    Accepts (per line): 'ID<TAB>Intent<TAB>Prompt', 'ID | Intent | Prompt',
    'ID   Intent   Prompt' (2+ spaces), 'Intent | Prompt', or a bare prompt.
    A header line like 'ID Intent Prompt' is skipped.
    """
    items: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("id") and "intent" in low and "prompt" in low:
            continue  # header row

        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        elif "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
        elif re.search(r"\s{2,}", line):
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
        else:
            parts = [line]

        pid, intent, prompt = "", "", ""
        if len(parts) >= 3 and _ID_RE.match(parts[0]):
            pid, intent, prompt = parts[0], parts[1], " ".join(parts[2:])
        elif len(parts) == 2:
            if _ID_RE.match(parts[0]):
                pid, prompt = parts[0], parts[1]
            else:
                intent, prompt = parts[0], parts[1]
        else:
            prompt = " ".join(parts)

        prompt = prompt.strip()
        if not prompt:
            continue
        items.append({"id": pid, "intent": intent or "Custom", "prompt": prompt, "topic": default_topic})
    return items
