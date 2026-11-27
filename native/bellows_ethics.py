"""FvF Bellows Local Ethical Gatekeeper.

Hybrid two-layer safety screen enforcing consistent rules across ALL modalities:
  1. Fast deterministic normalizer + policy pattern/lexicon screen
     (race/ethnicity, age, disability, gender/identity, LGBTQ+ people/symbols,
      religions/symbols, countries/nationalities/governments/national symbols,
      political views/parties, specific real OR fictional named people,
      critical safety warnings).
  2. Local compact NLI zero-shot model (`MoritzLaurer/DeBERTa-v3-base-zeroshot-v2.0`)
     for slang, euphemisms, indirect references, and named individuals/characters.

Always unloads the gatekeeper model from RAM/VRAM immediately after checking a
prompt, while keeping downloaded weights cached on disk (`bellows_cache/gatekeeper`)
until the user decides whether to keep or delete Bellows caches post-training.
"""

from __future__ import annotations

import gc
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

GATEKEEPER_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-zeroshot-v2.0"

PROHIBITED_POLICY_LIST: List[Dict[str, str]] = [
    {
        "code": "specific_person",
        "title": "Specific People (Real or Fictional)",
        "detail": "Targeting or filtering out any named individual—including private people, celebrities, politicians, or fictional characters.",
    },
    {
        "code": "race_ethnicity",
        "title": "Race, Ethnicity & Heritage",
        "detail": "Filtering people by race, skin tone, ethnic background, ancestry, or caste.",
    },
    {
        "code": "age",
        "title": "Age & Age Groups",
        "detail": "Filtering people based on age or age group (children, teenagers, adults, or elderly seniors).",
    },
    {
        "code": "disability",
        "title": "Disability & Accessibility",
        "detail": "Filtering disabled people, physical or cognitive disabilities, mobility aids, or neurodivergence.",
    },
    {
        "code": "gender_identity",
        "title": "Gender, Sex & Gender Identity",
        "detail": "Filtering out women, men, transgender, or non-binary people.",
    },
    {
        "code": "lgbtq",
        "title": "LGBTQ+ People & Symbols",
        "detail": "Filtering LGBTQ+ individuals, same-sex relationships, pride flags, or queer symbols.",
    },
    {
        "code": "religion_symbols",
        "title": "Religions & Religious Symbols",
        "detail": "Filtering people of any faith, religious attire (hijabs, turbans, kippahs, habits), places of worship, or religious symbols.",
    },
    {
        "code": "country_government",
        "title": "Countries, Nationalities & Governments",
        "detail": "Filtering specific nations, nationalities, national flags/emblems, or governments.",
    },
    {
        "code": "political_views",
        "title": "Political Views, Parties & Ideologies",
        "detail": "Filtering specific political opinions, parties, ideologies, or policy stances.",
    },
    {
        "code": "safety_warnings",
        "title": "Critical Safety & Security Alerts",
        "detail": "Bypassing or hiding security warnings, emergency broadcasts, or accessibility/medical alerts.",
    },
]

_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
})

# Contexts explicitly permitted regardless of sensitive keywords
_ALLOWED_SAFETY_PATTERNS = [
    re.compile(r"\b(hate\s*speech|racist\s*slurs?|racial\s*slurs?|ableist\s*slurs?|homophobi[ac]|transphobi[ac]|antisemiti[cs]m?|islamophobi[ac]|bigotry|xenophobi[ac])\b", re.I),
    re.compile(r"\b(gore|gory|gorey|blood|graphic\s+violence|brutal\s+violence|self[-\s]*harm|suicid(?:e|al)|eating\s+disorder)\b", re.I),
    re.compile(r"\b(nsfw|porn|pornograph\w*|nudit\w*|explicit\s+sex\w*|hentai|erotic\w*|profanit\w*|swear\s*words?|slurs?)\b", re.I),
    re.compile(r"\b(child\s+abuse|child\s+exploitation|csam|grooming|pedophil\w*|paedophil\w*)\b", re.I),
    re.compile(r"\b(gambling\s+ads?|casino\s+ads?|betting\s+ads?|crypto\s+ads?|scam\s+ads?|brand\s+logos?|commercial\s+logos?)\b", re.I),
]

_HARD_RULES: List[tuple[str, re.Pattern[str], str]] = [
    (
        "race_ethnicity",
        re.compile(
            r"\b(black\s+people|white\s+people|asian\s+people|hispanic\s+people|latino\w*|latina\w*|"
            r"african\s+people|indian\s+people|arab\s+people|indigenous\s+people|native\s+americans?|"
            r"caucasian\w*|afro\w*|brown\s+skin|dark\s+skin|light\s+skin|minorities|ethnicit(?:y|ies)|"
            r"non[-\s]*white|poc|bipoc|ethnic\s+minorities|people\s+of\s+color|"
            r"n+i+g+g+\w*|k+i+k+e+s*|ch+i+n+k+s*|sp+i+c+s*|w+e+t+b+a+c+k+s*)\b",
            re.I,
        ),
        "Filtering people by race, ethnicity, or skin tone is not permitted.",
    ),
    (
        "age",
        re.compile(
            r"\b(old\s+people|elderly(?:\s+people)?|senior\s+citizens?|seniors\s+only|"
            r"boomers?|millennials?|gen\s*[zx]|zoomers?|teenagers?|teens\s+only|"
            r"toddlers?|babies|infants?|children|kids|middle[-\s]*aged\s+people|young\s+people|ageism)\b",
            re.I,
        ),
        "Filtering people based on age or age group is not permitted.",
    ),
    (
        "disability",
        re.compile(
            r"\b(disabled\s+people|people\s+with\s+disabilit\w*|handicapped|"
            r"wheelchair\s*(?:users?|bound)?|blind\s+people|deaf\s+people|"
            r"autistic\s+people|neurodivergent\s+people|down\s+syndrome|cerebral\s+palsy|"
            r"amputees?|prosthetics?|mentally\s+ill\s+people|"
            r"r+e+t+a+r+d+s*|r+e+t+a+r+d+e+d+|cripples?|spastics?)\b",
            re.I,
        ),
        "Filtering disabled people, disabilities, or mobility aids is not permitted.",
    ),
    (
        "gender_identity",
        re.compile(
            r"\b(women|woman|female\s+people|females|men\s+only|male\s+people|males|girls\s+only|boys\s+only|"
            r"transgender\w*|trans\s+people|trans\s+women|trans\s+men|non[-\s]*binary\s+people|enby|genderqueer)\b",
            re.I,
        ),
        "Filtering content based on gender or gender identity is not permitted.",
    ),
    (
        "lgbtq",
        re.compile(
            r"\b(lgbt\w*|queer\s+people|queer\s+symbols?|gay\s+(?:people|men|women|couples?)|lesbian\w*|"
            r"bisexual\w*|pansexual\w*|asexual\s+people|pride\s+flags?|rainbow\s+flags?|trans\s+flags?|"
            r"progress\s+flags?|same[-\s]*sex\s+couples?|drag\s+queens?|f+a+g+g*o*t*s*|d+y+k+e+s*|"
            r"alphabet\s+mafia|rainbow\s+mafia|skittle\s+squad|troon\w*)\b",
            re.I,
        ),
        "Filtering LGBTQ+ people, relationships, or pride symbols is not permitted.",
    ),
    (
        "religion_symbols",
        re.compile(
            r"\b(muslims?|islamic|islam\b|christians?|christianity|jews?|jewish|judaism|hindus?|hinduism|"
            r"buddhists?|buddhism|sikhs?|sikhism|catholics?|protestants?|mormons?|atheists?|"
            r"hijabs?|niqabs?|burqas?|turbans?|kippahs?|yarmulkes?|crucifix\w*|christian\s+cross(?:es)?|"
            r"star\s+of\s+david|crescent\s+and\s+star|menorahs?|mosques?|churches|synagogues?|temples?|religious\s+symbols?)\b",
            re.I,
        ),
        "Filtering religions, believers, religious attire, or religious symbols is not permitted.",
    ),
    (
        "country_government",
        re.compile(
            r"\b(american\s+flags?|us\s+flags?|u\.?s\.?\s+flags?|uk\s+flags?|british\s+flags?|union\s+jack|"
            r"french\s+flags?|german\s+flags?|chinese\s+flags?|russian\s+flags?|israeli\s+flags?|palestinian\s+flags?|"
            r"ukrainian\s+flags?|canadian\s+flags?|mexican\s+flags?|indian\s+flags?|japanese\s+flags?|"
            r"national\s+flags?|country\s+flags?|national\s+symbols?|state\s+emblems?|"
            r"(?:chinese|russian|american|us|uk|british|french|german|israeli|palestinian|iranian|ukrainian|indian|japanese|korean)\s+(?:government|regime|state|citizens?|people|immigrants?|refugees?)|"
            r"immigrants?|refugees?|foreigners?|nationalit(?:y|ies))\b",
            re.I,
        ),
        "Filtering specific countries, nationalities, governments, or national symbols is not permitted.",
    ),
    (
        "political_views",
        re.compile(
            r"\b(conservatives?|liberals?|progressives?|left[-\s]*wing|right[-\s]*wing|far[-\s]*left|far[-\s]*right|"
            r"democrats?|republicans?|labour\s+party|tories|tory|green\s+party|libertarians?|"
            r"socialis[mt]\w*|communis[mt]\w*|capitalis[mt]\w*|anarchis[mt]\w*|feminis[mt]\w*|zionis[mt]\w*|"
            r"anti[-\s]*zionis[mt]\w*|maga|woke|political\s+(?:views?|opinions?|posts?|parties?|ideolog\w*)|"
            r"pro[-\s]*life|pro[-\s]*choice|abortion\s+views?|gun\s+control\s+views?|"
            r"orange\s+man|tds|libtard\w*|conservatard\w*|commie\w*|tankie\w*|fash|antifa)\b",
            re.I,
        ),
        "Filtering specific political views, ideologies, or political parties is not permitted.",
    ),
    (
        "safety_warnings",
        re.compile(
            r"\b(emergency\s+alerts?|amber\s+alerts?|security\s+warnings?|browser\s+warnings?|malware\s+warnings?|"
            r"recall\s+notices?|medical\s+warnings?)\b",
            re.I,
        ),
        "Hiding critical security, medical, or emergency warnings is not permitted.",
    ),
    (
        "specific_person",
        re.compile(
            r"(@[a-z0-9_.]{2,}|"
            r"\b(?:taylor\s+swift|elon\s+musk|donald\s+trump|joe\s+biden|kamala\s+harris|barack\s+obama|"
            r"keanu\s+reeves|beyonce|kim\s+kardashian|mrbeast|pewdiepie|andrew\s+tate|putin|zelensky|"
            r"netanyahu|spider[-\s]*man|batman|superman|iron\s*man|harry\s+potter|naruto|goku|luffy|mickey\s+mouse|"
            r"hatsune\s+miku|pikachu|darth\s+vader|shrek|deadpool|wolverine)\b)",
            re.I,
        ),
        "Filtering any specific real or fictional person/character is not permitted.",
    ),
]


def _normalize_prompt(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    lowered = stripped.lower().translate(_LEET_MAP)
    return re.sub(r"\s+", " ", lowered).strip()


def _deterministic_screen(prompt: str) -> Optional[Dict[str, Any]]:
    raw = (prompt or "").strip()
    if len(raw) < 2:
        return {
            "allowed": False,
            "category": "invalid_prompt",
            "reason": "Please enter a clear description of what you want your classifier to detect.",
            "source": "deterministic",
        }

    # Check explicit allowed contexts (e.g. filtering out "hate speech", "racial slurs", "gore", "nsfw", "child abuse")
    norm = _normalize_prompt(raw)
    is_explicit_abuse_or_safe_category = any(p.search(raw) or p.search(norm) for p in _ALLOWED_SAFETY_PATTERNS)

    for code, pat, msg in _HARD_RULES:
        if is_explicit_abuse_or_safe_category and code in ("race_ethnicity", "age", "disability", "gender_identity", "lgbtq", "religion_symbols"):
            # Allow filtering out *hate speech / slurs / child abuse* themselves, as long as the user isn't asking to filter the group itself
            if re.search(r"\b(hate\s*speech|slurs?|bigotry|nsfw|porn|gore|self[-\s]*harm|child\s+abuse|child\s+exploitation|csam|gambling|ads?)\b", raw, re.I):
                continue
        if pat.search(raw) or pat.search(norm):
            return {
                "allowed": False,
                "category": code,
                "reason": msg,
                "source": "deterministic",
            }
    return None


_NLI_PROHIBITED_SPECS: List[Tuple[str, str]] = [
    ("specific_person", "a specific real or fictional person, movie character, book character, anime character, villain, superhero, celebrity, or politician"),
    ("race_ethnicity", "people of a specific race, ethnicity, ancestry, or skin color"),
    ("age", "people of a specific age or age group, such as children, teenagers, young people, or elderly seniors"),
    ("disability", "disabled people, physical or mental disabilities, wheelchairs, or neurodivergent people"),
    ("gender_identity", "women, men, transgender people, or gender identity groups"),
    ("lgbtq", "LGBTQ+ people, gay or queer people, or pride flags"),
    ("religion_symbols", "a religion, religious believers, religious clothing, or religious symbols"),
    ("country_government", "a specific country, nationality, immigrants, national flag, or government"),
    ("political_views", "political views, political parties, ideologies, or partisan politics"),
    ("safety_warnings", "emergency alerts or browser security warnings"),
]

_NLI_ALLOWED_SPECS: List[Tuple[str, str]] = [
    ("ok", "animals, insects, spiders, ocean, nature, clowns, objects, food, medical needles, or phobias"),
    ("ok", "commercial brands, corporate logos, fast food, or advertisements"),
    ("ok", "NSFW, pornography, nudity, or sexually explicit content"),
    ("ok", "gore, gory or gorey violence, blood, horror, or self-harm"),
    ("ok", "child abuse, child exploitation, grooming, or CSAM discussions and warnings"),
    ("ok", "hate speech, profanity, slurs, or abusive language"),
    ("ok", "clickbait headlines, generic spoilers, or abstract visual styles"),
]


def _llm_screen(
    modality: str,
    prompt: str,
    cache_dir: Path,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run local DeBERTa-v3-base NLI zero-shot gatekeeper and ALWAYS unload from RAM/VRAM afterwards."""
    gatekeeper_dir = cache_dir / "gatekeeper"
    gatekeeper_dir.mkdir(parents=True, exist_ok=True)

    model = None
    tokenizer = None
    torch_mod = None
    inputs = None
    logits = None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch_mod = torch
        if progress_cb:
            progress_cb({
                "phase": "loading_gatekeeper",
                "message": f"Preparing local ethical guard ({GATEKEEPER_MODEL_ID})...",
                "percent": 20,
            })

        tokenizer = AutoTokenizer.from_pretrained(GATEKEEPER_MODEL_ID, cache_dir=str(gatekeeper_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            GATEKEEPER_MODEL_ID,
            cache_dir=str(gatekeeper_dir),
            low_cpu_mem_usage=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        if progress_cb:
            progress_cb({
                "phase": "evaluating_prompt",
                "message": "Evaluating filter topic against ethical policy...",
                "percent": 75,
            })

        entail_idx = next(
            (int(k) for k, v in (model.config.id2label or {}).items() if str(v).lower() == "entailment"),
            0,
        )
        all_specs = _NLI_PROHIBITED_SPECS + _NLI_ALLOWED_SPECS
        premise = f"The user wants a content filter to hide: {prompt.strip()}."
        hyps = [f"This content target refers to {desc}." for _, desc in all_specs]
        inputs = tokenizer(
            [premise] * len(hyps),
            hyps,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            scores = torch.softmax(logits[:, entail_idx], dim=0).cpu().tolist()

        ranked = sorted(zip(scores, all_specs), key=lambda x: x[0], reverse=True)
        top_score, (top_code, _top_desc) = ranked[0]
        prohib_sum = sum(s for s, (c, _) in ranked if c != "ok")

        policy_reasons = {item["code"]: item["detail"] for item in PROHIBITED_POLICY_LIST}
        if top_code != "ok" and (top_score >= 0.32 or prohib_sum >= 0.58):
            reason = policy_reasons.get(top_code, "This filter target violates the ethical policy.")
            return {
                "allowed": False,
                "category": top_code,
                "reason": reason,
                "source": "deberta-v3-zeroshot",
                "score": round(float(top_score), 4),
            }

        return {
            "allowed": True,
            "category": "ok",
            "reason": "Passed deterministic policy and local ethical guard review.",
            "source": "deberta-v3-zeroshot",
            "score": round(float(top_score), 4),
        }
    except Exception as exc:
        # Fail-open ONLY if already passed strict Layer 1 deterministic rules and local model couldn't download offline
        return {
            "allowed": True,
            "category": "ok",
            "reason": f"Passed deterministic ethical rules (local guard unavailable: {type(exc).__name__}).",
            "source": "deterministic_fallback",
            "warning": str(exc),
        }
    finally:
        # Critical user requirement: keep weights on disk, REMOVE immediately from RAM & VRAM
        try:
            del inputs
        except Exception:
            pass
        try:
            del logits
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            if torch_mod is not None and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass


def check_bellows_ethics(
    modality: str,
    prompt: str,
    cache_dir: Path,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run two-stage local ethical inspection on a Bellows filter prompt."""
    clean_mod = "text" if str(modality).lower() == "text" else "image"
    clean_prompt = (prompt or "").strip()

    det = _deterministic_screen(clean_prompt)
    if det is not None:
        det["modality"] = clean_mod
        det["prompt"] = clean_prompt
        return det

    res = _llm_screen(clean_mod, clean_prompt, cache_dir, progress_cb=progress_cb)
    res["modality"] = clean_mod
    res["prompt"] = clean_prompt
    if progress_cb:
        progress_cb({"phase": "done", "message": "Ethical check complete (model unloaded from RAM).", "percent": 100})
    return res
