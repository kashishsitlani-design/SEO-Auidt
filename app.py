"""
Listing Auditor — Streamlit edition
Cross-field + image consistency scanner for product listing exports.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ============================================================== CONFIG ==============================================================

MIN_RES = 1000                 # minimum recommended pixel dimension per side
MAX_IMAGES_PER_SKU = 4          # images sent to the AI per SKU
IMAGE_MAX_DIM = 768             # resize images to this before sending to the AI
AI_MODEL = "claude-sonnet-5"
DUPLICATE_HASH_DISTANCE = 4      # <= this many differing bits => "same image"

SYSTEM_PROMPT = """You are an e-commerce listing QA auditor. Compare the product's customer-facing content (title, description, bullet points) and any provided product images against the given backend/reference attributes for the same SKU (the approved source of truth). Also cross-check the title, description and bullets against each other.

Identify factual contradictions, unsupported claims, missing mandatory details, wrong variants (color/size/flavor/scent/capacity/pack quantity/gender/region/model), numeric mismatches (dimensions, weight, volume, wattage, voltage, battery, piece count), and image mismatches (wrong product, wrong variant color, wrong pack count visible, text printed on the image conflicting with the copy, blur, poor resolution, watermark, non-white/non-compliant background, cropping, wrong language on packaging).

Normalize units (inches/cm, oz/ml/g, lb/kg, W, V, storage units, apparel sizes) and synonyms before comparing, but preserve meaningful distinctions: water-resistant is not waterproof; leather is not PU/faux leather; wireless is not the same as Bluetooth unless stated; a range (e.g. "5000mAh battery") should not be flagged against a value inside that range.

Do not flag purely stylistic differences (tone, word choice, capitalization) unless they create a factual contradiction or violate a stated reference attribute. If reference attributes are absent for a given fact, judge internal consistency between title/description/bullets/images instead, and lower your confidence accordingly.

Severity guide: "Critical" = safety/legal/identity risk (wrong model, wrong dosage, incompatible voltage, wrong product entirely). "High" = likely to cause rejection/returns/major confusion (wrong color/size, wrong pack count, incompatible device/model). "Medium" = SEO quality or clarity issue (missing feature, inconsistent wording, minor numeric rounding beyond normal tolerance). "Low" = cosmetic/formatting only.

Return ONLY a JSON array (no prose, no markdown fences, no explanation) of at most 8 issues, ordered most severe first. Each element must have exactly these keys, all string values kept under 15 words each:
mismatch_type, severity ("Critical"|"High"|"Medium"|"Low"), affected_field, expected_value, detected_value, evidence, recommended_action, confidence (a number 0-1).
If there are no real issues, return []."""

SEV_COLORS = {
    "critical": "#FF5C4D",
    "high": "#F2A93B",
    "medium": "#6C8CFF",
    "low": "#8B93A1",
}

# ============================================================== LOCAL TEXT CHECKS ==============================================================

UNIT_MAP = [
    (re.compile(r"\b(milliliters?|millilitres?|ml)\b", re.I), "ml"),
    (re.compile(r"\b(liters?|litres?|l)\b", re.I), "l"),
    (re.compile(r"\b(kilograms?|kgs?)\b", re.I), "kg"),
    (re.compile(r"\b(milligrams?|mgs?)\b", re.I), "mg"),
    (re.compile(r"\b(grams?|gr|g)\b", re.I), "g"),
    (re.compile(r"\b(ounces?|oz)\b", re.I), "oz"),
    (re.compile(r"\b(pounds?|lbs?)\b", re.I), "lb"),
    (re.compile(r"\b(count|counts|ct|pcs|pieces?|packs?)\b", re.I), "ct"),
    (re.compile(r"\b(centimeters?|centimetres?|cm)\b", re.I), "cm"),
    (re.compile(r"\b(millimeters?|millimetres?|mm)\b", re.I), "mm"),
    (re.compile(r"\b(inches|inch|in)\b", re.I), "in"),
    (re.compile(r"\b(feet|foot|ft)\b", re.I), "ft"),
    (re.compile(r"\b(watts?|w)\b", re.I), "w"),
    (re.compile(r"\b(volts?|v)\b", re.I), "v"),
    (re.compile(r"\b(gb)\b", re.I), "gb"),
    (re.compile(r"\b(tb)\b", re.I), "tb"),
    (re.compile(r"\b(mah)\b", re.I), "mah"),
]
UNIT_ALT = ("milliliters?|millilitres?|ml|liters?|litres?|kilograms?|kgs?|milligrams?|mgs?|grams?|gr|g|ounces?|oz|"
            "pounds?|lbs?|count|counts|ct|pcs|pieces?|packs?|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
            "inches|inch|in|feet|foot|ft|watts?|w|volts?|v|gb|tb|mah")
QTY_REGEX = re.compile(r"(\d+(?:[.,]\d+)?)\s?-?\s?(" + UNIT_ALT + r")\b", re.I)
GROUPED_QTY_REGEX = re.compile(r"\b(?:set|pack|box|case|bundle|lot)\s+of\s+(\d+)\b", re.I)

VARIANT_GROUPS = {
    "flavor": ["vanilla", "chocolate", "strawberry", "mint", "lavender", "unscented", "citrus", "lemon", "lime",
               "coconut", "caramel", "coffee", "cinnamon", "honey", "apple", "cherry", "grape", "mango", "blueberry",
               "raspberry", "original flavor", "unflavored", "banana", "almond"],
    "color": ["black", "white", "red", "blue", "green", "yellow", "pink", "purple", "gray", "grey", "silver", "gold",
              "navy", "beige", "brown", "orange", "teal", "maroon", "ivory", "charcoal", "clear", "transparent"],
    "material": ["cotton", "polyester", "genuine leather", "faux leather", "vegan leather", "stainless steel",
                 "plastic", "silicone", "wood", "bamboo", "aluminum", "ceramic", "wool", "nylon", "rubber"],
}
VARIANT_ESCAPE_WORDS = ["variety", "assorted", "multi-", "multi ", "multipack", "set of", "combo", "bundle", "mixed",
                         "random color", "random colour", "colors may vary", "colours may vary"]

ODD_CHECKS = [
    (re.compile(r"&(?:amp|nbsp|quot|apos|reg|trade|copy|hellip|mdash|ndash|rsquo|lsquo|rdquo|ldquo|#\d{2,4});", re.I),
     "low", "contains an unescaped HTML entity (e.g. \"&amp;\", \"&nbsp;\")."),
    (re.compile(r"[ÂÃ][\u0080-\u00BF]"), "medium", "contains encoding corruption (mojibake), e.g. \"Ã©\"."),
    (re.compile(r"([!?])\1{1,}|\.{4,}"), "low", "has repeated punctuation (\"!!\", \"??\", or \"....\")."),
    (re.compile(r"[ \t]{2,}"), "low", "has double or extra spaces."),
    (re.compile(r"^\s|\s$"), "low", "has leading or trailing whitespace."),
    (re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,}){2,}\b"), "low", "has a run of ALL-CAPS words."),
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF]"), "low",
     "contains an emoji or pictographic symbol."),
    (re.compile(r"\u00A0"), "low", "contains a non-breaking space character."),
    (re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]"), "medium", "contains a hidden control character."),
]


def normalize_unit(raw):
    for pat, norm in UNIT_MAP:
        if pat.fullmatch(raw) or pat.search(raw):
            return norm
    return raw.lower()


def extract_quantities(text):
    found = []
    for m in QTY_REGEX.finditer(text):
        num = float(m.group(1).replace(",", "."))
        found.append({"num": num, "unit": normalize_unit(m.group(2)), "raw": m.group(0)})
    for m in GROUPED_QTY_REGEX.finditer(text):
        found.append({"num": float(m.group(1)), "unit": "ct", "raw": m.group(0)})
    return found


def extract_variants(text, words):
    low = text.lower()
    hits = []
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", low, re.I):
            hits.append(w)
    return hits


def has_escape_word(text):
    low = text.lower()
    return any(w in low for w in VARIANT_ESCAPE_WORDS)


def find_odd_formatting(field_name, text):
    issues = []
    if not text:
        return issues
    for pat, sev, desc in ODD_CHECKS:
        if pat.search(text):
            issues.append({
                "mismatch_type": "Formatting", "severity": sev, "affected_field": field_name,
                "expected_value": "", "detected_value": "", "evidence": f"{field_name} {desc}",
                "recommended_action": "Clean up formatting", "confidence": 0.7, "source": "local",
            })
    return issues


def local_fallback_issues(fields):
    """fields: dict[str, str] e.g. {'Title':..., 'Description':..., 'Bullet 1':...}"""
    issues = []
    for name, val in fields.items():
        issues.extend(find_odd_formatting(name, val))

    by_unit = {}
    for name, val in fields.items():
        if not val:
            continue
        for q in extract_quantities(val):
            by_unit.setdefault(q["unit"], []).append({"field": name, **q})
    for unit, entries in by_unit.items():
        distinct = {e["num"] for e in entries}
        if len(distinct) > 1:
            base = entries[0]
            other = next(e for e in entries if e["num"] != base["num"])
            issues.append({
                "mismatch_type": "Quantity mismatch", "severity": "high",
                "affected_field": ", ".join(e["field"] for e in entries),
                "expected_value": base["raw"], "detected_value": other["raw"],
                "evidence": "; ".join(f'{e["field"]}: "{e["raw"]}"' for e in entries),
                "recommended_action": "Align the value across fields", "confidence": 0.75, "source": "local",
            })

    for group_name, words in VARIANT_GROUPS.items():
        if any(v and has_escape_word(v) for v in fields.values()):
            continue
        hits_by_field = {}
        for name, val in fields.items():
            if not val:
                continue
            hits = extract_variants(val, words)
            if hits:
                hits_by_field[name] = hits
        names = list(hits_by_field.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = hits_by_field[names[i]], hits_by_field[names[j]]
                if not set(a) & set(b):
                    issues.append({
                        "mismatch_type": f"{group_name.capitalize()} mismatch", "severity": "high",
                        "affected_field": f"{names[i]} vs {names[j]}",
                        "expected_value": ", ".join(a), "detected_value": ", ".join(b),
                        "evidence": f'{names[i]} mentions "{", ".join(a)}"; {names[j]} mentions "{", ".join(b)}"',
                        "recommended_action": "Confirm the correct variant and align", "confidence": 0.7,
                        "source": "local",
                    })
    return issues


# ============================================================== LANGUAGE DETECTION + TRANSLATION ==============================================================

SCRIPT_RANGES = [
    (re.compile(r"[\u3040-\u30FF]"), "ja", "Japanese"), (re.compile(r"[\uAC00-\uD7A3]"), "ko", "Korean"),
    (re.compile(r"[\u4E00-\u9FFF]"), "zh", "Chinese"), (re.compile(r"[\u0600-\u06FF]"), "ar", "Arabic"),
    (re.compile(r"[\u0590-\u05FF]"), "he", "Hebrew"), (re.compile(r"[\u0400-\u04FF]"), "ru", "Russian"),
    (re.compile(r"[\u0900-\u097F]"), "hi", "Hindi"), (re.compile(r"[\u0E00-\u0E7F]"), "th", "Thai"),
    (re.compile(r"[\u0370-\u03FF]"), "el", "Greek"),
]
LANG_STOPWORDS = {
    "en": {"the", "and", "with", "for", "this", "that", "from", "your", "are", "was", "have", "you", "not"},
    "es": {"el", "la", "los", "las", "de", "para", "con", "este", "esta", "una", "uno", "y", "en", "del", "que"},
    "fr": {"le", "la", "les", "des", "pour", "avec", "ce", "cette", "une", "un", "et", "dans", "vous", "pas"},
    "de": {"der", "die", "das", "und", "für", "mit", "ein", "eine", "ist", "sind", "nicht", "auf", "sie", "oder"},
    "it": {"il", "lo", "la", "gli", "le", "per", "con", "questo", "questa", "una", "uno", "e", "in", "non", "che"},
    "pt": {"o", "a", "os", "as", "para", "com", "este", "esta", "uma", "um", "e", "em", "não", "que"},
    "nl": {"de", "het", "een", "en", "voor", "met", "deze", "dit", "niet", "op", "is", "van"},
}
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
              "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "he": "Hebrew",
              "ru": "Russian", "hi": "Hindi", "th": "Thai", "el": "Greek"}


def detect_language(text):
    if not text or len(text.strip()) < 4:
        return "en", "English"
    for pat, code, name in SCRIPT_RANGES:
        if pat.search(text):
            return code, name
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    if not words:
        return "en", "English"
    scores = {code: sum(1 for w in words if w in sw) for code, sw in LANG_STOPWORDS.items()}
    best, best_score = "en", scores.get("en", 0)
    for code, score in scores.items():
        if code != "en" and score > best_score:
            best, best_score = code, score
    if best == "en" or best_score < 2:
        return "en", "English"
    return best, LANG_NAMES.get(best, best)


@st.cache_data(show_spinner=False, ttl=3600)
def translate_text(text, src_code):
    chunks, cur = [], ""
    for w in text.split():
        if len((cur + " " + w).strip()) > 450:
            if cur:
                chunks.append(cur.strip())
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur.strip())
    parts = []
    for chunk in chunks:
        try:
            r = requests.get("https://api.mymemory.translated.net/get",
                              params={"q": chunk, "langpair": f"{src_code}|en"}, timeout=8)
            data = r.json()
            parts.append(data.get("responseData", {}).get("translatedText") or chunk)
        except Exception:
            parts.append(chunk)
    return " ".join(parts)


# ============================================================== IMAGE CHECKS ==============================================================

def fetch_image(url, timeout=9, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    last_err = "link failed to load or timed out"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            try:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                return img, None
            except Exception:
                last_err = "the response wasn't a readable image (wrong URL, or an HTML error page was returned)"
                continue
        except requests.exceptions.SSLError:
            last_err = "SSL certificate error on that host"
            break  # retrying won't help
        except requests.exceptions.Timeout:
            last_err = "timed out"
        except Exception:
            last_err = "link failed to load"
    return None, last_err


def image_hash(img):
    small = img.convert("L").resize((16, 16))
    arr = np.asarray(small, dtype=np.float32)
    avg = arr.mean()
    return "".join("1" if px > avg else "0" for px in arr.flatten())


def hash_distance(a, b):
    if not a or not b or len(a) != len(b):
        return 999
    return sum(1 for x, y in zip(a, b) if x != y)


def sharpness_variance(img):
    small = img.convert("L").resize((200, int(200 * img.height / max(img.width, 1))))
    arr = np.asarray(small, dtype=np.float32)
    # simple Laplacian kernel convolution
    lap = (-4 * arr[1:-1, 1:-1] + arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:])
    return float(lap.var())


def resize_to_base64(img, max_dim):
    w, h = img.size
    if w > h and w > max_dim:
        h, w = int(h * max_dim / w), max_dim
    elif h > max_dim:
        w, h = int(w * max_dim / h), max_dim
    resized = img.resize((max(w, 1), max(h, 1)))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image_local(url):
    img, err = fetch_image(url)
    if img is None:
        return {"url": url, "status": "broken", "note": err, "img": None, "hash": None}
    low_res = img.width < MIN_RES or img.height < MIN_RES
    h = image_hash(img)
    note = f"{img.width}×{img.height}px" + (f" — below the {MIN_RES}×{MIN_RES}px recommended minimum" if low_res else "")
    return {"url": url, "status": "lowres" if low_res else "ok", "note": note, "img": img, "hash": h}


# ============================================================== AI AUDIT ==============================================================

def normalize_severity(s):
    s = (s or "").lower()
    if s.startswith("crit"):
        return "critical"
    if s.startswith("high"):
        return "high"
    if s.startswith("med"):
        return "medium"
    return "low"


def call_claude_audit(record, api_key):
    image_parts, image_notes = [], []
    for i, res in enumerate(record.get("image_results", [])[:MAX_IMAGES_PER_SKU]):
        if not res:
            continue
        if res.get("img") is not None:
            b64 = resize_to_base64(res["img"], IMAGE_MAX_DIM)
            image_parts.append((f"Image {i+1}", b64))
        else:
            image_notes.append(f"Image {i+1} ({res['url']}) didn't load for visual analysis.")

    attrs = record.get("attributes", {})
    attr_lines = "\n".join(f"{k}: {v}" for k, v in attrs.items()) if attrs else "(none provided)"
    bullets = record.get("bullets", [])
    bullets_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets)) if bullets else "(none)"

    text_block = f"""SKU: {record['sku']}

=== Backend / reference attributes (source of truth) ===
{attr_lines}

=== Title ===
{record.get('title') or '(none)'}

=== Description ===
{record.get('description') or '(none)'}

=== Bullet points ===
{bullets_text}

=== Images ===
{(', '.join(p[0] for p in image_parts) + ' attached below.') if image_parts else 'No images could be analyzed visually.'}
{' '.join(image_notes)}"""

    content = [{"type": "text", "text": text_block}]
    for label, b64 in image_parts:
        content.append({"type": "text", "text": label + ":"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": AI_MODEL, "max_tokens": 1000, "system": SYSTEM_PROMPT,
              "messages": [{"role": "user", "content": content}]},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    text_resp = "\n".join(b["text"] for b in data.get("content", []) if b.get("type") == "text").strip()
    cleaned = re.sub(r"^```json|^```|```$", "", text_resp.strip(), flags=re.I).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("AI response was not a list")
    out = []
    for item in parsed:
        out.append({
            "mismatch_type": item.get("mismatch_type", "Issue"),
            "severity": normalize_severity(item.get("severity")),
            "affected_field": item.get("affected_field", ""),
            "expected_value": item.get("expected_value", ""),
            "detected_value": item.get("detected_value", ""),
            "evidence": item.get("evidence", ""),
            "recommended_action": item.get("recommended_action", ""),
            "confidence": item.get("confidence"),
            "source": "ai",
        })
    return out


# ============================================================== COLUMN GUESSING / RECORD BUILDING ==============================================================

ROLE_OPTIONS = ["Ignore", "SKU / ID", "Title", "Description", "Bullet point", "Image link",
                "Reference attribute (source of truth)"]


def guess_role(header):
    h = header.lower()
    if re.search(r"\b(sku|asin|item ?id|product ?id|upc|id)\b", h):
        return "SKU / ID"
    if "title" in h:
        return "Title"
    if re.search(r"\b(desc|description)\b", h):
        return "Description"
    if re.search(r"bullet|feature|key ?point", h):
        return "Bullet point"
    if re.search(r"image|img|photo|picture|thumbnail", h):
        return "Image link"
    if re.search(r"\b(color|colour|size|material|voltage|wattage|weight|dimension|capacity|model|compat|"
                 r"pack ?count|spec|attribute|flavor|scent|gender|origin|country|warranty|certification)\b", h):
        return "Reference attribute (source of truth)"
    return "Ignore"


def split_multi(text):
    if not text:
        return []
    text = str(text)
    parts = re.split(r"\r?\n|\s*\|\s*|\s*;\s*|,(?=\s*http)", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]


def build_records(df, roles):
    sku_cols = [c for c, r in roles.items() if r == "SKU / ID"]
    title_cols = [c for c, r in roles.items() if r == "Title"]
    desc_cols = [c for c, r in roles.items() if r == "Description"]
    bullet_cols = [c for c, r in roles.items() if r == "Bullet point"]
    image_cols = [c for c, r in roles.items() if r == "Image link"]
    attr_cols = [c for c, r in roles.items() if r == "Reference attribute (source of truth)"]

    records = []
    for i, row in df.iterrows():
        def get(c):
            v = row.get(c, "")
            return "" if pd.isna(v) else str(v)

        sku = " / ".join(get(c) for c in sku_cols if get(c)) or f"Row {i+1}"
        title = " ".join(get(c) for c in title_cols).strip()
        description = "\n".join(get(c) for c in desc_cols).strip()
        bullets = []
        for c in bullet_cols:
            val = get(c)
            if not val:
                continue
            bullets.extend(split_multi(val) if len(bullet_cols) == 1 else [val.strip()])
        bullets = [b for b in bullets if b]
        images = []
        for c in image_cols:
            val = get(c)
            if val:
                images.extend(split_multi(val))
        images = [u for u in images if re.match(r"^https?://", u, re.I)]
        attributes = {c: get(c) for c in attr_cols if get(c)}

        if not (title or description or bullets or images or attributes):
            continue
        records.append({"sku": sku, "title": title, "description": description, "bullets": bullets,
                         "images": images, "attributes": attributes})
    return records


def build_fields(record):
    fields = {"Title": record["title"], "Description": record["description"]}
    for i, b in enumerate(record["bullets"]):
        fields[f"Bullet {i+1}"] = b
    return fields


# ============================================================== ANALYSIS PIPELINE ==============================================================

def analyze_all(records, use_ai, api_key, progress_cb):
    total_steps = len(records) * 2 or 1
    step = 0
    analyzed = []

    # phase 1: local checks (images + text), threaded for speed
    def process_images(rec):
        results = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(analyze_image_local, url): idx for idx, url in enumerate(rec["images"])}
            tmp = [None] * len(rec["images"])
            for fut in as_completed(futs):
                tmp[futs[fut]] = fut.result()
            results = tmp
        return results

    for rec in records:
        fields = build_fields(rec)
        image_results = process_images(rec)
        issues = []

        # duplicate images
        for a in range(len(image_results)):
            for b in range(a + 1, len(image_results)):
                ra, rb = image_results[a], image_results[b]
                if ra and rb and ra.get("hash") and rb.get("hash") and hash_distance(ra["hash"], rb["hash"]) <= DUPLICATE_HASH_DISTANCE:
                    issues.append({"mismatch_type": "Duplicate images", "severity": "low",
                                   "affected_field": f"Image {a+1} vs Image {b+1}",
                                   "expected_value": "Distinct images per slot", "detected_value": "Near-identical images",
                                   "evidence": f"Image {a+1} and Image {b+1} look the same",
                                   "recommended_action": "Replace one with a different angle/shot",
                                   "confidence": 0.6, "source": "local"})
        for i, res in enumerate(image_results):
            if not res:
                continue
            if res["status"] == "broken":
                issues.append({"mismatch_type": "Broken image link", "severity": "high",
                               "affected_field": f"Image {i+1}", "expected_value": "Link loads",
                               "detected_value": res["note"], "evidence": res["url"],
                               "recommended_action": "Fix or replace the image URL", "confidence": 0.95, "source": "local"})
            elif res["status"] == "lowres":
                issues.append({"mismatch_type": "Low resolution", "severity": "medium",
                               "affected_field": f"Image {i+1}", "expected_value": f"≥ {MIN_RES}×{MIN_RES}px",
                               "detected_value": res["note"], "evidence": res["url"],
                               "recommended_action": "Upload a higher-resolution image", "confidence": 0.9, "source": "local"})

        step += 1
        progress_cb(step / total_steps, f"Checked images for {rec['sku']}")

        ai_status = "skipped"
        ai_error = None
        if use_ai and api_key:
            try:
                rec_for_ai = {**rec, "image_results": image_results}
                ai_issues = call_claude_audit(rec_for_ai, api_key)
                issues.extend(ai_issues)
                ai_status = "ok"
            except Exception as e:
                issues.extend(local_fallback_issues(fields))
                ai_status = "error"
                ai_error = str(e)
        else:
            issues.extend(local_fallback_issues(fields))
            ai_status = "skipped"

        step += 1
        progress_cb(step / total_steps, f"Analyzed {rec['sku']}")

        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for iss in issues:
            counts[iss["severity"]] = counts.get(iss["severity"], 0) + 1

        analyzed.append({**rec, "fields": fields, "image_results": image_results, "issues": issues,
                          "counts": counts, "ai_status": ai_status, "ai_error": ai_error})
    return analyzed


# ============================================================== STREAMLIT UI ==============================================================

st.set_page_config(page_title="Listing Auditor", layout="wide")

if "step" not in st.session_state:
    st.session_state.step = "upload"
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "filter" not in st.session_state:
    st.session_state.filter = "All"

st.title("🔍 Listing Auditor")
st.caption("AI-powered cross-field & image consistency scanner for product listing exports")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key", type="password",
                             help="Needed only for the AI audit. Get one at console.anthropic.com. "
                                  "Without it, the tool still runs fast local checks (quantities, flavor/color, "
                                  "formatting, broken links, resolution, duplicate images).")
    use_ai = st.checkbox("Run AI content & image audit", value=bool(api_key))
    if st.button("Test AI connection", use_container_width=True):
        if not api_key:
            st.error("Enter an API key above first.")
        else:
            try:
                test_resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": AI_MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "Say OK."}]},
                    timeout=20,
                )
                if test_resp.status_code == 200:
                    st.success("Connected — the AI audit will run for real.")
                else:
                    st.error(f"API error {test_resp.status_code}: {test_resp.text[:300]}")
            except Exception as e:
                st.error(f"Couldn't reach the API: {e}")
    st.number_input("Minimum recommended resolution (px per side)", min_value=200, max_value=4000, value=MIN_RES,
                     key="min_res_input")
    MIN_RES = st.session_state.min_res_input
    if st.button("Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------------- STEP 1: upload ----------------
if st.session_state.step == "upload":
    uploaded = st.file_uploader("Upload your listing export", type=["csv", "xlsx", "xls"])
    st.markdown("""
**What this checks:**
- **AI cross-check** — Claude reads title, description, bullets, your reference attributes, and the images together and reasons about real contradictions.
- **Structured findings** — mismatch type, severity, expected vs. detected value, evidence, recommended action.
- **Local safety net** — broken image links, low-resolution images, and duplicate images are checked directly, no API key needed.
""")
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            else:
                df = pd.read_excel(uploaded, dtype=str)
                df = df.fillna("")
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            df = None

        if df is not None:
            before = len(df)
            df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)
            dropped = before - len(df)
            st.session_state.df = df
            st.session_state.dropped = dropped
            st.session_state.step = "mapping"
            st.rerun()

# ---------------- STEP 2: mapping ----------------
elif st.session_state.step == "mapping":
    df = st.session_state.df
    note = f"{len(df)} row(s) with content detected"
    if st.session_state.get("dropped"):
        note += f" (ignored {st.session_state.dropped} blank row(s))"
    st.info(note)
    st.subheader("Confirm your columns")
    st.caption("Map any backend/spec columns (color, model, voltage, pack count, compatibility, etc.) as "
               "**Reference attribute** — that's the source of truth the audit checks the copy against.")

    roles = {}
    cols = st.columns(2)
    for i, col in enumerate(df.columns):
        guess = guess_role(col)
        sample = str(df[col].iloc[0])[:60] if len(df) else ""
        with cols[i % 2]:
            roles[col] = st.selectbox(f"**{col}**  \n`{sample}`", ROLE_OPTIONS,
                                       index=ROLE_OPTIONS.index(guess), key=f"role_{col}")

    if st.button("Run audit →", type="primary"):
        st.session_state.roles = roles
        st.session_state.use_ai = use_ai
        st.session_state.api_key = api_key
        st.session_state.step = "analyzing"
        st.rerun()

# ---------------- STEP 3: analyzing ----------------
elif st.session_state.step == "analyzing":
    df = st.session_state.df
    records = build_records(df, st.session_state.roles)
    if not records:
        st.warning("No usable rows found after mapping — check your column roles.")
        if st.button("← Back to mapping"):
            st.session_state.step = "mapping"
            st.rerun()
    else:
        progress = st.progress(0.0)
        status = st.empty()

        def cb(frac, label):
            progress.progress(min(frac, 1.0))
            status.caption(label)

        analyzed = analyze_all(records, st.session_state.use_ai, st.session_state.api_key, cb)
        st.session_state.analyzed = analyzed
        st.session_state.step = "results"
        st.rerun()

# ---------------- STEP 4: results ----------------
elif st.session_state.step == "results":
    analyzed = st.session_state.analyzed
    total = len(analyzed)
    crit_n = sum(1 for r in analyzed if r["counts"]["critical"] > 0)
    high_n = sum(1 for r in analyzed if r["counts"]["high"] > 0)
    clean_n = sum(1 for r in analyzed if len(r["issues"]) == 0)
    error_n = sum(1 for r in analyzed if r["ai_status"] == "error")
    skipped_n = sum(1 for r in analyzed if r["ai_status"] == "skipped")

    if error_n > 0:
        sample_err = next((r["ai_error"] for r in analyzed if r["ai_status"] == "error"), "")
        st.error(f"AI audit failed for {error_n} of {total} SKU(s) and fell back to local keyword checks only "
                 f"(which mainly catch quantity/unit mismatches — that's likely why results look thin). "
                 f"First error seen: {sample_err}")
    elif skipped_n == total:
        st.info("AI audit was off for this whole run — every SKU below only has fast local keyword checks "
                 "(mainly quantities/units), not the full AI comparison.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SKUs", total)
    m2.metric("Critical", crit_n)
    m3.metric("High", high_n)
    m4.metric("Clean", clean_n)

    col_side, col_main = st.columns([1, 2.6])

    with col_side:
        search = st.text_input("Search SKU", "")
        flt = st.radio("Show", ["All", "Flagged only"], horizontal=True, key="filter_radio")
        indices = [i for i, r in enumerate(analyzed)
                   if (flt == "All" or len(r["issues"]) > 0) and (search.lower() in str(r["sku"]).lower())]
        if st.session_state.current_idx not in indices and indices:
            st.session_state.current_idx = indices[0]

        st.markdown(f"**{len(indices)} SKU(s)**")
        list_box = st.container(height=520)
        with list_box:
            for i in indices:
                r = analyzed[i]
                worst = ("🔴" if r["counts"]["critical"] else "🟠" if r["counts"]["high"] else
                         "🔵" if len(r["issues"]) else "🟢")
                label = f"{worst} {r['sku']}  ·  {len(r['issues'])} issue(s)"
                if st.button(label, key=f"skubtn_{i}", use_container_width=True,
                             type="primary" if i == st.session_state.current_idx else "secondary"):
                    st.session_state.current_idx = i
                    st.rerun()

    with col_main:
        if not indices:
            st.info("No SKUs match your filter.")
        else:
            idx = st.session_state.current_idx
            rec = analyzed[idx]
            pos = indices.index(idx) if idx in indices else 0

            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("‹ Prev", use_container_width=True):
                    st.session_state.current_idx = indices[(pos - 1) % len(indices)]
                    st.rerun()
            with nav2:
                st.markdown(f"<div style='text-align:center;color:#8B93A1;'>SKU {pos+1} of {len(indices)}</div>",
                             unsafe_allow_html=True)
            with nav3:
                if st.button("Next ›", use_container_width=True):
                    st.session_state.current_idx = indices[(pos + 1) % len(indices)]
                    st.rerun()

            st.markdown(f"### `{rec['sku']}`")

            if rec["ai_status"] == "error":
                st.warning(f"AI audit hit an error for this SKU ({rec['ai_error']}) — showing local checks as a fallback.")
            elif rec["ai_status"] == "skipped":
                st.info("AI audit was off for this run (no API key, or unchecked) — showing fast local checks only.")

            if not rec["issues"]:
                st.success("✓ No mismatches detected for this SKU.")
            else:
                sev_order = ["critical", "high", "medium", "low"]
                for iss in sorted(rec["issues"], key=lambda x: sev_order.index(x["severity"])):
                    color = SEV_COLORS[iss["severity"]]
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                            f"<b style='font-size:15px;'>{iss['mismatch_type']}</b>"
                            f"<span style='background:{color}22;color:{color};padding:2px 10px;border-radius:100px;"
                            f"font-size:11px;font-weight:700;text-transform:uppercase;'>{iss['severity']}</span></div>",
                            unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**Affected field**  \n{iss.get('affected_field') or '—'}")
                        c1.markdown(f"**Expected value**  \n{iss.get('expected_value') or '—'}")
                        c2.markdown(f"**Detected value**  \n{iss.get('detected_value') or '—'}")
                        c2.markdown(f"**Recommended action**  \n{iss.get('recommended_action') or '—'}")
                        if iss.get("evidence"):
                            st.caption(f"Evidence: {iss['evidence']}")
                        conf = iss.get("confidence")
                        src = "AI" if iss.get("source") == "ai" else "local heuristic"
                        if conf is not None:
                            st.caption(f"Confidence {round(conf*100)}% · {src}")

            if rec["attributes"]:
                st.markdown("**Reference attributes**")
                st.table(pd.DataFrame(list(rec["attributes"].items()), columns=["Attribute", "Value"]))

            for name, val in rec["fields"].items():
                if not val:
                    continue
                code, lang_name = detect_language(val)
                label = f"**{name}**" + (f"  `{lang_name} detected`" if code != "en" else "")
                st.markdown(label)
                st.markdown(f"> {val}")
                if code != "en":
                    with st.spinner("Translating…"):
                        translated = translate_text(val, code)
                    st.markdown(f"*English translation:* {translated}")

            if rec["images"]:
                st.markdown(f"**Images ({len(rec['images'])})**")
                img_cols = st.columns(min(4, len(rec["images"])))
                for i, url in enumerate(rec["images"]):
                    res = rec["image_results"][i] if i < len(rec["image_results"]) else None
                    with img_cols[i % len(img_cols)]:
                        if res and res.get("img") is not None:
                            st.image(res["img"], use_container_width=True)
                        else:
                            st.markdown("⚠️ *couldn't load*")
                        if res:
                            icon = "🟢" if res["status"] == "ok" else "🟠" if res["status"] == "lowres" else "🔴"
                            st.caption(f"{icon} {res['note']}")
                        st.caption(url)

    st.divider()
    st.caption(
        "Duplicate images, broken links and low-resolution images are checked directly and always run. "
        "AI-flagged issues are Claude's read of the content and images against your reference attributes — "
        "treat expected-vs-detected as a strong lead, not an automatic edit; verify before publishing a fix."
    )
