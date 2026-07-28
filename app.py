import io
import json
import os
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageStat

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="SEO & Image Mismatch Detector", layout="wide")


SOURCE_FIELDS = [
    "brand", "model", "product_type", "color", "size", "material", "capacity",
    "dimensions", "weight", "pack_quantity", "gender", "age_group", "compatibility",
    "power", "voltage", "frequency", "feature", "certification", "country_of_origin"
]
SEO_FIELDS = ["seo_title", "bullet_points", "description", "search_terms"]


@dataclass
class Finding:
    sku: str
    area: str
    field: str
    severity: str
    mismatch_type: str
    expected: str
    found: str
    explanation: str
    confidence: float


def normalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9.%/+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combined_seo(row: pd.Series) -> str:
    return " ".join(normalize(row.get(c, "")) for c in SEO_FIELDS)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def token_present(value: str, text: str) -> bool:
    value_n = normalize(value)
    text_n = normalize(text)
    if not value_n:
        return True
    return value_n in text_n


def extract_numbers_with_units(text: str) -> List[str]:
    pattern = r"\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|ml|l|g|kg|w|kw|v|hz|mah|inch|in|\"|%)\b"
    return [normalize(x) for x in re.findall(pattern, normalize(text), flags=re.I)]


def basic_text_checks(row: pd.Series) -> List[Finding]:
    sku = str(row.get("sku", "UNKNOWN"))
    seo = combined_seo(row)
    findings: List[Finding] = []

    critical_fields = ["brand", "model", "product_type", "color", "size", "capacity", "pack_quantity"]
    for field in critical_fields:
        expected = str(row.get(field, "") or "").strip()
        if expected and not token_present(expected, seo):
            findings.append(Finding(
                sku, "SEO", field, "High" if field in {"brand", "model", "product_type"} else "Medium",
                "Missing expected value", expected, "Not found",
                f"The approved {field.replace('_', ' ')} is not present in the SEO content.", 0.95
            ))

    # Title checks
    title = str(row.get("seo_title", "") or "")
    if not title.strip():
        findings.append(Finding(sku, "SEO", "seo_title", "High", "Missing title", "Populated title", "Blank", "SEO title is empty.", 1.0))
    else:
        if len(title) > 200:
            findings.append(Finding(sku, "SEO", "seo_title", "Medium", "Title too long", "<= 200 characters", str(len(title)), "Title may exceed common marketplace limits.", 0.9))
        if re.search(r"\b(best|#1|guaranteed|100% safe|cures?|miracle)\b", title, flags=re.I):
            findings.append(Finding(sku, "SEO", "seo_title", "High", "Potentially unsupported claim", "Factual wording", title, "Title contains a promotional, medical, or absolute claim requiring substantiation.", 0.85))

    # Contradictory numbers/units
    expected_numbers = set()
    for field in ["capacity", "dimensions", "weight", "pack_quantity", "power", "voltage", "frequency"]:
        expected_numbers.update(extract_numbers_with_units(str(row.get(field, "") or "")))
    seo_numbers = set(extract_numbers_with_units(seo))
    unexpected = sorted(seo_numbers - expected_numbers)
    if expected_numbers and unexpected:
        findings.append(Finding(
            sku, "SEO", "numeric specifications", "High", "Unexpected numeric specification",
            ", ".join(sorted(expected_numbers)), ", ".join(unexpected),
            "SEO content contains measurements or specifications not found in the approved source data.", 0.8
        ))

    # Competitor-brand heuristic
    known_brands = ["adidas", "apple", "bosch", "dyson", "hp", "ikea", "lg", "loreal", "nike", "philips", "samsung", "sony"]
    approved_brand = normalize(row.get("brand", ""))
    mentioned = [b for b in known_brands if b in seo and b != approved_brand]
    if mentioned:
        findings.append(Finding(
            sku, "SEO", "brand", "High", "Possible competitor brand",
            str(row.get("brand", "")), ", ".join(mentioned),
            "SEO content appears to mention another brand.", 0.9
        ))

    # Duplicate/keyword stuffing
    words = re.findall(r"\b[a-z0-9]+\b", seo)
    if words:
        max_ratio = max(words.count(w) / len(words) for w in set(words))
        if max_ratio > 0.08 and len(words) > 30:
            repeated = max(set(words), key=words.count)
            findings.append(Finding(
                sku, "SEO", "all SEO fields", "Low", "Possible keyword stuffing",
                "Natural language", f"'{repeated}' repeated {words.count(repeated)} times",
                "One term is repeated unusually often across the SEO content.", 0.7
            ))

    return findings


def image_metrics(url: str, timeout: int = 12) -> Dict[str, Any]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "image" not in content_type and not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise ValueError("URL does not appear to be an image")
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    width, height = image.size
    stat = ImageStat.Stat(image.resize((50, 50)))
    mean = stat.mean
    corner_pixels = [image.getpixel((0, 0)), image.getpixel((width - 1, 0)), image.getpixel((0, height - 1)), image.getpixel((width - 1, height - 1))]
    corner_brightness = sum(sum(px) / 3 for px in corner_pixels) / 4
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3) if height else 0,
        "mean_brightness": round(sum(mean) / 3, 1),
        "corner_brightness": round(corner_brightness, 1),
        "format": image.format or "Unknown",
        "bytes": len(response.content),
    }


def image_checks(row: pd.Series, required_width: int, required_height: int, min_size: int) -> List[Finding]:
    sku = str(row.get("sku", "UNKNOWN"))
    findings: List[Finding] = []
    urls = []
    for col in [c for c in row.index if str(c).lower().startswith("image")]:
        value = str(row.get(col, "") or "").strip()
        if value:
            urls.extend([u.strip() for u in re.split(r"[;|,\n]", value) if u.strip().startswith("http")])

    if not urls:
        findings.append(Finding(sku, "Images", "images", "High", "Missing images", "At least one image", "None", "No valid image URL was found.", 1.0))
        return findings

    if len(urls) != len(set(urls)):
        findings.append(Finding(sku, "Images", "images", "Medium", "Duplicate image URLs", "Unique images", "Duplicates found", "The same image URL is repeated.", 1.0))

    for idx, url in enumerate(urls[:10], start=1):
        field = f"image{idx}"
        try:
            metrics = image_metrics(url)
            if metrics["width"] < min_size or metrics["height"] < min_size:
                findings.append(Finding(sku, "Images", field, "High", "Low resolution", f">= {min_size}px on each side", f"{metrics['width']}x{metrics['height']}", "Image may be too small for marketplace use.", 1.0))
            if required_width and required_height and (metrics["width"] != required_width or metrics["height"] != required_height):
                findings.append(Finding(sku, "Images", field, "Medium", "Dimension mismatch", f"{required_width}x{required_height}", f"{metrics['width']}x{metrics['height']}", "Image dimensions do not match the selected marketplace requirement.", 1.0))
            if metrics["corner_brightness"] < 235:
                findings.append(Finding(sku, "Images", field, "Low", "Possible non-white background", "White/approved background", f"Corner brightness {metrics['corner_brightness']}", "The image corners are not close to white; visually confirm the background requirement.", 0.65))
        except Exception as exc:
            findings.append(Finding(sku, "Images", field, "High", "Broken or inaccessible image", "Accessible image URL", url, f"Image could not be downloaded or opened: {exc}", 1.0))
    return findings


def ai_review(row: pd.Series, model: str) -> List[Finding]:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    client = OpenAI()
    sku = str(row.get("sku", "UNKNOWN"))
    source = {field: str(row.get(field, "") or "") for field in SOURCE_FIELDS}
    seo = {field: str(row.get(field, "") or "") for field in SEO_FIELDS}
    image_urls = []
    for col in [c for c in row.index if str(c).lower().startswith("image")]:
        val = str(row.get(col, "") or "").strip()
        if val.startswith("http"):
            image_urls.append(val)

    prompt = f"""
You are a product content QA auditor. Compare the approved source data against SEO content and available product images.
Return JSON only as an array of objects with these keys: area, field, severity, mismatch_type, expected, found, explanation, confidence.
Severity must be High, Medium, or Low. Confidence must be between 0 and 1.
Only report genuine contradictions, missing critical facts, unsupported claims, wrong variants, misleading compatibility, or visible image/content mismatches. Do not invent facts.
Approved source data: {json.dumps(source, ensure_ascii=False)}
SEO content: {json.dumps(seo, ensure_ascii=False)}
"""
    content = [{"type": "input_text", "text": prompt}]
    for url in image_urls[:3]:
        content.append({"type": "input_image", "image_url": url, "detail": "low"})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )
    raw = response.output_text.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I | re.S)
    data = json.loads(raw)
    return [Finding(
        sku=sku,
        area=str(item.get("area", "AI Review")),
        field=str(item.get("field", "unknown")),
        severity=str(item.get("severity", "Medium")),
        mismatch_type=str(item.get("mismatch_type", "AI-detected mismatch")),
        expected=str(item.get("expected", "")),
        found=str(item.get("found", "")),
        explanation=str(item.get("explanation", "")),
        confidence=float(item.get("confidence", 0.7)),
    ) for item in data]


def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def make_template() -> bytes:
    sample = pd.DataFrame([{
        "sku": "SKU-001",
        "brand": "Bosch",
        "model": "Tower Fan 4000",
        "product_type": "Tower Fan",
        "color": "Grey",
        "size": "95 cm",
        "material": "Plastic",
        "capacity": "2170 m3/h",
        "dimensions": "30 x 30 x 95 cm",
        "weight": "5.5 kg",
        "pack_quantity": "1",
        "gender": "",
        "age_group": "Adult",
        "compatibility": "Indoor use",
        "power": "45 W",
        "voltage": "220-240 V",
        "frequency": "50-60 Hz",
        "feature": "Remote control; timer; oscillation",
        "certification": "",
        "country_of_origin": "",
        "seo_title": "Bosch Tower Fan 6000 Black 65 W",
        "bullet_points": "Quiet cooling fan with remote control and timer.",
        "description": "A 108.5 cm tower fan for bedrooms and offices.",
        "search_terms": "tower fan cooling fan",
        "image1": "https://example.com/product-main.jpg",
        "image2": ""
    }])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="Products")
    return buffer.getvalue()


st.title("SEO & Image Mismatch Detector")
st.caption("Rule-based validation with optional AI-assisted text and visual review")

with st.sidebar:
    st.header("Validation settings")
    required_width = st.number_input("Required image width", min_value=0, value=2000, step=100)
    required_height = st.number_input("Required image height", min_value=0, value=2000, step=100)
    min_size = st.number_input("Minimum side length", min_value=100, value=1000, step=100)
    enable_ai = st.checkbox("Enable AI review", value=False)
    model = st.text_input("OpenAI model", value="gpt-5")
    st.download_button("Download input template", make_template(), "seo_image_mismatch_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("Upload an Excel or CSV product file", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        df = load_data(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    st.subheader("Uploaded data")
    st.dataframe(df.head(20), use_container_width=True)

    missing = [c for c in ["sku", "seo_title", "description"] if c not in df.columns]
    if missing:
        st.warning("Recommended columns missing: " + ", ".join(missing))

    if st.button("Run mismatch detection", type="primary"):
        all_findings: List[Finding] = []
        progress = st.progress(0)
        status = st.empty()

        for i, (_, row) in enumerate(df.iterrows()):
            sku = str(row.get("sku", f"Row {i+1}"))
            status.write(f"Checking {sku}...")
            all_findings.extend(basic_text_checks(row))
            all_findings.extend(image_checks(row, int(required_width), int(required_height), int(min_size)))
            if enable_ai:
                try:
                    all_findings.extend(ai_review(row, model))
                except Exception as exc:
                    all_findings.append(Finding(sku, "AI Review", "all", "Low", "AI review failed", "Successful AI review", str(exc), "Rule-based checks still completed.", 1.0))
            progress.progress((i + 1) / max(len(df), 1))

        status.empty()
        result = pd.DataFrame([asdict(f) for f in all_findings])
        if result.empty:
            st.success("No mismatches were detected by the selected checks.")
        else:
            severity_order = pd.CategoricalDtype(["High", "Medium", "Low"], ordered=True)
            result["severity"] = result["severity"].astype(severity_order)
            result = result.sort_values(["severity", "sku", "area"]).reset_index(drop=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total findings", len(result))
            c2.metric("High severity", int((result["severity"] == "High").sum()))
            c3.metric("Affected SKUs", result["sku"].nunique())

            st.subheader("Mismatch report")
            st.dataframe(result, use_container_width=True, hide_index=True)

            csv_data = result.to_csv(index=False).encode("utf-8")
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                result.to_excel(writer, index=False, sheet_name="Mismatch Report")
                df.to_excel(writer, index=False, sheet_name="Source Data")

            c1, c2 = st.columns(2)
            c1.download_button("Download CSV report", csv_data, "mismatch_report.csv", "text/csv")
            c2.download_button("Download Excel report", excel_buffer.getvalue(), "mismatch_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Start with the downloadable template or upload your existing product master file.")

with st.expander("What the starter version checks"):
    st.markdown("""
- Missing or inconsistent brand, model, product type, color, size, capacity, and pack quantity
- Unsupported or risky title claims
- Unexpected numeric specifications and units
- Competitor-brand mentions and possible keyword stuffing
- Missing, duplicate, broken, low-resolution, incorrectly sized, or potentially non-white-background images
- Optional AI review of SEO content and up to three product images per SKU
""")
