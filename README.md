SEO & Image Mismatch Detector
A Streamlit starter tool that compares approved product data with SEO content and image URLs.
Run locally
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Open the local Streamlit URL shown in the terminal.
Optional AI review
Set your API key as an environment variable before launching:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"

# macOS/Linux
export OPENAI_API_KEY="your-key"
```
Then enable AI review in the sidebar. The tool sends the source data, SEO fields, and up to three image URLs per SKU for mismatch analysis.
Expected columns
Minimum recommended columns:
`sku`
`brand`, `model`, `product_type`, `color`, `size`
`capacity`, `dimensions`, `weight`, `pack_quantity`
`power`, `voltage`, `frequency`, `feature`
`seo_title`, `bullet_points`, `description`, `search_terms`
`image1` through `image10`
The app includes a downloadable sample template.
Important limitation
Rule-based checks identify likely errors. Visual claims such as exact product color, included accessories, logo correctness, and wrong product variant need AI review or human verification.
