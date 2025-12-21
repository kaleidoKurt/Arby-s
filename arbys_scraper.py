import os
import requests
import pdfplumber
import json
import asyncio
import re
import uuid
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
RESTAURANT_NAME = "Arby's"
RESTAURANT_ID = "40a157f4-a98b-4de1-aad3-3a6d4bd7e1b5"

# Deterministic ID Namespace (using Restaurant ID)
NAMESPACE_UUID = uuid.UUID(RESTAURANT_ID)

NUTRITION_PDF_URL = "https://assets.ctfassets.net/30q5w5l98nbx/4hzFS4dafgrpYnY0vSr4N5/d91072b55d9c504036428f6d5aa33a02/Arbys_Nutritional_and_Allergen_OCT_2025.pdf"
INGREDIENTS_PDF_URL = "https://assets.ctfassets.net/30q5w5l98nbx/7l0ZsRojpuNsSD34KFmxrR/74a4eeafba50e798f504b54be6867d3e/Arbys_Menu_Items_and_Ingredients_OCT_2025.pdf"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Clients
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Data Models
class Nutrition(BaseModel):
    fat_g: float
    saturated_fat_g: Optional[float] = 0
    trans_fat_g: Optional[float] = 0
    cholesterol_mg: Optional[float] = 0
    sodium_mg: float
    carbohydrates_g: float
    fiber_g: Optional[float] = 0
    sugar_g: float
    protein_g: float

class MenuItem(BaseModel):
    display_name: str
    slug: str
    category: Optional[str]
    calories: int
    nutrition: Nutrition
    ingredients_raw: List[str]
    allergens: List[str]

class MenuExtraction(BaseModel):
    items: List[MenuItem]

class GlossaryMapping(BaseModel):
    glossary: Dict[str, List[str]] = Field(description="Mapping of compound ingredient names to their sub-ingredients")

def download_pdf(url: str, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"File {filename} already exists.")
        return path
    
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path

def extract_text_from_pdf(path: str, pages: Optional[List[int]] = None) -> str:
    print(f"Extracting text from {path}...")
    text = ""
    with pdfplumber.open(path) as pdf:
        target_pages = [pdf.pages[i] for i in pages] if pages else pdf.pages
        for page in target_pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

async def get_glossary_mapping(glossary_text: str) -> Dict[str, List[str]]:
    print("Parsing glossary mapping with Gemini...")
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": GlossaryMapping,
        }
    )
    
    prompt = f"""
    You are an expert food scientist. Analyze the following Arby's ingredient glossary text.
    Extract all compound ingredients (like 'Onion Roll', 'Roast Beef', 'Cheddar Cheese Sauce') and their detailed sub-ingredients.
    
    Rules:
    - The glossary entries usually start with the name of the ingredient followed by its components.
    - Return a flat mapping of 'Ingredient Name' to a list of its sub-ingredients.
    - Flatten any nested lists if found.
    
    GLOSSARY TEXT:
    {glossary_text}
    """
    
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        data = GlossaryMapping.model_validate_json(response.text)
        return data.glossary
    except Exception as e:
        print(f"Error parsing glossary: {e}")
        return {}

async def extract_menu_data(menu_text: str, nutrition_text: str, glossary: Dict[str, List[str]]) -> List[MenuItem]:
    print("Extracting menu data with Gemini...")
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": MenuExtraction,
        }
    )
    
    glossary_json = json.dumps(glossary, indent=2)
    
    prompt = f"""
    You are a nutrition data expert. Extract Arby's menu items using the provided menu text, nutrition text, and ingredient glossary.
    
    CRITICAL RULES:
    1. **Expand Ingredients**: For every ingredient listed for a menu item, check if it exists in the GLOSSARY. If it does, replace it with its sub-ingredients. If it doesn't, keep it as is.
    2. **Flatten**: The final `ingredients_raw` list for each item should be a flat list of individual ingredients.
    3. **Nutrition**: Extract fat, sodium, carbs, sugar, protein, and calories. If a value is '< 1g', use 0.
    4. **Allergens**: Extract explicitly listed allergens.
    5. **Slug**: Generate a URL-friendly slug (e.g., 'classic-beef-n-cheddar').
    
    GLOSSARY:
    {glossary_json}
    
    MENU LAYOUT TEXT:
    {menu_text}
    
    NUTRITION TEXT:
    {nutrition_text}
    """
    
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        data = MenuExtraction.model_validate_json(response.text)
        return data.items
    except Exception as e:
        print(f"Error parsing menu data: {e}")
        return []

def generate_deterministic_id(slug: str) -> str:
    return str(uuid.uuid5(NAMESPACE_UUID, f"{RESTAURANT_ID}:{slug}"))

# --- Category Normalization (V3.5 Standard) ---
ALLOWED_KINDS = {
    "entree", "side", "drink", "dessert", "breakfast", "condiment", "kids", "other"
}

# Order matters: these are “hard gates” for Swap safety (esp. drinks).
_RX_KIDS = re.compile(r"\bkids?\b", re.I)
_RX_CONDIMENT = re.compile(r"\b(condiment|sauce|dipping|dip|spread|dressing|syrup)\b", re.I)

_RX_DRINK = re.compile(
    r"\b("
    r"coffee|latte|cappuccino|macchiato|espresso|americano|cortado|flat white|"
    r"cold brew|nitro|frappuccino|tea|matcha|lemonade|refresher|"
    r"smoothie|shake|milk|steamer|energy|punch|soda|coolatta|juice|water"
    r")\b",
    re.I,
)

# Protein/snack SKUs should not get swept into dessert just because of “bar”.
_RX_SIDE_STRONG = re.compile(
    r"\b(protein box|protein boxes|protein & snack bars|snack bars?)\b", re.I
)

_RX_DESSERT = re.compile(
    r"\b(donut|munchkin|cookie|brownie|cake|cupcake|muffin|loaf|pie|frosty|ice cream|dessert|sweet)\b",
    re.I,
)

_RX_BREAKFAST = re.compile(
    r"\b(breakfast|bagel|omelet|pancake|waffle|crepe|biscuit|croissant|danish|egg bites?)\b",
    re.I,
)

_RX_SIDE = re.compile(
    r"\b(side|fries|chips|nuts|hash browns?|snack|salty)\b", re.I
)

def normalize_category_raw(raw: Optional[str]) -> Optional[str]:
    """
    Lightweight cleanup:
    - trim / collapse whitespace
    - drop ' - Regional' suffix
    - drop long disclaimer after ' - ' (keeps left side)
    - de-slugify if it looks like a slug (e.g., fries-sides)
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    # Remove " - Regional" suffix
    s = re.sub(r"\s*-\s*regional\s*$", "", s, flags=re.I).strip()

    # Strip disclaimers after " - " when right side looks like a sentence/paragraph
    if " - " in s:
        left, right = s.split(" - ", 1)
        if "." in right or len(right) > 45:
            s = left.strip()

    # If slug-like (no spaces, contains hyphens/underscores), de-slugify a bit
    if (" " not in s) and re.search(r"[-_]", s):
        slug = s.replace("_", "-").strip("-")
        parts = [p for p in slug.split("-") if p]
        if len(parts) == 2:
            s = f"{parts[0]} & {parts[1]}"
        elif len(parts) >= 2 and parts[-1].lower() == "more":
            s = " ".join(parts[:-1]) + " & More"
        else:
            s = " ".join(parts)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.title()

    # Collapse whitespace (final)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def infer_category_kind(
    *,
    category: Optional[str],
    display_name: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    """
    Returns one of ALLOWED_KINDS.
    If category is missing and we can’t infer safely, returns 'other' (not 'entree').
    """
    cat = (category or "").strip()
    name = (display_name or "").strip()

    # Optional brand-specific overrides (keys should be normalized category strings)
    if overrides and cat:
        forced = overrides.get(cat)
        if forced in ALLOWED_KINDS:
            return forced

    haystack = f"{cat} {name}".lower()

    if _RX_KIDS.search(haystack):
        return "kids"
    if _RX_CONDIMENT.search(haystack):
        return "condiment"
    if _RX_DRINK.search(haystack):
        return "drink"
    if _RX_SIDE_STRONG.search(haystack):
        return "side"
    if _RX_DESSERT.search(haystack):
        return "dessert"
    if _RX_BREAKFAST.search(haystack):
        return "breakfast"
    if _RX_SIDE.search(haystack):
        return "side"

    # If scraper gave us a category and it’s not any of the above, treat as entree.
    return "entree" if cat else "other"

def normalize_category_fields(
    *,
    display_name: str,
    raw_category: Optional[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], str]:
    category = normalize_category_raw(raw_category)
    kind = infer_category_kind(category=category or raw_category, display_name=display_name, overrides=overrides)
    return category, kind

# --- Ingestion & Validation Logic (V3.5 Standard) ---
MIN_INGREDIENTS_CHARS = 10

def hallucination_check(display_name: str, source_text: str) -> bool:
    """Check if the display name exists in the source text (case-insensitive)."""
    clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", display_name).lower()
    clean_source = re.sub(r"[^a-zA-Z0-9\s]", "", source_text).lower()
    return clean_name in clean_source

async def get_current_item_count() -> int:
    try:
        res = supabase.table("menu_items").select("id", count="exact").eq("restaurant_id", RESTAURANT_ID).execute()
        return res.count if res.count is not None else 0
    except Exception:
        return 0

async def ingest_item(item: MenuItem, source_text: str):
    category, kind = normalize_category_fields(
        display_name=item.display_name,
        raw_category=item.category
    )
    menu_item_id = generate_deterministic_id(item.slug)
    print(f"Processing {item.display_name} (ID: {menu_item_id}, Kind: {kind})...")
    
    # 1. Hallucination Gate
    if not hallucination_check(item.display_name, source_text):
        print(f"  -> REJECTED: Hallucination detected - '{item.display_name}' not in source text.")
        return

    # 2. Ingredient Quality Gate
    total_ing_chars = sum(len(ing) for ing in item.ingredients_raw)
    if total_ing_chars < MIN_INGREDIENTS_CHARS:
        print(f"  -> REJECTED: Insufficient ingredients ({total_ing_chars} chars).")
        return

    # 3. Nutrition Gate
    if item.calories <= 0 and "water" not in item.display_name.lower() and "diet" not in item.display_name.lower():
        print(f"  -> REJECTED: Invalid calories ({item.calories})")
        return

    try:
        # Upsert Menu Item
        res = supabase.table("menu_items").upsert({
            "id": menu_item_id,
            "restaurant_id": RESTAURANT_ID,
            "slug": item.slug,
            "display_name": item.display_name,
            "calories": item.calories,
            "category": category,
            "category_kind": kind,
            "nutrition_raw": item.nutrition.model_dump(),
            "allergens_raw": item.allergens,
            "scraped_at": "now()"
        }, on_conflict="restaurant_id,slug").execute()
        
        if not res.data:
            print(f"  -> Failed to upsert menu item: {item.display_name}")
            return
            
        # Update Ingredients
        supabase.table("item_ingredients").delete().eq("menu_item_id", menu_item_id).execute()
        
        ingredients_payload = [
            {"menu_item_id": menu_item_id, "ingredient": ing}
            for ing in item.ingredients_raw
        ]
        if ingredients_payload:
            supabase.table("item_ingredients").insert(ingredients_payload).execute()
            print(f"  -> SUCCESS: {len(ingredients_payload)} ingredients saved.")
            
    except Exception as e:
        print(f"  -> Error ingesting {item.display_name}: {e}")

async def main():
    # 1. Download PDFs
    nutrition_path = download_pdf(NUTRITION_PDF_URL, "nutrition.pdf")
    ingredients_path = download_pdf(INGREDIENTS_PDF_URL, "ingredients.pdf")
    
    # 2. Extract Text
    # Ingredients PDF: Pages 1-2 are menu layout, 3-7 are glossary (0-indexed: 0-1 and 2-6)
    menu_layout_text = extract_text_from_pdf(ingredients_path, pages=[0, 1])
    glossary_text = extract_text_from_pdf(ingredients_path, pages=[2, 3, 4, 5, 6])
    nutrition_text = extract_text_from_pdf(nutrition_path)
    
    # Save raw text for debug
    with open(os.path.join(DEBUG_DIR, "menu_layout.txt"), "w") as f: f.write(menu_layout_text)
    with open(os.path.join(DEBUG_DIR, "glossary.txt"), "w") as f: f.write(glossary_text)
    with open(os.path.join(DEBUG_DIR, "nutrition.txt"), "w") as f: f.write(nutrition_text)
    
    # 3. Get Glossary Mapping
    glossary = await get_glossary_mapping(glossary_text)
    with open(os.path.join(DEBUG_DIR, "glossary_mapping.json"), "w") as f: json.dump(glossary, f, indent=2)
    
    # 4. Extract Final Menu Data
    items = await extract_menu_data(menu_layout_text, nutrition_text, glossary)
    with open(os.path.join(DEBUG_DIR, "extracted_items.json"), "w") as f: 
        f.write(json.dumps([item.model_dump() for item in items], indent=2))
    
    print(f"Extracted {len(items)} items.")
    
    # 5. Ingest into Supabase
    if not items:
        print("No items extracted. Aborting ingestion.")
        return

    # Count Sanity Check (V3.5 Standard)
    current_count = await get_current_item_count()
    new_count = len(items)
    print(f"Current DB items: {current_count}, New extracted items: {new_count}")
    
    if current_count > 0 and new_count < current_count * 0.8:
        print(f"CRITICAL: New count ({new_count}) is < 80% of current count ({current_count}). Possible partial scrape. ABORTING SWAP.")
        return

    full_source_text = menu_layout_text + "\n" + nutrition_text
    for item in items:
        await ingest_item(item, full_source_text)

if __name__ == "__main__":
    asyncio.run(main())

