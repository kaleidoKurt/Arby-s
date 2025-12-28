import os
import sys
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
RESTAURANT_NAME = "Arby's"
RESTAURANT_ID = "40a157f4-a98b-4de1-aad3-3a6d4bd7e1b5"
MIN_ITEMS_THRESHOLD = 50 # Fail if we don't get at least this many items

# Global Stats
INGEST_STATS = {
    "total_extracted": 0,
    "success": 0,
    "failed": 0,
    "rejected_hallucination": 0,
    "rejected_quality": 0,
    "rejected_nutrition": 0
}

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
    saturated_fat_g: Optional[float]
    trans_fat_g: Optional[float]
    cholesterol_mg: Optional[float]
    sodium_mg: float
    carbohydrates_g: float
    fiber_g: Optional[float]
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

class MenuItemBasic(BaseModel):
    display_name: str
    slug: str
    category: Optional[str]

class MenuDiscovery(BaseModel):
    items: List[MenuItemBasic]

class MenuExtraction(BaseModel):
    items: List[MenuItem]

class GlossaryEntry(BaseModel):
    compound_ingredient: str = Field(description="Name of the compound ingredient, e.g. 'Onion Roll'")
    sub_ingredients: List[str] = Field(description="List of individual sub-ingredients")

class GlossaryMapping(BaseModel):
    entries: List[GlossaryEntry] = Field(description="List of compound ingredients and their sub-ingredients")

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
        GEMINI_MODEL,
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
    - Extract each compound ingredient and its list of sub-ingredients.
    - Flatten any nested lists if found.
    
    GLOSSARY TEXT:
    {glossary_text}
    """
    
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        data = GlossaryMapping.model_validate_json(response.text)
        return {entry.compound_ingredient: entry.sub_ingredients for entry in data.entries}
    except Exception as e:
        print(f"Error parsing glossary: {e}")
        raise

async def discover_menu_items(menu_text: str, nutrition_text: str) -> List[MenuItemBasic]:
    print("Discovering menu items list with Gemini...")
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": MenuDiscovery,
        }
    )
    
    prompt = f"""
    You are a nutrition data expert. Analyze the Arby's menu text and nutrition text.
    Extract a unique list of all menu items found.
    Return ONLY the display_name, a URL-friendly slug, and the category.
    
    MENU LAYOUT TEXT:
    {menu_text}
    
    NUTRITION TEXT:
    {nutrition_text}
    """
    
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        data = MenuDiscovery.model_validate_json(response.text)
        return data.items
    except Exception as e:
        print(f"Error discovering items: {e}")
        raise

async def extract_item_details_batch(items: List[MenuItemBasic], full_source_text: str, glossary: Dict[str, List[str]]) -> List[MenuItem]:
    """Extract details for a small batch of items to stay within token limits."""
    item_names = [item.display_name for item in items]
    print(f"Extracting details for batch: {item_names}...")
    
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": MenuExtraction,
        }
    )
    
    glossary_json = json.dumps(glossary, indent=2)
    
    prompt = f"""
    You are a nutrition data expert. Extract full details for these Arby's menu items: {', '.join(item_names)}.
    
    CRITICAL RULES:
    1. **Expand Ingredients**: Use the GLOSSARY to expand compound ingredients into their sub-ingredients.
    2. **Flatten**: Return a flat list of individual ingredients in `ingredients_raw`.
    3. **Nutrition**: Extract calories, fat, sodium, carbs, sugar, protein. Use 0 for "< 1g".
    4. **Allergens**: Extract explicitly listed allergens.
    5. **Categorization**: Keep the original category if possible.
    6. **Hallucination**: Only extract items actually found in the SOURCE TEXT. If an item is missing, skip it.
    
    GLOSSARY:
    {glossary_json}
    
    SOURCE TEXT CONTEXT (excerpts):
    {full_source_text[:20000]}
    """
    
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        data = MenuExtraction.model_validate_json(response.text)
        return data.items
    except Exception as e:
        print(f"Error extracting batch {item_names}: {e}")
        return []

def generate_deterministic_id(slug: str) -> str:
    return str(uuid.uuid5(NAMESPACE_UUID, f"{RESTAURANT_ID}:{slug}"))

# --- Category Normalization (Standardized) ---
import sys
from pathlib import Path
# Import shared normalization logic from root
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from shared_utils import normalize_category_fields
except ImportError:
    print("⚠️  shared_utils.py not found in root. Using local limited fallback.")
    def normalize_category_fields(display_name, raw_category, overrides=None):
        return raw_category, "other"

# --- Ingestion & Validation Logic ---
MIN_INGREDIENTS_CHARS = 10

def hallucination_check(display_name: str, source_text: str) -> bool:
    clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", display_name).lower()
    clean_source = re.sub(r"[^a-zA-Z0-9\s]", "", source_text).lower()
    
    # Direct match first (handles most cases)
    if clean_name in clean_source:
        return True
        
    # Word set overlap approach (V3.5 flex)
    # Check if at least 70% of words in the display name appear in the source text
    name_words = [w for w in clean_name.split() if len(w) > 1]
    if not name_words:
        return False
        
    source_words = set(clean_source.split())
    matches = [w for w in name_words if w in source_words]
    
    overlap = len(matches) / len(name_words)
    if overlap >= 0.7:
        print(f"  -> Flex match found: {overlap:.1%} overlap for '{display_name}'")
        return True
        
    return False

async def get_current_item_count() -> int:
    try:
        res = supabase.table("menu_items").select("id", count="exact").eq("restaurant_id", RESTAURANT_ID).execute()
        return res.count if res.count is not None else 0
    except Exception:
        return 0

async def ingest_item(item: MenuItem, source_text: str) -> bool:
    category, kind = normalize_category_fields(display_name=item.display_name, raw_category=item.category)
    menu_item_id = generate_deterministic_id(item.slug)
    print(f"Processing {item.display_name} (ID: {menu_item_id}, Kind: {kind})...")
    
    if not hallucination_check(item.display_name, source_text):
        print(f"  -> REJECTED: Hallucination detected.")
        INGEST_STATS["rejected_hallucination"] += 1
        return False

    if sum(len(ing) for ing in item.ingredients_raw) < MIN_INGREDIENTS_CHARS:
        print(f"  -> REJECTED: Insufficient ingredients.")
        INGEST_STATS["rejected_quality"] += 1
        return False

    if item.calories <= 0 and "water" not in item.display_name.lower() and "diet" not in item.display_name.lower():
        print(f"  -> REJECTED: Invalid calories.")
        INGEST_STATS["rejected_nutrition"] += 1
        return False

    try:
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
            INGEST_STATS["failed"] += 1
            return False
            
        supabase.table("item_ingredients").delete().eq("menu_item_id", menu_item_id).execute()
        
        # Deduplicate ingredients to prevent unique constraint violations (SQL Error 23505)
        unique_ingredients = list(dict.fromkeys(item.ingredients_raw))
        ingredients_payload = [{"menu_item_id": menu_item_id, "ingredient": ing} for ing in unique_ingredients]
        
        if ingredients_payload:
            supabase.table("item_ingredients").insert(ingredients_payload).execute()
        
        INGEST_STATS["success"] += 1
        return True
    except Exception as e:
        print(f"  -> Error ingesting {item.display_name}: {e}")
        INGEST_STATS["failed"] += 1
        return False

async def main():
    try:
        nutrition_path = download_pdf(NUTRITION_PDF_URL, "nutrition.pdf")
        ingredients_path = download_pdf(INGREDIENTS_PDF_URL, "ingredients.pdf")
        
        menu_layout_text = extract_text_from_pdf(ingredients_path, pages=[0, 1])
        glossary_text = extract_text_from_pdf(ingredients_path, pages=[2, 3, 4, 5, 6])
        nutrition_text = extract_text_from_pdf(nutrition_path)
        
        glossary = await get_glossary_mapping(glossary_text)
        basic_items = await discover_menu_items(menu_layout_text, nutrition_text)
        print(f"Discovered {len(basic_items)} items. Extracting in batches...")
        
        full_source_text = menu_layout_text + "\n" + nutrition_text
        batch_size = 5
        
        for i in range(0, len(basic_items), batch_size):
            batch = basic_items[i:i + batch_size]
            detailed_items = await extract_item_details_batch(batch, full_source_text, glossary)
            
            # Fallback: If batch fails (e.g. JSON truncation), retry items individually
            if not detailed_items:
                print(f"Batch {i//batch_size + 1} failed. Retrying items individually...")
                for single_item in batch:
                    retry_item = await extract_item_details_batch([single_item], full_source_text, glossary)
                    if retry_item:
                        await ingest_item(retry_item[0], full_source_text)
                    else:
                        print(f"  -> CRITICAL: Failed to extract {single_item.display_name} even on retry.")
                        INGEST_STATS["failed"] += 1
                continue

            for item in detailed_items:
                await ingest_item(item, full_source_text)
            
            if i + batch_size < len(basic_items):
                await asyncio.sleep(1)

        print("\n--- Scrape Summary ---")
        print(json.dumps(INGEST_STATS, indent=2))
        
        if INGEST_STATS["success"] < MIN_ITEMS_THRESHOLD:
            print(f"CRITICAL: Successful ingestions below threshold. FAIL.")
            sys.exit(1)
            
        print(f"\nSUCCESS: Arby's scraper completed with {INGEST_STATS['success']} items saved.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
