import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

"""
Web scraping utilities to enrich marketing campaign metadata
from French banking websites (Boursorama, Crédit Agricole).

Design goals:
- Explicit business rules (auditable)
- Robust error handling (no hard crashes)
- Logging instead of silent failures
- Normalized, schema-validated outputs
- Unit-test-friendly pure functions
Legal & Ethical Considerations:
- Only public, non-authenticated pages were accessed
- No personal data collected
- Requests rate is low and non-intrusive
- Scraping used solely for academic and analytical purposes
"""

import logging
import unicodedata
from datetime import datetime
from typing import Dict, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)

# In your main app/notebook you can configure:
# logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------
# Business rules and constants
# -------------------------------------------------------------------

HEADERS: Dict[str, str] = {"User-Agent": "Mozilla/5.0"}

CHANNEL_RULES: Dict[str, Iterable[str]] = {
    "Online": ["en ligne", "internet", "digital"],
    "Branch": ["agence", "conseiller"],
    "Phone": ["appel", "téléphone"],
}

EXPECTED_COLUMNS = {
    "campaign_name",
    "campaign_type",
    "campaign_channel",
    "source_url",
    "scrape_date",
}

BOURSORAMA_URL = "https://www.boursorama.com/banque-en-ligne/produits-bancaires/"
CA_URL = "https://www.credit-agricole.fr/ca-paris/particulier/compte-bancaire.html"


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize French text:
    - lowercasing
    - removing accents (diacritics)

    This makes keyword-based rules more robust.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Remove accents
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def infer_channel_from_text(text: str) -> str:
    """
    Infer campaign channel based on rule-based keyword matching.
    Business rules are explicit and auditable.

    Examples
    --------
    >>> infer_channel_from_text("Ouvrir un compte en ligne")
    'Online'
    """
    normalized = normalize_text(text)

    for channel, keywords in CHANNEL_RULES.items():
        if any(keyword in normalized for keyword in keywords):
            return channel

    # Fallback when no rule matches
    return "Mixed"


def safe_get_html(url: str, timeout: int = 10) -> str:
    """
    Fetch HTML content from a URL with error handling.

    Returns
    -------
    str
        HTML content as string. Returns empty string on failure.
    """
    try:
        logger.info(f"Fetching URL: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"Failed to fetch URL: {url} | Error: {e}", exc_info=True)
        return ""


def validate_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Validate that the DataFrame contains the expected columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        msg = f"Output schema validation failed for {source_name}. Missing: {missing}"
        logger.error(msg)
        raise ValueError(msg)
    return df


def add_scrape_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a scrape_date column with the current UTC timestamp.
    """
    df["scrape_date"] = datetime.utcnow()
    return df


# -------------------------------------------------------------------
# Boursorama scraper
# -------------------------------------------------------------------

def scrape_boursorama_campaign_metadata() -> pd.DataFrame:
    """
    scraper for Boursorama Banque (2025 HTML structure).
    """

    html = safe_get_html(BOURSORAMA_URL)
    if not html:
        logger.warning("Empty HTML for Boursorama. Returning empty DataFrame.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    soup = BeautifulSoup(html, "html.parser")

    # NEW selector (2025)
    product_cards = soup.select("div.c-product-card")

    if not product_cards:
        logger.error("No product cards found on Boursorama page. HTML structure may have changed.")
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    records = []

    for card in product_cards:
        # Extract name
        name_el = card.select_one(".c-product-card__title")
        name = name_el.get_text(strip=True) if name_el else "Unknown product"

        # Extract category
        category_el = card.select_one(".c-product-card__category")
        category = category_el.get_text(strip=True) if category_el else "Unknown category"

        # Extract URL
        link_el = card.select_one("a")
        relative_url = link_el.get("href", "") if link_el else ""
        full_url = f"https://www.boursorama.com{relative_url}"

        # Infer channel
        full_text = card.get_text(" ", strip=True)
        channel = infer_channel_from_text(full_text)

        records.append({
            "campaign_name": name,
            "campaign_type": category,
            "campaign_channel": channel,
            "source_url": full_url,
            "scrape_date": datetime.utcnow()
        })

    df = pd.DataFrame(records)

    # Deduplicate
    df = df.drop_duplicates(subset=["campaign_name"])

    # Schema validation
    df = validate_schema(df, source_name="Boursorama scraper")

    logger.info(f"Boursorama scraper produced {len(df)} rows.")
    return df


# -------------------------------------------------------------------
# Nickel  scraper
# -------------------------------------------------------------------

NICKEL_URL = "https://nickel.eu/fr/offre"


def scrape_nickel_campaign_metadata() -> pd.DataFrame:
    """
    Scrapes public offer pages from Nickel to infer
    marketing campaign channel metadata.

    Returns
    -------
    pd.DataFrame
        Columns:
        - campaign_name
        - campaign_type
        - campaign_channel
        - source_url
        - scrape_date
    """

    html = safe_get_html(NICKEL_URL)
    if not html:
        logger.warning("Empty HTML for Nickel. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    soup = BeautifulSoup(html, "html.parser")

    # Nickel uses offer blocks; we target generic sections/cards
    offer_cards = soup.select("section, article, div.card, div.offre")

    if not offer_cards:
        logger.error("No offer cards found on Nickel page. HTML structure may have changed.")
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    records = []

    for card in offer_cards:
        name_el = (
            card.select_one("h2")
            or card.select_one(".title")
            or card.select_one(".card-title")
        )
        name = name_el.get_text(strip=True) if name_el else "Unknown offer"

        category_el = (
            card.select_one(".subtitle")
            or card.select_one(".category")
            or card.select_one("h3")
        )
        category = category_el.get_text(strip=True) if category_el else "Unknown category"

        link_el = card.select_one("a")
        relative_url = link_el.get("href", "") if link_el else ""
        if relative_url.startswith("http"):
            full_url = relative_url
        else:
            full_url = f"https://nickel.eu{relative_url}"

        full_text = card.get_text(" ", strip=True)
        channel = infer_channel_from_text(full_text)

        records.append(
            {
                "campaign_name": name,
                "campaign_type": category,
                "campaign_channel": channel,
                "source_url": full_url,
                "scrape_date": datetime.utcnow(),
            }
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["campaign_name"])
    df = validate_schema(df, source_name="Nickel scraper")

    logger.info(f"Nickel scraper produced {len(df)} rows.")
    return df