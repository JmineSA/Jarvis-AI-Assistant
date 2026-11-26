"""Turn the town or suburb in your address into weather coordinates (optional, best effort).

Open-Meteo's free geocoding finds places by name (city or suburb), not street numbers, so only the
place name is sent: never the street or house number. If it finds nothing, nothing changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float


def place_names(address: str) -> list[str]:
    """Candidate place names from an address, most specific first, without house numbers."""
    parts = [p.strip() for p in re.split(r"[,\n]", address) if p.strip()]
    candidates: list[str] = []
    for part in parts:
        if re.search(r"\b(street|st|road|rd|avenue|ave|drive|dr|lane|ln|crescent|cres|close|way|str)\b", part, re.IGNORECASE):
            continue  # a street line: never sent anywhere
        words = [w for w in part.split() if not re.fullmatch(r"\d+[a-z]?", w, re.IGNORECASE)]
        if words:
            candidates.append(" ".join(words))
    return list(dict.fromkeys(c for c in candidates if len(c) >= 3))


def find_place(address: str, transport: httpx.BaseTransport | None = None, timeout: float = 6.0) -> Place | None:
    """The best matching South African place for the address, or None."""
    with httpx.Client(timeout=timeout, transport=transport) as client:
        for name in place_names(address):
            try:
                response = client.get(URL, params={"name": name, "count": 5, "language": "en", "format": "json"})
                response.raise_for_status()
                results = response.json().get("results") or []
            except (httpx.HTTPError, ValueError) as exc:
                logger.info("Geocoding %r failed: %s", name, exc)
                raise
            for item in results:
                if item.get("country_code") == "ZA":
                    area = item.get("admin1") or ""
                    label = f"{item['name']}, {area}" if area else str(item["name"])
                    return Place(label, float(item["latitude"]), float(item["longitude"]))
    return None
