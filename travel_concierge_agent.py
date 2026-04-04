#!/usr/bin/env python3
"""Specialized travel concierge agent for local experiences.

Given a city and a date window, this agent pulls fresh local listings from
Eventbrite and builds a practical schedule focused on:
  - festivals
  - gallery openings
  - restaurant pop-ups

Only events with explicit start times and physical venue locations are
included in the final itinerary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dateutil.parser import isoparse


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
EVENTBRITE_BASE = "https://www.eventbrite.com"

CATEGORY_SEARCH_TERMS: dict[str, list[str]] = {
    "festival": [
        "festival",
        "street festival",
        "food festival",
    ],
    "gallery opening": [
        "gallery opening",
        "art opening",
        "exhibition opening",
    ],
    "restaurant pop-up": [
        "restaurant pop up",
        "restaurant popup",
        "chef pop up dinner",
        "food pop up",
    ],
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "festival": (
        "festival",
        "fest",
        "carnival",
        "fair",
    ),
    "gallery opening": (
        "gallery opening",
        "art opening",
        "opening reception",
        "private view",
        "vernissage",
        "exhibition opening",
    ),
    "restaurant pop-up": (
        "pop up",
        "popup",
        "chef dinner",
        "guest chef",
        "restaurant takeover",
        "supper club",
    ),
}


@dataclasses.dataclass(frozen=True)
class Event:
    title: str
    category: str
    start: datetime
    end: datetime | None
    venue: str
    address: str
    source_url: str
    description: str

    @property
    def day_key(self) -> str:
        return self.start.date().isoformat()


def _slugify_city(city: str) -> str:
    lowered = city.strip().lower()
    lowered = re.sub(r"[^\w\s-]", "", lowered)
    lowered = re.sub(r"\s+", "-", lowered)
    return lowered


def _normalize_eventbrite_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    if "eventbrite." not in parsed.netloc:
        return None
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return normalized.rstrip("/")


def _safe_json_load(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _flatten_json_ld(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        if "@graph" in payload and isinstance(payload["@graph"], list):
            for item in payload["@graph"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _build_address(address: dict[str, Any]) -> str:
    if not isinstance(address, dict):
        return ""
    parts = [
        address.get("streetAddress", ""),
        address.get("addressLocality", ""),
        address.get("addressRegion", ""),
        address.get("postalCode", ""),
        address.get("addressCountry", ""),
    ]
    return ", ".join(part for part in parts if part)


def _looks_online(location_name: str, attendance_mode: str) -> bool:
    lower_location = location_name.lower()
    lower_mode = attendance_mode.lower()
    return (
        "online" in lower_location
        or "virtual" in lower_location
        or "onlineeventattendancemode" in lower_mode
    )


def _parse_event_details(html: str, source_url: str, fallback_category: str) -> Event | None:
    soup = BeautifulSoup(html, "html.parser")

    event_obj: dict[str, Any] | None = None
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        payload = _safe_json_load(script.get_text())
        if payload is None:
            continue
        for obj in _flatten_json_ld(payload):
            obj_type = str(obj.get("@type", "")).lower()
            if obj_type in {"webpage", "breadcrumblist"}:
                continue
            if "startDate" in obj and "location" in obj:
                event_obj = obj
                break
        if event_obj:
            break

    if not event_obj:
        return None

    start_raw = event_obj.get("startDate")
    if not isinstance(start_raw, str) or "T" not in start_raw:
        # Require explicit times, not date-only listings.
        return None

    try:
        start_dt = isoparse(start_raw)
    except (TypeError, ValueError):
        return None

    end_dt: datetime | None = None
    end_raw = event_obj.get("endDate")
    if isinstance(end_raw, str):
        try:
            end_dt = isoparse(end_raw)
        except (TypeError, ValueError):
            end_dt = None

    location_obj = event_obj.get("location", {})
    location_name = ""
    address = ""
    if isinstance(location_obj, dict):
        location_name = str(location_obj.get("name", "")).strip()
        address = _build_address(location_obj.get("address", {}))

    attendance_mode = str(event_obj.get("eventAttendanceMode", ""))
    if _looks_online(location_name, attendance_mode):
        return None
    if not location_name or not address:
        return None

    title = str(event_obj.get("name", "")).strip()
    description = str(event_obj.get("description", "")).strip()
    category = _classify_category(
        title=title,
        description=description,
        fallback_category=fallback_category,
    )

    if category is None:
        return None

    return Event(
        title=title,
        category=category,
        start=start_dt,
        end=end_dt,
        venue=location_name,
        address=address,
        source_url=source_url,
        description=re.sub(r"\s+", " ", description),
    )


def _classify_category(title: str, description: str, fallback_category: str) -> str | None:
    haystack = f"{title} {description}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category

    # Keep a strict fallback only when a query path was category-driven.
    if fallback_category in CATEGORY_SEARCH_TERMS:
        return fallback_category
    return None


def _in_date_range(dt: datetime, start_date: date, end_date: date) -> bool:
    return start_date <= dt.date() <= end_date


class TravelConciergeAgent:
    """Fetches local listings and builds a scheduled itinerary."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def build_schedule(
        self,
        city: str,
        start_date: date,
        end_date: date,
        *,
        max_events_per_category: int = 5,
        max_total_events: int = 12,
        max_workers: int = 8,
    ) -> list[Event]:
        city_slug = _slugify_city(city)
        candidates: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        for category, terms in CATEGORY_SEARCH_TERMS.items():
            for term in terms:
                urls = self._search_event_urls(city_slug, term, start_date, end_date)
                for url in urls:
                    normalized = _normalize_eventbrite_url(url)
                    if not normalized or normalized in seen_urls:
                        continue
                    seen_urls.add(normalized)
                    candidates.append((normalized, category))

        events: list[Event] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._fetch_event, url, category): (url, category)
                for url, category in candidates
            }
            for future in as_completed(futures):
                event = future.result()
                if event is None:
                    continue
                if not _in_date_range(event.start, start_date, end_date):
                    continue
                events.append(event)

        events.sort(key=lambda e: e.start)
        return self._select_balanced_schedule(
            events,
            max_events_per_category=max_events_per_category,
            max_total_events=max_total_events,
        )

    def _search_event_urls(
        self,
        city_slug: str,
        query: str,
        start_date: date,
        end_date: date,
    ) -> list[str]:
        url = f"{EVENTBRITE_BASE}/d/{city_slug}/all-events/"
        params = {
            "q": query,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        response = self.session.get(url, params=params, timeout=30, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        ld_script = soup.find("script", {"type": "application/ld+json"})
        if not ld_script:
            return []

        payload = _safe_json_load(ld_script.get_text())
        if not isinstance(payload, dict):
            return []

        urls: list[str] = []
        for item in payload.get("itemListElement", []):
            if not isinstance(item, dict):
                continue
            event_obj = item.get("item", {})
            if not isinstance(event_obj, dict):
                continue
            event_url = event_obj.get("url")
            if isinstance(event_url, str):
                urls.append(event_url)
        return urls

    def _fetch_event(self, url: str, category: str) -> Event | None:
        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException:
            return None
        return _parse_event_details(response.text, source_url=url, fallback_category=category)

    @staticmethod
    def _select_balanced_schedule(
        events: list[Event],
        *,
        max_events_per_category: int,
        max_total_events: int,
    ) -> list[Event]:
        selected: list[Event] = []
        category_counts: defaultdict[str, int] = defaultdict(int)
        used_slots: set[tuple[str, str]] = set()

        for event in events:
            if len(selected) >= max_total_events:
                break
            if category_counts[event.category] >= max_events_per_category:
                continue

            slot_key = (event.day_key, event.start.strftime("%H:%M"))
            if slot_key in used_slots:
                continue

            selected.append(event)
            category_counts[event.category] += 1
            used_slots.add(slot_key)

        return selected


def render_schedule(city: str, start_date: date, end_date: date, events: list[Event]) -> str:
    lines: list[str] = []
    lines.append(
        f"Travel Concierge Schedule for {city} "
        f"({start_date.isoformat()} to {end_date.isoformat()})"
    )
    lines.append("=" * 80)

    if not events:
        lines.append("No qualifying events found with explicit start times and venue addresses.")
        lines.append(
            "Try widening the date range or switching to a larger nearby city."
        )
        return "\n".join(lines)

    grouped: defaultdict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.day_key].append(event)

    for day in sorted(grouped.keys()):
        lines.append("")
        lines.append(day)
        lines.append("-" * len(day))
        day_events = sorted(grouped[day], key=lambda e: e.start)
        for event in day_events:
            local_time = event.start.strftime("%H:%M")
            lines.append(f"{local_time} | {event.title} [{event.category}]")
            lines.append(f"  Venue: {event.venue}")
            lines.append(f"  Address: {event.address}")
            lines.append(f"  Source: {event.source_url}")
    return "\n".join(lines)


def _json_serializer(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a travel itinerary of festivals, gallery openings, and "
            "restaurant pop-ups for a city and date range."
        )
    )
    parser.add_argument("--city", required=True, help='City name, e.g. "Berlin"')
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-total-events",
        type=int,
        default=12,
        help="Maximum number of events in the output schedule.",
    )
    parser.add_argument(
        "--max-events-per-category",
        type=int,
        default=5,
        help="Maximum number of events per category.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of formatted text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    except ValueError as exc:
        print(f"Invalid date format: {exc}", file=sys.stderr)
        return 2

    if end_date < start_date:
        print("end-date must be on or after start-date", file=sys.stderr)
        return 2

    agent = TravelConciergeAgent()
    events = agent.build_schedule(
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        max_events_per_category=args.max_events_per_category,
        max_total_events=args.max_total_events,
    )

    if args.json:
        payload = {
            "city": args.city,
            "start_date": start_date,
            "end_date": end_date,
            "events": events,
        }
        print(json.dumps(payload, default=_json_serializer, indent=2))
    else:
        print(render_schedule(args.city, start_date, end_date, events))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
