"""Download the public-domain sample documents.

These are works of the United States Government, reproduced verbatim from the
Electronic Code of Federal Regulations. They carry no copyright (17 U.S.C. 105)
and may be redistributed freely, which is why the repository keeps them rather
than only a download script. This script exists to regenerate them and to record
exactly where they came from.

    python scripts/fetch_public_corpus.py

Needs a network connection. Everything else in the project runs offline.
"""

from __future__ import annotations

import html
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

ECFR_RENDER = (
    "https://www.ecfr.gov/api/renderer/v1/content/enhanced/{date}/title-{title}"
)

# Each entry: output filename, banner, and the eCFR query that returns it.
SOURCES = [
    {
        "filename": "nfip_flood_insurance_policy.md",
        "title": "Standard Flood Insurance Policy - Dwelling Form",
        "query": {"title": "44", "chapter": "I", "subchapter": "B", "part": "61"},
        "extract": ("Appendix A(1) to Part 61", "Appendix A(2) to Part 61"),
        "banner": (
            "> PUBLIC DOMAIN. Standard Flood Insurance Policy, Dwelling Form,\n"
            "> reproduced verbatim from 44 CFR Part 61, Appendix A(1). A work of the\n"
            "> United States Government, no copyright (17 U.S.C. 105).\n"
        ),
        "intro": (
            "Policy issued by the Federal Emergency Management Agency under the\n"
            "National Flood Insurance Program.\n"
        ),
    },
    {
        "filename": "nfip_general_property_form.md",
        "title": "Standard Flood Insurance Policy - General Property Form",
        "query": {"title": "44", "chapter": "I", "subchapter": "B", "part": "61"},
        "extract": ("Appendix A(2) to Part 61", "Appendix A(3) to Part 61"),
        "banner": (
            "> PUBLIC DOMAIN. Standard Flood Insurance Policy, General Property Form,\n"
            "> reproduced verbatim from 44 CFR Part 61, Appendix A(2). A work of the\n"
            "> United States Government, no copyright (17 U.S.C. 105).\n"
        ),
        "intro": (
            "Policy issued by the Federal Emergency Management Agency for other\n"
            "residential and non-residential buildings under the National Flood\n"
            "Insurance Program. Its clauses closely parallel the Dwelling Form.\n"
        ),
    },
    {
        "filename": "nfip_condominium_form.md",
        "title": "Standard Flood Insurance Policy - Residential Condominium Building Association Policy",
        "query": {"title": "44", "chapter": "I", "subchapter": "B", "part": "61"},
        "extract": ("Appendix A(3) to Part 61", None),
        "banner": (
            "> PUBLIC DOMAIN. Residential Condominium Building Association Policy,\n"
            "> reproduced verbatim from 44 CFR Part 61, Appendix A(3). A work of the\n"
            "> United States Government, no copyright (17 U.S.C. 105).\n"
        ),
        "intro": (
            "Policy issued by the Federal Emergency Management Agency to condominium\n"
            "associations under the National Flood Insurance Program. Its claims and\n"
            "conditions clauses closely parallel the other two flood forms.\n"
        ),
    },
    {
        "filename": "ftc_cooling_off_rule.md",
        "title": (
            "FTC Cooling-Off Rule: Cancelling Sales Made at Home or at Certain "
            "Other Locations"
        ),
        "query": {"title": "16", "chapter": "I", "subchapter": "D", "part": "429"},
        "extract": None,
        "banner": (
            "> PUBLIC DOMAIN. Reproduced verbatim from 16 CFR Part 429, the Federal\n"
            "> Trade Commission's Cooling-Off Rule. A work of the United States\n"
            "> Government, no copyright (17 U.S.C. 105).\n"
        ),
        "intro": "",
    },
    {
        "filename": "ftc_mail_order_rule.md",
        "title": "FTC Mail, Internet, or Telephone Order Merchandise Rule",
        "query": {"title": "16", "chapter": "I", "subchapter": "D", "part": "435"},
        "extract": None,
        "banner": (
            "> PUBLIC DOMAIN. Reproduced verbatim from 16 CFR Part 435, the Federal\n"
            "> Trade Commission's Mail Order Rule. A work of the United States\n"
            "> Government, no copyright (17 U.S.C. 105).\n"
        ),
        "intro": (
            "What a seller must do about shipment dates and delays when taking mail,\n"
            "internet or telephone orders.\n"
        ),
    },
    {
        "filename": "osha_injury_recordkeeping.md",
        "title": "OSHA Recording and Reporting Occupational Injuries and Illnesses",
        "query": {"title": "29", "chapter": "XVII", "part": "1904"},
        "extract": None,
        "banner": (
            "> PUBLIC DOMAIN. Reproduced verbatim from 29 CFR Part 1904, the\n"
            "> Occupational Safety and Health Administration's injury and illness\n"
            "> recordkeeping rule. A work of the United States Government, no\n"
            "> copyright (17 U.S.C. 105).\n"
        ),
        "intro": (
            "What an employer must record and report about workplace injuries and\n"
            "illnesses, and the deadlines that apply.\n"
        ),
    },
]

# A wider corpus, one whole CFR part per file. These are consumer, household and
# workplace rules: the kind of federal regulation a person actually runs into
# (credit cards, mortgages, leave, flood cover, door-to-door sales, workplace
# injuries). Each tuple is (filename stem, title text, CFR citation, query).
# The banner and intro are templated in expand_parts().
WIDE_PARTS = [
    # --- 12 CFR, Consumer Financial Protection Bureau -------------------
    ("reg_z_truth_in_lending", "Regulation Z, Truth in Lending",
     "12 CFR Part 1026", {"title": "12", "chapter": "X", "part": "1026"}),
    ("reg_e_electronic_fund_transfers", "Regulation E, Electronic Fund Transfers",
     "12 CFR Part 1005", {"title": "12", "chapter": "X", "part": "1005"}),
    ("reg_b_equal_credit_opportunity", "Regulation B, Equal Credit Opportunity",
     "12 CFR Part 1002", {"title": "12", "chapter": "X", "part": "1002"}),
    ("reg_m_consumer_leasing", "Regulation M, Consumer Leasing",
     "12 CFR Part 1013", {"title": "12", "chapter": "X", "part": "1013"}),
    ("reg_dd_truth_in_savings", "Regulation DD, Truth in Savings",
     "12 CFR Part 1030", {"title": "12", "chapter": "X", "part": "1030"}),
    ("reg_x_real_estate_settlement", "Regulation X, Real Estate Settlement Procedures",
     "12 CFR Part 1024", {"title": "12", "chapter": "X", "part": "1024"}),
    ("fair_debt_collection_practices", "Regulation F, Fair Debt Collection Practices",
     "12 CFR Part 1006", {"title": "12", "chapter": "X", "part": "1006"}),
    ("fair_credit_reporting", "Regulation V, Fair Credit Reporting",
     "12 CFR Part 1022", {"title": "12", "chapter": "X", "part": "1022"}),
    # --- 16 CFR, Federal Trade Commission ------------------------------
    ("ftc_used_car_rule", "FTC Used Motor Vehicle Trade Regulation Rule",
     "16 CFR Part 455", {"title": "16", "chapter": "I", "subchapter": "D", "part": "455"}),
    ("ftc_credit_practices_rule", "FTC Credit Practices Rule",
     "16 CFR Part 444", {"title": "16", "chapter": "I", "subchapter": "D", "part": "444"}),
    ("ftc_holder_in_due_course", "FTC Preservation of Consumers' Claims and Defenses",
     "16 CFR Part 433", {"title": "16", "chapter": "I", "subchapter": "D", "part": "433"}),
    ("ftc_negative_option_rule", "FTC Rule Concerning Recurring Subscriptions and Negative Option Programs",
     "16 CFR Part 425", {"title": "16", "chapter": "I", "subchapter": "D", "part": "425"}),
    ("ftc_funeral_rule", "FTC Funeral Industry Practices",
     "16 CFR Part 453", {"title": "16", "chapter": "I", "subchapter": "D", "part": "453"}),
    ("ftc_care_labeling_rule", "FTC Care Labeling of Textile Wearing Apparel",
     "16 CFR Part 423", {"title": "16", "chapter": "I", "subchapter": "D", "part": "423"}),
    ("ftc_free_credit_reports", "FTC Free Annual File Disclosures",
     "16 CFR Part 610", {"title": "16", "chapter": "I", "subchapter": "F", "part": "610"}),
    ("ftc_identity_theft_red_flags", "FTC Identity Theft Rules",
     "16 CFR Part 681", {"title": "16", "chapter": "I", "subchapter": "F", "part": "681"}),
    # --- 29 CFR, Department of Labor ---------------------------------
    ("fmla_family_medical_leave", "The Family and Medical Leave Act of 1993",
     "29 CFR Part 825", {"title": "29", "chapter": "V", "subchapter": "C", "part": "825"}),
    ("flsa_overtime_compensation", "Overtime Compensation under the Fair Labor Standards Act",
     "29 CFR Part 778", {"title": "29", "chapter": "V", "subchapter": "B", "part": "778"}),
    ("flsa_white_collar_exemptions", "Executive, Administrative, Professional and Outside Sales Exemptions",
     "29 CFR Part 541", {"title": "29", "chapter": "V", "subchapter": "A", "part": "541"}),
    ("flsa_wage_deductions", "Wage Payments and Deductions under the Fair Labor Standards Act",
     "29 CFR Part 531", {"title": "29", "chapter": "V", "subchapter": "A", "part": "531"}),
    ("osha_inspections_citations", "OSHA Inspections, Citations and Proposed Penalties",
     "29 CFR Part 1903", {"title": "29", "chapter": "XVII", "part": "1903"}),
    # --- 44 CFR, National Flood Insurance Program ----------------------
    ("nfip_general_provisions", "NFIP General Provisions",
     "44 CFR Part 59", {"title": "44", "chapter": "I", "subchapter": "B", "part": "59"}),
    ("nfip_criteria_for_land_management", "NFIP Criteria for Land Management and Use",
     "44 CFR Part 60", {"title": "44", "chapter": "I", "subchapter": "B", "part": "60"}),
    ("nfip_eligible_communities", "NFIP Insurance and Eligible Communities",
     "44 CFR Part 62", {"title": "44", "chapter": "I", "subchapter": "B", "part": "62"}),
    ("nfip_appeals_map_changes", "NFIP Identification and Mapping of Special Hazard Areas",
     "44 CFR Part 65", {"title": "44", "chapter": "I", "subchapter": "B", "part": "65"}),
    ("nfip_rating", "NFIP Flood Insurance Rate Maps and Rating",
     "44 CFR Part 61", {"title": "44", "chapter": "I", "subchapter": "B", "part": "61"},
     ("PART 61", "Appendix A(1) to Part 61")),
]

EDITION_DATE = "2024-01-01"


def expand_parts() -> list[dict]:
    """Turn the compact WIDE_PARTS tuples into full source dicts."""
    expanded: list[dict] = []
    for entry in WIDE_PARTS:
        stem, title, citation, query = entry[:4]
        extract = entry[4] if len(entry) > 4 else None
        expanded.append({
            "filename": f"{stem}.md",
            "title": title,
            "query": query,
            "extract": extract,
            "banner": (
                f"> PUBLIC DOMAIN. Reproduced verbatim from {citation}. A work of\n"
                "> the United States Government, no copyright (17 U.S.C. 105).\n"
            ),
            "intro": "",
        })
    return expanded


def fetch(query: dict) -> str:
    """Return the rendered HTML for one eCFR part."""
    params = "&".join(f"{k}={v}" for k, v in query.items() if k != "title")
    url = ECFR_RENDER.format(date=EDITION_DATE, title=query["title"]) + "?" + params
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def fix_mojibake(text: str) -> str:
    """Repair the U+FFFD replacement character the eCFR renderer emits.

    It stands for the section sign when a section number follows (Part 1904 §
    1904.41), and for a curly apostrophe or quote otherwise (employer's).
    """
    text = re.sub(r"�(?=\s*\d)", "§ ", text)
    text = re.sub(r"�(?=\d)", "§", text)
    return text.replace("�", "'")


def to_markdown(raw: str) -> str:
    """Strip HTML to text, keeping headings as markdown."""
    text = re.sub(r"<h([1-6])[^>]*>", lambda m: "\n\n" + "#" * int(m.group(1)) + " ", raw)
    text = re.sub(r"</h[1-6]>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = fix_mojibake(html.unescape(text))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def promote_headings(text: str) -> str:
    """Turn regulation section markers into markdown headings."""
    text = re.sub(r"#{2,6}\s*\"?\s*§\s*(\d+\.\d+)\s+([^\n]+)", r"## \1 \2", text)
    text = re.sub(r"#{2,6}\s*§\s*(\d+\.\d+)\s+([^\n]+)", r"## \1 \2", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        roman = re.match(r"^([IVX]{1,4})\.\s+(.+)$", stripped)
        if roman:
            lines.append(f"\n## {roman.group(1)}. {roman.group(2)}")
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build(source: dict) -> str:
    text = to_markdown(fetch(source["query"]))
    if source["extract"]:
        start, end = source["extract"]
        i = text.find(start)
        if i == -1:
            raise ValueError(f"start marker {start!r} not found")
        j = text.find(end, i + len(start)) if end else len(text)
        if j == -1:
            raise ValueError(f"end marker {end!r} not found")
        text = text[i:j]
    text = promote_headings(text)
    parts = [source["banner"], f"\n# {source['title']}\n"]
    if source["intro"]:
        parts.append("\n" + source["intro"])
    parts.append("\n" + text + "\n")
    return "".join(parts)


def main() -> None:
    config.EVAL_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    wide = "--wide" in sys.argv
    sources = SOURCES + (expand_parts() if wide else [])
    wrote = 0
    for source in sources:
        path = config.EVAL_CORPUS_DIR / source["filename"]
        try:
            content = build(source)
        except Exception as error:  # noqa: BLE001
            print(f"  failed: {source['filename']}: {error}")
            continue
        if len(content) < 800:
            print(f"  skipped {source['filename']}: only {len(content)} chars")
            continue
        path.write_text(content, encoding="utf-8")
        wrote += 1
        print(f"  wrote {source['filename']}  ({len(content):,} chars)")
    print(f"\n{wrote} documents written. Run 'python -m rag.ingest' to index them.")


if __name__ == "__main__":
    main()
