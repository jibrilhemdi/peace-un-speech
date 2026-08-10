"""Extract individual speaker turns from UN verbatim meeting records (PDF) into a CSV.

Each PDF in ``data/raw/pdfs/`` is a full verbatim record (S/PV.* or A/*/PV.*) containing many
speeches. Many PDFs are byte-identical duplicates, because the collector downloaded the
same meeting record once per speaker, so records are de-duplicated by content hash first.

Turn detection relies on the typography of UN verbatim records rather than on regular
expressions: the speaker's name is always set in bold at the start of the paragraph that
begins their intervention, e.g.

    **Mr. Nebenzia** (Russian Federation) (*spoke in Russian*): We thank ...

Usage:
    python scripts/collect/extract_speech_turns.py [--pdf-dir DIR] [--out CSV]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

# Defaults are anchored on the repo root so the script works from any working directory.
DATA = Path(__file__).resolve().parents[2] / "data"

# Running heads/feet are set ~8.6pt; body text is ~10.1pt.
# Body text is set at 10.1pt (one stray block renders at 12.0pt). Running heads/feet are
# 8.6pt, the "Accessible document" flag is 9.0pt and cover mastheads reach 20.3pt.
BODY_MIN_SIZE = 9.5
BODY_MAX_SIZE = 12.1
BOLD_FLAG = 1 << 4
ITALIC_FLAG = 1 << 1

# Printing artefacts that survive the size filter: job numbers, barcodes, masthead flags.
JUNK_BLOCK = re.compile(
    r"^(?:\d{2}-\d{5}\s*(?:\(E\))?\s*(?:\*\d+\*)?|\*\d+\*|Accessible ?document|"
    r"Please ?recycle|Official Records|Provisional|Agenda)\s*$",
    re.I,
)
# On a cover page everything above the President/roster line is masthead, not speech.
COVER_MASTHEAD_Y = 200
# Fraction of page height treated as top/bottom margin when hunting for running heads.
MARGIN_BAND = 0.12
MIN_HEAD_REPEATS = 2

# Long-form country names as they appear in the parenthetical, mapped to common short form.
COUNTRY_ALIASES = {
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United States of America": "United States",
    "Russian Federation": "Russian Federation",
    "Republic of Korea": "Republic of Korea",
    "Democratic People's Republic of Korea": "Democratic People's Republic of Korea",
    "Bolivarian Republic of Venezuela": "Venezuela",
    "Plurinational State of Bolivia": "Bolivia",
    "Kingdom of the Netherlands": "Netherlands",
    "Islamic Republic of Iran": "Iran",
    "Syrian Arab Republic": "Syrian Arab Republic",
    "United Republic of Tanzania": "United Republic of Tanzania",
    "Lao People's Democratic Republic": "Lao People's Democratic Republic",
    "State of Palestine": "State of Palestine",
    "Observer State of Palestine": "State of Palestine",
}

# Parenthetical qualifiers that are not a country/entity.
NON_COUNTRY_PAREN = re.compile(
    r"^\s*(spoke|continued|interpretation|in\s+(English|French|Arabic)|resumed)\b",
    re.I,
)

# Parentheticals naming a UN entity rather than a Member State.
UN_BODY = re.compile(
    r"\b(Department|Office|Secretariat|Division|Programme|Agency|Commission|Committee|"
    r"Fund|Organization|United Nations)\b",
    re.I,
)

TITLES = (
    r"Mr|Mrs|Ms|Miss|Mme|M|Dr|Sir|Dame|Lord|Lady|Baroness|Prince|Princess|Sheikh|Sheikha|"
    r"Archbishop|Bishop|Cardinal|Monsignor|Rabbi|Judge|Justice|President|Chief|Hon|"
    r"Excellency|(?:Major |Lieutenant |Brigadier |Rear |Vice )?(?:General|Admiral)|"
    r"Colonel|Captain|Commander"
)

PRESIDING = re.compile(r"^The\s+(Acting\s+|Temporary\s+|Vice-)?(President|Chair(man|person)?)\b", re.I)


# --------------------------------------------------------------------------------------
# PDF -> ordered paragraphs
# --------------------------------------------------------------------------------------
def margin_signature(bbox, text: str, height: float) -> str | None:
    """Signature for a block sitting in the top/bottom margin, with digits masked.

    Running heads repeat on every page with only the date and page number changing, so
    masking digits makes them collapse to one recurring signature.
    """
    y_mid = (bbox[1] + bbox[3]) / 2
    if MARGIN_BAND < y_mid / height < 1 - MARGIN_BAND:
        return None
    return re.sub(r"\d+", "#", text)[:120]


def running_head_signatures(doc) -> set[str]:
    """Signatures of blocks that recur in the margins across pages of this document."""
    counts: collections.Counter = collections.Counter()
    for page in doc:
        height = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            text = "".join(s["text"] for line in b["lines"] for s in line["spans"]).strip()
            if not text:
                continue
            sig = margin_signature(b["bbox"], text, height)
            if sig:
                counts[sig] += 1
    return {sig for sig, n in counts.items() if n >= MIN_HEAD_REPEATS}


def page_paragraphs(page, is_cover: bool = False, drop_sigs: set[str] | None = None) -> list[dict]:
    """Return body paragraphs of a page in reading order.

    UN records alternate between one- and two-column layouts across years, so blocks are
    bucketed into columns by their horizontal midpoint before being sorted top-to-bottom.
    """
    blocks = []
    height = page.rect.height
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        # Drop running heads/feet, mastheads and barcodes by font size.
        if not BODY_MIN_SIZE <= max(s["size"] for s in spans) <= BODY_MAX_SIZE:
            continue
        if is_cover and b["bbox"][1] < COVER_MASTHEAD_Y:
            continue
        raw = "".join(s["text"] for s in spans).strip()
        if JUNK_BLOCK.match(raw):
            continue
        # Some records set the running head at body size, so fall back to position:
        # a margin block whose text recurs across pages is a head/foot, not speech.
        if drop_sigs and margin_signature(b["bbox"], raw, height) in drop_sigs:
            continue
        blocks.append((b["bbox"], b["lines"]))

    if not blocks:
        return []

    mid = page.rect.width / 2
    two_col = any(bb[2] < mid for bb, _ in blocks) and any(bb[0] > mid for bb, _ in blocks)
    if two_col:
        blocks.sort(key=lambda kv: (0 if kv[0][0] < mid else 1, round(kv[0][1], 1)))
    else:
        blocks.sort(key=lambda kv: round(kv[0][1], 1))

    paras = [build_paragraph(lines) for _, lines in blocks]
    return [p for p in paras if p["text"].strip()]


def normalize(text: str) -> str:
    """Collapse whitespace, drop soft hyphens and tidy spacing inside parentheses."""
    # A soft hyphen marks a discretionary break: drop it and any whitespace it introduced,
    # so "Amer­ ica" rejoins as "America".
    text = re.sub(r"­\s*", "", text).replace("​", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def build_paragraph(lines) -> dict:
    """Join a block's lines into one paragraph, de-hyphenating across line breaks."""
    text = ""
    bold_prefix = ""
    prefix_open = True  # still accumulating the leading run of bold spans
    n_bold = n_italic = n_total = 0

    for line in lines:
        # Spans within a line are contiguous and already carry their own spacing, so they
        # are concatenated as-is; only the join between lines may need a space inserted.
        line_text = ""
        for s in (s for s in line["spans"] if s["text"]):
            piece = s["text"]
            is_bold = bool(s["flags"] & BOLD_FLAG) or "Bold" in s["font"]
            is_italic = bool(s["flags"] & ITALIC_FLAG) or "Italic" in s["font"]
            if piece.strip():
                n_total += 1
                n_bold += is_bold
                n_italic += is_italic
            if prefix_open and is_bold:
                bold_prefix += piece
            elif piece.strip():
                prefix_open = False
            line_text += piece

        if not line_text:
            continue
        if text.endswith("-"):
            # A hyphen at a line break is never a word boundary: "justi-|fied" rejoins as
            # "justified", while "two-|State" keeps its hyphen. Neither takes a space.
            text = (text[:-1] if line_text[:1].islower() else text) + line_text.lstrip()
        elif text and not text.endswith(" ") and not line_text.startswith(" "):
            text += " " + line_text
        else:
            text += line_text

    return {
        "text": normalize(text),
        "bold_prefix": bold_prefix.strip(),
        "all_bold": n_total > 0 and n_bold == n_total,
        "all_italic": n_total > 0 and n_italic == n_total,
    }


def merge_continuations(paras: list[dict]) -> list[dict]:
    """Re-join paragraphs split across a column or page break."""
    out: list[dict] = []
    for p in paras:
        if (
            out
            and not p["bold_prefix"]
            and out[-1]["text"]
            and not re.search(r'[.!?:;”"’\)]\s*$', out[-1]["text"])
            and p["text"][:1].islower()
        ):
            prev = out[-1]
            if prev["text"].endswith("-") and p["text"][:1].islower():
                prev["text"] = prev["text"][:-1] + p["text"]
            else:
                prev["text"] = prev["text"] + " " + p["text"]
        else:
            out.append(dict(p))
    return out


# --------------------------------------------------------------------------------------
# Cover page metadata
# --------------------------------------------------------------------------------------
FILENAME_SYMBOL = re.compile(
    r"^((?:A|S|E)_[A-Za-z0-9._\-]*PV\.\d+)(?:_Resumption(\d+))?$"
)


def symbol_from_filename(path: Path) -> str:
    """Recover the UN document symbol from a filename written by fetch_meeting_records.py.

    That collector encodes the symbol with '/' replaced by '_' ("A_ES-10_PV.43.pdf"), so the
    mapping back is exact. Returns "" for filenames that do not encode a symbol, which is how
    PDFs named by catalogue record id are left to the cover-page parser.
    """
    m = FILENAME_SYMBOL.match(path.stem)
    if not m:
        return ""
    symbol = m.group(1).replace("_", "/")
    return f"{symbol} (Resumption {m.group(2)})" if m.group(2) else symbol


def parse_cover(page) -> dict:
    """Pull doc symbol, meeting date and the President/members roster off the cover page."""
    text = page.get_text()
    flat = re.sub(r"[.…]{2,}", " ", text)  # dot leaders
    info: dict = {"president_country": "", "roster": {}}

    m = re.search(r"\b((?:A|S|E)/[A-Za-z0-9./\-]*PV\.\s?\d+(?:\s*\(Resumption\s*\d+\))?)", text)
    info["doc_symbol"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    m = re.search(
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*"
        r"(\d{1,2}\s+\w+\s+\d{4})",
        text,
    )
    info["meeting_date"] = m.group(2) if m else ""

    m = re.search(r"(\d+)(?:st|nd|rd|th)\s+(?:plenary\s+)?meeting", text, re.I)
    info["meeting_number"] = m.group(1) if m else ""

    # President: Mrs. Thomas-Greenfield . . . . . . . (United States of America)
    m = re.search(r"President:\s*(.+?)\((.+?)\)", flat, re.S)
    if m:
        info["president_name"] = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
        info["president_country"] = re.sub(r"\s+", " ", m.group(2)).strip()
    else:
        info["president_name"] = ""

    # Members roster. Country and name are separated by dot leaders and may share a line or
    # sit on consecutive lines depending on the year's cover layout:
    #     Albania . . . . . . . . . .
    #     Mr. Hoxha
    title_re = re.compile(r"^(?:%s)\b\.?\s+\S" % TITLES)
    chunk = text.split("Members:", 1)
    if len(chunk) > 1:
        raw_lines = chunk[1].split("Agenda")[0].split("\n")
        pending_country = ""
        for ln in raw_lines:
            ln = re.sub(r"\s+", " ", ln).strip()
            if not ln:
                continue
            head, sep, tail = re.sub(r"(?:\s*[.\u2026]){3,}", "\x00", ln).partition("\x00")
            head, tail = head.strip(" .,"), tail.strip(" .,")
            if sep:  # this line carried dot leaders, so `head` is a country
                if tail and title_re.match(tail):
                    _add_roster(info["roster"], tail, head)
                    pending_country = ""
                else:
                    pending_country = head
            elif pending_country and title_re.match(ln):
                _add_roster(info["roster"], ln, pending_country)
                pending_country = ""

    if info.get("president_name"):
        _add_roster(info["roster"], info["president_name"], info["president_country"])
    return info


def _add_roster(roster: dict, name_field: str, country: str) -> None:
    """Record surname -> country, splitting alternates such as "Mr. Waltz/Ms. Wu"."""
    country = country.strip(" .,")
    if not country:
        return
    for person in name_field.split("/"):
        surname = surname_of(person)
        if surname:
            roster.setdefault(surname, country)


def surname_of(person: str) -> str:
    person = re.sub(r"\s+", " ", person).strip(" .,")
    person = re.sub(r"^(?:%s)\.?\s+" % TITLES, "", person).strip()
    return person


# --------------------------------------------------------------------------------------
# Procedural vs substantive text
# --------------------------------------------------------------------------------------
# Formulaic chair housekeeping: giving the floor, invitations under the rules of procedure,
# vote mechanics, speaking-time reminders. Matched per paragraph, so that a President turn
# that opens with housekeeping and then delivers a national statement keeps its substance.
PROCEDURAL_PARA = re.compile(
    r"""^(?:
    I\s(?:now\s|shall\snow\s|shall\sfirst\s|will\snow\s|should\snow\s|then\s|once\sagain\s|
        again\s|warmly\s)*(?:give|call\son|invite|welcome)\b
  | I\s(?:now\s)?(?:give|call)\sthe\sfloor\b
  | I\sshall\s(?:now|first)\s(?:give|call)\b
  | I\stake\snote\sof\sthe\sstatement\b
  | I\sam\sgrateful\b[^.]{0,80}\binterpreters\b
  | The\s(?:presidency|President)\sof\sthe\sSecurity\sCouncil\b[^.]{0,80}\bfloor\b
  | We\sare\spresiding\sover\sthe\sCouncil\b
  | The\s(?:Security\sCouncil|General\sAssembly)\swill\snow\s
        (?:begin|proceed|take|resume|consider|hear|vote)\b
  | The\sGeneral\sAssembly\shas\sthus\sconcluded\b
  | In\saccordance\swith\s(?:rule|the\sunderstanding|Article)\b
  | (?:Under\s)?Rule\s\d+\sof\sthe\s(?:Council|Assembly)
  | (?:There\sbeing|In\sthe\sabsence\sof)\sno\sobjection\b
  | If\sI\shear\sno\sobjection\b
  | It\sis\sso\sdecided\b
  | I\spropose\sthat\s(?:the\s(?:Council|Assembly)\sinvite|we\smove\son)\b
  | I\s(?:now\s)?resume\smy\sfunctions\sas\s(?:the\s)?President\b
  | The\s(?:representative|Permanent\s(?:Observer|Representative)|Minister|Observer)\b
        [^.]{0,140}?\b(?:asked\sfor\sthe\sfloor|has\ssubmitted)\b
  | There\s(?:are|is|appear\sto\sbe)\s(?:no\smore|still|\d+)\b
        [^.]{0,60}\b(?:names\sinscribed|speakers)\b
  | The\snext\sspeaker\b
  | Members\sof\sthe\s(?:Council|Assembly)\shave\sbefore\sthem\b
  | (?:It\sis\smy\sunderstanding\sthat\s)?The?\s?(?:Council|Assembly)\sis\sready\sto\sproceed\b
  | It\sis\smy\sunderstanding\sthat\sthe\s(?:Council|Assembly)\sis\sready\b
  | I\sshall\s(?:now\s|first\s)?put\b
  | Accordingly,\sI\sintend\sto\sput\b
  | I\sintend,\swith\sthe\sconcurrence\b
  | In\sview\sof\sthe\srequest\smade\b[^.]{0,120}\bput\b
  | (?:The\sprovisional\sagenda|The\s(?:draft\s)?resolution|The\s(?:proposed\s)?
        (?:oral\s)?amendment)\s(?:received|was\s(?:adopted|not\sadopted)|
        has\s(?:been\s)?(?:adopted|not\sbeen\sadopted))\b
  | I\s(?:would\slike\sto|wish\sto|should\slike\sto)\s
        (?:remind|inform|request|encourage|draw\sthe\sattention|take\sthis\sopportunity)\b
  | (?:May\sI|Before\s(?:giving\sthe\sfloor|continuing|we\s(?:begin|proceed)|I\sadjourn))\b
  | May\sI\stake\sit\sthat\b
  | On\sbehalf\sof\sthe\s(?:Council|Assembly),\sI\s
        (?:wish|should\slike|thank|congratulate|welcome)\b
  | We\shave\sheard\sthe\s(?:last|final)\sspeaker\b
  | We\sshall\s(?:hear|now\sproceed|now\sturn|now\sconsider)\b
  | I\s(?:shall|will)\snow\smake\sa\sstatement\sin\smy\scapacity\b
  | The\smeeting\s(?:is|stands)\s(?:adjourned|suspended)\b
  | I\snow\sinvite\b
  | The\sPermanent\sObserver\sof\sthe\sObserver\sState\b
  | I\sreiterate\sto\sall\sparticipating\sdelegations\b
  | In\sorder\sto\sensure\sthat\sall\sspeakers\b
  | Allow\sme\sto\sexpress\smy\ssincere\sthanks\sto\sthe\sinterpreters\b
  | Did\syou\sfinish\syour\sstatement\b
)""",
    re.X | re.I,
)
# Courtesy thanks for a briefing, e.g. "I thank Mr. Wennesland for his briefing."
THANKS = re.compile(r"^I\s(?:also\s|now\s|again\s)?(?:thank|wish\sto\sthank|should\slike\sto\sthank|"
                    r"express\smy\s\w+\s(?:appreciation|thanks))\b", re.I)
THANKS_OBJECT = re.compile(
    r"\b(briefing|statement|presentation|remarks|clarification|participation|being\shere|"
    r"Secretary-General|his|her|their)\b",
    re.I,
)


def is_procedural_paragraph(paragraph: str, prev_procedural: bool = False) -> bool:
    """True for chair housekeeping that carries no substantive position.

    ``prev_procedural`` lets a quoted passage inherit its introduction, so the text of a
    rule of procedure read out by the chair is not mistaken for a substantive statement.
    """
    p = paragraph.strip()
    if PROCEDURAL_PARA.match(p):
        return True
    if prev_procedural and p.startswith(("“", '"')):
        return True
    # Thanks are procedural only when that is the whole paragraph, not an opening courtesy
    # followed by an actual statement.
    return bool(THANKS.match(p) and len(p.split()) <= 60 and THANKS_OBJECT.search(p))


def substantive_words(speech: str) -> int:
    total = 0
    prev = False
    for p in speech.split("\n\n"):
        prev = is_procedural_paragraph(p, prev)
        if not prev:
            total += len(p.split())
    return total


# --------------------------------------------------------------------------------------
# Speaker header parsing
# --------------------------------------------------------------------------------------
def split_header(para: dict) -> tuple[str, str] | None:
    """Split a turn-opening paragraph into (speaker header, speech text)."""
    text = para["text"]
    prefix = para["bold_prefix"]
    if not prefix:
        return None

    idx = text.find(":")
    if idx != -1 and idx <= 220:
        header, body = text[:idx].strip(), text[idx + 1 :].strip()
    else:
        # A few records drop the colon after the speaker's name, e.g.
        # "Mrs. Shino (Japan) I thank ...". Split after the trailing parentheticals
        # instead; the person check below still has to pass.
        m = re.match(re.escape(prefix.strip()) + r"\s*((?:\([^()]*\)\s*)+)", text)
        if not m:
            return None  # bold section heading, not a speaker
        header, body = text[: m.end()].strip(), text[m.end() :].strip()
    # The colon must come after the bold name, not inside it.
    if len(header) < len(prefix.rstrip(" :")) - 2:
        return None
    # Guard against bold headings that happen to contain a colon: the header has to look
    # like a person ("Mr. Nebenzia") or a presiding officer ("The President").
    stripped = re.sub(r"\([^()]*\)", "", header).strip()
    if not (
        PRESIDING.match(stripped)
        or re.match(r"^The Secretary-General\b", stripped, re.I)
        or re.match(r"^(?:%s)\b\.?\s+\S" % TITLES, stripped)
    ):
        return None
    return header, body


def parse_speaker(header: str, cover: dict) -> dict:
    parens = re.findall(r"\(([^()]*)\)", header)
    name = re.sub(r"\([^()]*\)", "", header)
    name = re.sub(r"\s+", " ", name).strip(" ,;:")

    language = ""
    country = ""
    notes = []
    for p in parens:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if re.match(r"^spoke\b", p, re.I) or NON_COUNTRY_PAREN.match(p):
            lm = re.search(r"spoke in ([A-Za-z]+)", p, re.I)
            if lm:
                language = lm.group(1)
            notes.append(p)
        elif not country and not UN_BODY.search(p):
            country = p
        else:
            notes.append(p)

    is_presiding = bool(PRESIDING.match(name))
    if is_presiding and not country:
        country = cover.get("president_country", "")
    if not country:
        country = cover.get("roster", {}).get(surname_of(name), "")

    # Trim role/office suffixes: "Jordan. Deputy Prime Minister" -> "Jordan"
    country = re.sub(r"\s*\.\s*(Minister|Deputy|President|Prime|Secretary|Ambassador).*$", "", country)
    country = country.strip(" .,")
    country_norm = COUNTRY_ALIASES.get(country, country)

    if is_presiding:
        role = "president"
    elif country:
        role = "representative"
    else:
        role = "official/briefer"

    # The record says only "The President"; the cover page names the person holding the
    # chair, so report that and keep the generic label in `speaker_header`.
    if re.fullmatch(r"The President", name, re.I) and cover.get("president_name"):
        name = cover["president_name"]

    return {
        "name": name,
        "country": country_norm,
        "country_raw": country,
        "language": language or ("English" if role != "official/briefer" else ""),
        "role": role,
        "notes": "; ".join(notes),
    }


# --------------------------------------------------------------------------------------
# Document -> turns
# --------------------------------------------------------------------------------------
# Stage directions are set entirely in italics ("The meeting rose at 1.35 p.m."). Only the
# ones below imply that the previous speaker's turn is definitively over.
TURN_ENDING = re.compile(
    r"^(The meeting (was|rose)|A vote was taken|The (General Assembly|Security Council) "
    r"(then )?(adjourned|resumed))",
    re.I,
)
# Italic label whose following roman paragraph is a list of countries, not speech.
VOTE_LABEL = re.compile(r"^(In favour|Against|Abstaining)\s*:", re.I)
CALLED_TO_ORDER = re.compile(r"^The meeting was called to order", re.I)
# Bold section headings that are not fully bold (the "(continued)" suffix is roman).
HEADING = re.compile(r"^(Agenda item|Adoption of the agenda|Expression of|Tribute to)\b", re.I)


def extract_document(path: Path) -> tuple[dict, list[dict]]:
    doc = fitz.open(path)
    cover = parse_cover(doc[0])
    heads = running_head_signatures(doc)

    # Security Council covers are masthead-only, but General Assembly records start the
    # verbatim text on the cover page itself, so keep whatever follows "called to order".
    paras: list[dict] = []
    first_page = page_paragraphs(doc[0], is_cover=True, drop_sigs=heads)
    for i, p in enumerate(first_page):
        if CALLED_TO_ORDER.match(p["text"]):
            paras.extend(first_page[i:])
            break
    for pno in range(1, doc.page_count):
        paras.extend(page_paragraphs(doc[pno], drop_sigs=heads))
    doc.close()
    paras = merge_continuations(paras)

    turns: list[dict] = []
    current: dict | None = None
    skip_next_plain = False

    for p in paras:
        text = p["text"]

        if p["all_italic"] and len(text) < 300:
            if TURN_ENDING.match(text):
                current = None
            skip_next_plain = bool(VOTE_LABEL.match(text))
            continue

        if p["bold_prefix"]:
            split = split_header(p)
            if split:
                header, body = split
                info = parse_speaker(header, cover)
                current = {**info, "header": header, "parts": [body] if body else []}
                turns.append(current)
                skip_next_plain = False
                continue
            if p["all_bold"] or HEADING.match(text):
                current = None  # section heading such as "Agenda item 35 (continued)"
                skip_next_plain = False
                continue

        if skip_next_plain:
            skip_next_plain = False
            continue
        if current is not None:
            current["parts"].append(text)

    for t in turns:
        t["speech"] = "\n\n".join(x for x in t["parts"] if x).strip()
        del t["parts"]
    return cover, [t for t in turns if t["speech"]]


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(DATA / "raw" / "pdfs"))
    ap.add_argument("--out", default=str(DATA / "interim" / "un_speeches_extracted.csv"))
    ap.add_argument("--metadata", default=str(DATA / "raw" / "un_speeches_palestine.csv"))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1

    # De-duplicate byte-identical records, remembering every record id that pointed at them.
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in pdfs:
        by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(p)
    print(f"{len(pdfs)} PDFs -> {len(by_hash)} unique meeting records")

    rows = []
    for i, (h, paths) in enumerate(sorted(by_hash.items(), key=lambda kv: kv[1][0].name), 1):
        src = paths[0]
        try:
            cover, turns = extract_document(src)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {src.name}: {exc}", file=sys.stderr)
            continue
        # Prefer the symbol encoded in the filename when there is one. fetch_meeting_records.py
        # names every file after the symbol it asked ODS for, which is authoritative; the
        # cover-page regex is a guess that misfires on documents whose running head drops the
        # period (`A/ES-10/PV51`), silently attributing the turns to whichever OTHER meeting
        # the text happens to cross-reference first. Files named any other way (the old
        # collector used record ids) fall through to the parsed value unchanged.
        from_name = symbol_from_filename(src)
        if from_name:
            cover["doc_symbol"] = from_name
        record_ids = ",".join(sorted(p.stem for p in paths))
        for order, t in enumerate(turns, 1):
            sub = substantive_words(t["speech"])
            rows.append(
                {
                    "ambassador_name": t["name"],
                    "country": t["country"],
                    "speech": t["speech"],
                    "is_procedural": sub == 0,
                    "substantive_word_count": sub,
                    "role": t["role"],
                    "language": t["language"],
                    "speaker_header": t["header"],
                    "doc_symbol": cover["doc_symbol"],
                    "meeting_date": cover["meeting_date"],
                    "meeting_number": cover["meeting_number"],
                    "turn_index": order,
                    "word_count": len(t["speech"].split()),
                    "source_pdf": str(src),
                    "record_ids": record_ids,
                }
            )
        print(f"  [{i:3}/{len(by_hash)}] {src.name} {cover['doc_symbol']:<28} {len(turns):3} turns")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} speeches to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
