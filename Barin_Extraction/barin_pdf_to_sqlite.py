from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import pdfplumber

PHASE_RE = re.compile(r"^(GAS|LIQ|SOL(?:-[A-Z0-9]+)?)$")
PRINTED_PAGE_RE = re.compile(r"^\d+$")
FORMULA_CHAR_RE = re.compile(r"^[A-Za-z0-9\[\]\(\)\+\-]+$")
FORMULA_GROUP_RE = re.compile(r"[A-Z][a-z]?(?:\d+)?")

DATA_COLUMNS = (
    "T_K",
    "Cp_J_molK",
    "S_J_molK",
    "minus_G_minus_H298_over_T_J_molK",
    "H_kJ_mol",
    "H_minus_H298_kJ_mol",
    "G_kJ_mol",
    "dHf_kJ_mol",
    "dGf_kJ_mol",
    "logKf",
)

DEFAULT_COLUMN_CENTERS = {
    "phase": 63.0,
    "T_K": 101.0,
    "Cp_J_molK": 145.0,
    "S_J_molK": 190.0,
    "minus_G_minus_H298_over_T_J_molK": 235.0,
    "H_kJ_mol": 287.0,
    "H_minus_H298_kJ_mol": 326.0,
    "G_kJ_mol": 375.0,
    "dHf_kJ_mol": 418.0,
    "dGf_kJ_mol": 463.0,
    "logKf": 511.0,
}


@dataclass
class Row:
    page_no: int
    row_no: int
    y_center: float
    words: list[dict]
    text: str


@dataclass
class Block:
    page_no: int
    block_index: int
    start_row_no: int
    end_row_no: int
    header_start_row_no: int
    header_end_row_no: int
    refs_start_row_no: int | None
    refs_header_end_row_no: int | None


@dataclass
class HeaderMetadata:
    raw_header: str
    name_raw: str
    name_key: str
    formula_raw: str | None
    formula_key: str | None
    formula_confidence: int
    molar_mass: float | None
    is_gas_header: bool
    is_continuation: bool
    entry_key: tuple


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalized_row_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def normalize_key(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"[^A-Z0-9]+", "", text.upper())
    return cleaned or None


def normalize_formula_key(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"\s+", "", text)
    return cleaned or None


def canonicalize_numeric_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = (
        text.strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("O", "0")
        .replace("o", "0")
    )
    return cleaned


def to_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = canonicalize_numeric_text(text)
    if cleaned is None:
        return None
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(",", ".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def row_text(words: Sequence[dict]) -> str:
    ordered = sorted(words, key=lambda word: word["x0"])
    return normalize_space(" ".join(word["text"] for word in ordered))


def extract_page_words(page) -> list[dict]:
    page = page.dedupe_chars()
    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        extra_attrs=["upright", "fontname", "size"],
    )
    upright_words: list[dict] = []
    for index, word in enumerate(words):
        if not word.get("upright", True):
            continue
        text = normalize_space(word["text"])
        if not text:
            continue
        copy = dict(word)
        copy["text"] = text
        copy["word_index"] = index
        upright_words.append(copy)
    return upright_words


def cluster_rows(page_no: int, words: Sequence[dict], y_tol: float = 2.2) -> list[Row]:
    buckets: list[dict] = []
    for word in sorted(words, key=lambda entry: ((entry["top"] + entry["bottom"]) / 2.0, entry["x0"])):
        y_center = (word["top"] + word["bottom"]) / 2.0
        matched = None
        for bucket in buckets:
            if abs(y_center - bucket["y_center"]) <= y_tol:
                matched = bucket
                break
        if matched is None:
            buckets.append({"y_center": y_center, "words": [word]})
        else:
            matched["words"].append(word)
            matched["y_center"] = median(
                ((item["top"] + item["bottom"]) / 2.0) for item in matched["words"]
            )

    rows: list[Row] = []
    for row_no, bucket in enumerate(sorted(buckets, key=lambda item: item["y_center"]), start=1):
        ordered_words = sorted(bucket["words"], key=lambda entry: entry["x0"])
        for word in ordered_words:
            word["row_no"] = row_no
        rows.append(
            Row(
                page_no=page_no,
                row_no=row_no,
                y_center=round(bucket["y_center"], 3),
                words=ordered_words,
                text=row_text(ordered_words),
            )
        )
    return rows


def is_page_label_row(row: Row) -> bool:
    compact = row.text.replace(" ", "")
    return row.y_center < 80 and bool(PRINTED_PAGE_RE.fullmatch(compact))


def extract_printed_page_label(rows: Sequence[Row]) -> str | None:
    for row in rows:
        if is_page_label_row(row):
            return row.text.replace(" ", "")
    return None


def extract_phase_label_from_words(words: Sequence[dict]) -> str | None:
    if not words:
        return None
    token = normalize_space(words[0]["text"]).upper().rstrip(".")
    token = token.replace(" ", "")
    if PHASE_RE.fullmatch(token):
        return token
    return None


def detect_reference_marker(row: Row) -> bool:
    return normalized_row_text(row.text).startswith("REFERENCES")


def is_header_candidate(row: Row) -> bool:
    normalized = normalized_row_text(row.text)
    if "PHASE" not in normalized:
        return False
    if "H-H298" not in row.text and "GH298" not in normalized:
        return False
    return "T" in normalized


def is_numericish_token(text: str) -> bool:
    compact = canonicalize_numeric_text(text) or ""
    compact = compact.replace(" ", "")
    if not compact or not any(character.isdigit() for character in compact):
        return False
    return all(character in "-+./0123456789AOILaoil" for character in compact)


def extract_numeric_groups(words: Sequence[dict], *, skip_phase: bool) -> list[str]:
    ordered = sorted(words, key=lambda entry: entry["x0"])
    remaining = ordered[1:] if skip_phase and ordered else ordered
    groups: list[list[dict]] = []
    for word in remaining:
        if not is_numericish_token(word["text"]):
            continue
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]
        gap = word["x0"] - previous["x1"]
        if gap <= 8:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [normalize_space(" ".join(part["text"] for part in group)) for group in groups]


def extract_numeric_word_groups(words: Sequence[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda entry: entry["x0"]):
        if not is_numericish_token(word["text"]):
            continue
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]
        gap = word["x0"] - previous["x1"]
        if gap <= 8:
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def looks_like_data_row(row: Row) -> bool:
    phase = extract_phase_label_from_words(row.words)
    numeric_groups = extract_numeric_groups(row.words, skip_phase=bool(phase))
    if phase and numeric_groups:
        return True
    return len(numeric_groups) >= 4


def find_header_band_end(rows: Sequence[Row], header_index: int) -> int:
    end_index = header_index
    for index in range(header_index, len(rows) - 1):
        next_index = index + 1
        gap = rows[next_index].y_center - rows[index].y_center
        if gap > 16:
            end_index = index
            break
        if looks_like_data_row(rows[next_index]):
            end_index = index
            break
        end_index = next_index
    return end_index


def looks_like_header_metadata(row: Row) -> bool:
    if is_page_label_row(row) or detect_reference_marker(row):
        return False
    if looks_like_data_row(row):
        return False
    return bool(re.search(r"[A-Za-z]", row.text))


def find_block_start(rows: Sequence[Row], header_index: int) -> int:
    start_index = header_index
    candidate = header_index - 1
    if candidate >= 0 and not is_page_label_row(rows[candidate]) and not detect_reference_marker(rows[candidate]):
        start_index = candidate
    while start_index > 0:
        previous = start_index - 1
        gap = rows[start_index].y_center - rows[previous].y_center
        if gap > 18:
            break
        if not looks_like_header_metadata(rows[previous]):
            break
        start_index = previous
    return start_index


def detect_blocks(rows: Sequence[Row]) -> list[Block]:
    header_indices = [index for index, row in enumerate(rows) if is_header_candidate(row)]
    blocks: list[Block] = []
    if not header_indices:
        return blocks

    start_indices = [find_block_start(rows, header_index) for header_index in header_indices]
    for block_index, header_index in enumerate(header_indices):
        start_index = start_indices[block_index]
        end_index = start_indices[block_index + 1] - 1 if block_index + 1 < len(start_indices) else len(rows) - 1
        header_end_index = find_header_band_end(rows, header_index)
        refs_start_index = None
        refs_header_end_index = None
        for scan in range(header_end_index + 1, end_index + 1):
            if detect_reference_marker(rows[scan]):
                refs_start_index = scan
                refs_header_end_index = scan
                phase_rows_started = False
                for header_scan in range(scan + 1, end_index + 1):
                    if extract_phase_label_from_words(rows[header_scan].words):
                        phase_rows_started = True
                        break
                    refs_header_end_index = header_scan
                if not phase_rows_started:
                    refs_header_end_index = end_index
                break

        blocks.append(
            Block(
                page_no=rows[start_index].page_no,
                block_index=block_index,
                start_row_no=rows[start_index].row_no,
                end_row_no=rows[end_index].row_no,
                header_start_row_no=rows[header_index].row_no,
                header_end_row_no=rows[header_end_index].row_no,
                refs_start_row_no=rows[refs_start_index].row_no if refs_start_index is not None else None,
                refs_header_end_row_no=rows[refs_header_end_index].row_no if refs_header_end_index is not None else None,
            )
        )
    return blocks


def row_by_number(rows: Sequence[Row], row_no: int) -> Row:
    return rows[row_no - 1]


def rows_for_block(rows: Sequence[Row], block: Block) -> list[Row]:
    return [row for row in rows if block.start_row_no <= row.row_no <= block.end_row_no]


def words_between(rows: Sequence[Row], start_row_no: int, end_row_no: int) -> list[dict]:
    words: list[dict] = []
    for row in rows:
        if start_row_no <= row.row_no <= end_row_no:
            words.extend(row.words)
    return words


def infer_column_centers(header_rows: Sequence[Row]) -> dict[str, float]:
    observed: dict[str, float] = {}
    log_parts: list[float] = []
    header_words = words_between(header_rows, header_rows[0].row_no, header_rows[-1].row_no)
    for word in header_words:
        text = normalize_space(word["text"])
        normalized = normalized_row_text(text)
        x_mid = (word["x0"] + word["x1"]) / 2.0
        if normalized == "PHASE" or normalized.startswith("PHASE"):
            observed["phase"] = x_mid
        elif normalized == "T":
            observed["T_K"] = x_mid
        elif normalized in {"CP", "C"} or normalized.startswith("CP") or normalized.startswith("C"):
            observed["Cp_J_molK"] = x_mid
        elif normalized == "S":
            observed["S_J_molK"] = x_mid
        elif "GH298" in normalized:
            observed["minus_G_minus_H298_over_T_J_molK"] = x_mid
        elif normalized == "H":
            observed["H_kJ_mol"] = x_mid
        elif "HH298" in normalized:
            observed["H_minus_H298_kJ_mol"] = x_mid
        elif normalized == "G":
            observed["G_kJ_mol"] = x_mid
        elif normalized in {"AHF", "4HF", "DHF"} or normalized.endswith("HF"):
            observed["dHf_kJ_mol"] = x_mid
        elif normalized in {"AGF", "AGT", "DGF"} or normalized.endswith("GF") or normalized.endswith("GT"):
            observed["dGf_kJ_mol"] = x_mid
        elif normalized in {"LOG", "KF", "LOGKF"} or ("LOG" in normalized and "KF" in normalized):
            log_parts.append(x_mid)

    if log_parts:
        observed["logKf"] = sum(log_parts) / len(log_parts)

    centers = dict(DEFAULT_COLUMN_CENTERS)
    offsets = [observed[key] - DEFAULT_COLUMN_CENTERS[key] for key in observed if key in DEFAULT_COLUMN_CENTERS]
    page_offset = median(offsets) if offsets else 0.0
    for key, default_value in DEFAULT_COLUMN_CENTERS.items():
        centers[key] = observed.get(key, default_value + page_offset)
    return centers


def assign_words_to_columns(words: Sequence[dict], centers: dict[str, float]) -> dict[str, str]:
    assigned: dict[str, list[str]] = {key: [] for key in centers}
    ordered_keys = list(centers)
    for word in sorted(words, key=lambda entry: entry["x0"]):
        x_mid = (word["x0"] + word["x1"]) / 2.0
        nearest_key = min(ordered_keys, key=lambda key: abs(x_mid - centers[key]))
        assigned[nearest_key].append(word["text"])
    return {key: normalize_space(" ".join(parts)) for key, parts in assigned.items()}


def fallback_row_values(words: Sequence[dict]) -> dict[str, str]:
    phase = extract_phase_label_from_words(words)
    groups = extract_numeric_groups(words, skip_phase=bool(phase))
    values = {key: "" for key in DATA_COLUMNS}
    for key, value in zip(DATA_COLUMNS, groups):
        values[key] = value
    return values


def formula_confidence(text: str | None) -> int:
    if not text:
        return -1
    compact = normalize_formula_key(text)
    if not compact:
        return -1
    stripped = re.sub(r"\[[A-Za-z]+\]$", "", compact)
    if not FORMULA_CHAR_RE.fullmatch(compact):
        return -1
    groups = FORMULA_GROUP_RE.findall(stripped)
    if "".join(groups) != stripped:
        return 0
    if len(groups) == 1:
        match = re.fullmatch(r"([A-Z][a-z]?)(\d+)?", stripped)
        if match and match.group(2):
            return 0 if int(match.group(2)) > 20 else 1
        return 2
    return 2


def parse_header_metadata(block_rows: Sequence[Row]) -> HeaderMetadata:
    header_words = [word for row in block_rows for word in row.words]
    raw_header = " | ".join(row.text for row in block_rows)
    upper_header = raw_header.upper()
    is_continuation = "[CONTINUED]" in upper_header
    is_gas_header = "(GAS)" in upper_header

    mass = None
    mass_word_indices: set[int] = set()
    numeric_groups = extract_numeric_word_groups(header_words)
    if numeric_groups:
        center_x = 288.0
        scored_groups: list[tuple[float, list[dict], float | None]] = []
        for group in numeric_groups:
            text = normalize_space(" ".join(word["text"] for word in group))
            value = to_float(text)
            if value is None:
                continue
            group_mid = (group[0]["x0"] + group[-1]["x1"]) / 2.0
            score = (1000.0 if "." in text else 0.0) + abs(group_mid - center_x)
            scored_groups.append((score, group, value))
        if scored_groups:
            _, chosen_mass_group, chosen_mass_value = max(scored_groups, key=lambda item: item[0])
            mass = chosen_mass_value
            mass_word_indices.update(word["word_index"] for word in chosen_mass_group)

    side_groups: list[tuple[int, list[dict]]] = []
    left_words = [word for word in header_words if word["x0"] < 150 and word["word_index"] not in mass_word_indices]
    right_words = [word for word in header_words if word["x0"] > 430 and word["word_index"] not in mass_word_indices]
    if left_words:
        side_groups.append((formula_confidence(" ".join(word["text"] for word in left_words)), left_words))
    if right_words:
        side_groups.append((formula_confidence(" ".join(word["text"] for word in right_words)), right_words))
    side_groups.sort(key=lambda item: item[0], reverse=True)

    chosen_formula_words: list[dict] = []
    formula_raw = None
    formula_key = None
    formula_score = -1
    if side_groups and side_groups[0][0] >= 0:
        formula_score, chosen_formula_words = side_groups[0]
        formula_raw = normalize_space(" ".join(word["text"] for word in chosen_formula_words))
        formula_key = normalize_formula_key(formula_raw)

    excluded_word_indices = set(mass_word_indices)
    excluded_word_indices.update(word["word_index"] for word in chosen_formula_words)

    name_parts: list[str] = []
    for word in sorted(header_words, key=lambda entry: entry["x0"]):
        if word["word_index"] in excluded_word_indices:
            continue
        text = normalize_space(word["text"])
        if text.upper() == "[CONTINUED]":
            continue
        if text.upper() == "(GAS)":
            continue
        name_parts.append(text)

    name_raw = normalize_space(" ".join(name_parts))
    name_raw = normalize_space(name_raw.replace("(GAS)", ""))
    name_key = normalize_key(name_raw) or "UNKNOWN"
    rounded_mass = round(mass or -1.0, 3)
    if formula_score >= 2 and formula_key:
        entry_key = ("formula", formula_key, name_key, rounded_mass, int(is_gas_header))
    else:
        entry_key = ("fallback", name_key, rounded_mass, int(is_gas_header))

    return HeaderMetadata(
        raw_header=raw_header,
        name_raw=name_raw,
        name_key=name_key,
        formula_raw=formula_raw,
        formula_key=formula_key,
        formula_confidence=formula_score,
        molar_mass=mass,
        is_gas_header=is_gas_header,
        is_continuation=is_continuation,
        entry_key=entry_key,
    )


def parse_reference_row_text(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    parts = normalize_space(text).split()
    if not parts:
        return None, None, None, None
    phase_label = parts[0].upper()
    hs_ref = parts[1] if len(parts) > 1 else None
    cp_ref = parts[2] if len(parts) > 2 else None
    remarks = " ".join(parts[3:]).strip() or None
    return phase_label, hs_ref, cp_ref, remarks


def section_kind_for_row(block: Block, row_no: int) -> str:
    if row_no < block.header_start_row_no:
        return "entry_header"
    if block.header_start_row_no <= row_no <= block.header_end_row_no:
        return "main_header"
    if block.refs_start_row_no is None:
        return "body"
    if row_no == block.refs_start_row_no:
        return "references_marker"
    if block.refs_start_row_no < row_no <= (block.refs_header_end_row_no or block.refs_start_row_no):
        return "references_header"
    if row_no > (block.refs_header_end_row_no or block.refs_start_row_no):
        return "reference"
    return "body"


def insert_parse_issue(
    conn: sqlite3.Connection,
    document_id: int,
    *,
    page_no: int | None,
    entry_id: int | None,
    block_index: int | None,
    severity: str,
    issue_type: str,
    message: str,
    raw_context: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO parse_issue(document_id, page_no, entry_id, block_index, severity, issue_type, message, raw_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, page_no, entry_id, block_index, severity, issue_type, message, raw_context),
    )


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = MEMORY;
        PRAGMA synchronous = OFF;

        CREATE TABLE IF NOT EXISTS document (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS page (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            page_no INTEGER NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            printed_page_label TEXT,
            UNIQUE(document_id, page_no)
        );

        CREATE TABLE IF NOT EXISTS entry (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            formula_raw TEXT,
            formula_key TEXT,
            name_raw TEXT NOT NULL,
            name_key TEXT NOT NULL,
            molar_mass REAL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            is_gas_header INTEGER NOT NULL DEFAULT 0,
            raw_header TEXT
        );

        CREATE TABLE IF NOT EXISTS entry_block (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL REFERENCES entry(id),
            page_no INTEGER NOT NULL,
            block_index INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            row_end INTEGER NOT NULL,
            y0 REAL NOT NULL,
            y1 REAL NOT NULL,
            is_continuation INTEGER NOT NULL DEFAULT 0,
            raw_header TEXT
        );

        CREATE TABLE IF NOT EXISTS phase (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL REFERENCES entry(id),
            phase_label TEXT NOT NULL,
            phase_order INTEGER NOT NULL,
            UNIQUE(entry_id, phase_label)
        );

        CREATE TABLE IF NOT EXISTS datum (
            id INTEGER PRIMARY KEY,
            phase_id INTEGER NOT NULL REFERENCES phase(id),
            page_no INTEGER NOT NULL,
            row_no INTEGER NOT NULL,
            T_K REAL,
            Cp_J_molK REAL,
            S_J_molK REAL,
            minus_G_minus_H298_over_T_J_molK REAL,
            H_kJ_mol REAL,
            H_minus_H298_kJ_mol REAL,
            G_kJ_mol REAL,
            dHf_kJ_mol REAL,
            dGf_kJ_mol REAL,
            logKf REAL,
            raw_row_text TEXT
        );

        CREATE TABLE IF NOT EXISTS transition (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL REFERENCES entry(id),
            page_no INTEGER NOT NULL,
            row_no INTEGER NOT NULL,
            from_phase_label TEXT,
            to_phase_label TEXT,
            T_K REAL,
            dH_transition_kJ_mol REAL,
            dS_transition_J_molK REAL,
            note TEXT,
            raw_row_text TEXT
        );

        CREATE TABLE IF NOT EXISTS reference_row (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL REFERENCES entry(id),
            page_no INTEGER NOT NULL,
            phase_label TEXT,
            hs_ref TEXT,
            cp_ref TEXT,
            remarks TEXT,
            raw_row_text TEXT
        );

        CREATE TABLE IF NOT EXISTS raw_word (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            page_no INTEGER NOT NULL,
            block_index INTEGER,
            row_no INTEGER NOT NULL,
            word_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            x0 REAL NOT NULL,
            x1 REAL NOT NULL,
            top REAL NOT NULL,
            bottom REAL NOT NULL,
            upright INTEGER NOT NULL,
            fontname TEXT,
            size REAL
        );

        CREATE TABLE IF NOT EXISTS raw_row (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            page_no INTEGER NOT NULL,
            block_index INTEGER,
            row_no INTEGER NOT NULL,
            y_center REAL NOT NULL,
            section_kind TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS parse_issue (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES document(id),
            page_no INTEGER,
            entry_id INTEGER,
            block_index INTEGER,
            severity TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            message TEXT NOT NULL,
            raw_context TEXT
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    for table in (
        "parse_issue",
        "raw_row",
        "raw_word",
        "reference_row",
        "transition",
        "datum",
        "phase",
        "entry_block",
        "entry",
        "page",
        "document",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def ensure_phase(
    conn: sqlite3.Connection,
    phase_state: dict[int, dict],
    entry_id: int,
    phase_label: str,
) -> int:
    state = phase_state.setdefault(entry_id, {"order": 0, "ids": {}})
    if phase_label in state["ids"]:
        return state["ids"][phase_label]
    state["order"] += 1
    cursor = conn.execute(
        "INSERT INTO phase(entry_id, phase_label, phase_order) VALUES (?, ?, ?)",
        (entry_id, phase_label, state["order"]),
    )
    phase_id = int(cursor.lastrowid)
    state["ids"][phase_label] = phase_id
    return phase_id


def validate_phase_temperatures(conn: sqlite3.Connection, document_id: int) -> None:
    rows = conn.execute(
        """
        SELECT p.id, p.entry_id, p.phase_label, d.page_no, d.row_no, d.T_K
        FROM phase p
        JOIN datum d ON d.phase_id = p.id
        WHERE d.T_K IS NOT NULL
        ORDER BY p.id, d.T_K
        """
    ).fetchall()
    last_by_phase: dict[int, float] = {}
    for phase_id, entry_id, phase_label, page_no, row_no, temperature in rows:
        if temperature is None:
            continue
        previous = last_by_phase.get(phase_id)
        if previous is not None and temperature <= previous:
            insert_parse_issue(
                conn,
                document_id,
                page_no=page_no,
                entry_id=entry_id,
                block_index=None,
                severity="warning",
                issue_type="temperature_not_increasing",
                message=f"{phase_label} temperature sequence is not strictly increasing at row {row_no}",
                raw_context=str(temperature),
            )
        last_by_phase[phase_id] = temperature


def validate_smoothness(conn: sqlite3.Connection, document_id: int) -> None:
    rows = conn.execute(
        """
        SELECT p.id, p.entry_id, p.phase_label, d.page_no, d.row_no, d.T_K, d.S_J_molK, d.H_minus_H298_kJ_mol
        FROM phase p
        JOIN datum d ON d.phase_id = p.id
        WHERE d.T_K IS NOT NULL
        ORDER BY p.id, d.T_K
        """
    ).fetchall()
    previous_by_phase: dict[int, tuple[float | None, float | None, float | None, int, int]] = {}
    for phase_id, entry_id, phase_label, page_no, row_no, temperature, entropy, enthalpy_delta in rows:
        previous = previous_by_phase.get(phase_id)
        if previous is not None:
            _, previous_entropy, previous_enthalpy, previous_page, previous_row = previous
            if entropy is not None and previous_entropy is not None and entropy + 1.0 < previous_entropy:
                insert_parse_issue(
                    conn,
                    document_id,
                    page_no=page_no,
                    entry_id=entry_id,
                    block_index=None,
                    severity="warning",
                    issue_type="entropy_drop",
                    message=f"{phase_label} entropy drops between rows {previous_page}:{previous_row} and {page_no}:{row_no}",
                    raw_context=f"{previous_entropy} -> {entropy}",
                )
            if enthalpy_delta is not None and previous_enthalpy is not None and enthalpy_delta + 1.0 < previous_enthalpy:
                insert_parse_issue(
                    conn,
                    document_id,
                    page_no=page_no,
                    entry_id=entry_id,
                    block_index=None,
                    severity="warning",
                    issue_type="enthalpy_drop",
                    message=f"{phase_label} H-H298 drops between rows {previous_page}:{previous_row} and {page_no}:{row_no}",
                    raw_context=f"{previous_enthalpy} -> {enthalpy_delta}",
                )
        previous_by_phase[phase_id] = (temperature, entropy, enthalpy_delta, page_no, row_no)


def parse_page_numbers(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(chunk))
    return selected


def parse_pdf_to_sqlite(
    pdf_path: str | Path,
    db_path: str | Path,
    *,
    replace: bool = False,
    max_pages: int | None = None,
    page_numbers: Sequence[int] | None = None,
) -> dict[str, int]:
    pdf_path = Path(pdf_path)
    db_path = Path(db_path)
    selected_pages = set(page_numbers) if page_numbers else None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    if replace and db_path.exists():
        reset_db(conn)
    init_db(conn)

    document_cursor = conn.execute(
        "INSERT INTO document(path, sha256) VALUES (?, ?)",
        (str(pdf_path), compute_sha256(pdf_path)),
    )
    document_id = int(document_cursor.lastrowid)
    open_entries: dict[tuple, int] = {}
    phase_state: dict[int, dict] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if max_pages is not None and page_no > max_pages:
                break
            if selected_pages is not None and page_no not in selected_pages:
                continue

            words = extract_page_words(page)
            rows = cluster_rows(page_no, words)
            printed_page_label = extract_printed_page_label(rows)
            conn.execute(
                """
                INSERT INTO page(document_id, page_no, width, height, printed_page_label)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, page_no, float(page.width), float(page.height), printed_page_label),
            )

            blocks = detect_blocks(rows)
            row_to_block: dict[int, Block] = {}
            row_kinds: dict[int, str] = {}
            for row in rows:
                row_kinds[row.row_no] = "page_label" if is_page_label_row(row) else "unassigned"
            for block in blocks:
                for row_no in range(block.start_row_no, block.end_row_no + 1):
                    row_to_block[row_no] = block
                    row_kinds[row_no] = section_kind_for_row(block, row_no)

            if not blocks:
                insert_parse_issue(
                    conn,
                    document_id,
                    page_no=page_no,
                    entry_id=None,
                    block_index=None,
                    severity="warning",
                    issue_type="missing_block",
                    message="Could not detect any entry blocks on page",
                    raw_context="\n".join(row.text for row in rows),
                )

            for block in blocks:
                block_rows = rows_for_block(rows, block)
                metadata_rows = [row for row in block_rows if row.row_no < block.header_start_row_no]
                if not metadata_rows:
                    insert_parse_issue(
                        conn,
                        document_id,
                        page_no=page_no,
                        entry_id=None,
                        block_index=block.block_index,
                        severity="warning",
                        issue_type="missing_entry_header",
                        message="Could not find block header rows above main thermochemical header",
                        raw_context="\n".join(row.text for row in block_rows),
                    )
                    continue

                metadata = parse_header_metadata(metadata_rows)
                if metadata.is_continuation:
                    entry_id = open_entries.get(metadata.entry_key)
                    if entry_id is None:
                        insert_parse_issue(
                            conn,
                            document_id,
                            page_no=page_no,
                            entry_id=None,
                            block_index=block.block_index,
                            severity="warning",
                            issue_type="continuation_unresolved",
                            message="Continuation block could not be linked to an earlier entry",
                            raw_context=metadata.raw_header,
                        )
                        entry_cursor = conn.execute(
                            """
                            INSERT INTO entry(
                                document_id, formula_raw, formula_key, name_raw, name_key, molar_mass,
                                page_start, page_end, is_gas_header, raw_header
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                document_id,
                                metadata.formula_raw,
                                metadata.formula_key,
                                metadata.name_raw or metadata.raw_header,
                                metadata.name_key,
                                metadata.molar_mass,
                                page_no,
                                page_no,
                                int(metadata.is_gas_header),
                                metadata.raw_header,
                            ),
                        )
                        entry_id = int(entry_cursor.lastrowid)
                        open_entries[metadata.entry_key] = entry_id
                    else:
                        conn.execute("UPDATE entry SET page_end = ? WHERE id = ?", (page_no, entry_id))
                else:
                    entry_cursor = conn.execute(
                        """
                        INSERT INTO entry(
                            document_id, formula_raw, formula_key, name_raw, name_key, molar_mass,
                            page_start, page_end, is_gas_header, raw_header
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            metadata.formula_raw,
                            metadata.formula_key,
                            metadata.name_raw or metadata.raw_header,
                            metadata.name_key,
                            metadata.molar_mass,
                            page_no,
                            page_no,
                            int(metadata.is_gas_header),
                            metadata.raw_header,
                        ),
                    )
                    entry_id = int(entry_cursor.lastrowid)
                    open_entries[metadata.entry_key] = entry_id

                if metadata.formula_confidence < 2:
                    insert_parse_issue(
                        conn,
                        document_id,
                        page_no=page_no,
                        entry_id=entry_id,
                        block_index=block.block_index,
                        severity="info",
                        issue_type="low_confidence_formula",
                        message="Entry formula is missing or low-confidence; fallback identity was used",
                        raw_context=metadata.raw_header,
                    )

                header_rows = [
                    row for row in block_rows if block.header_start_row_no <= row.row_no <= block.header_end_row_no
                ]
                column_centers = infer_column_centers(header_rows)
                conn.execute(
                    """
                    INSERT INTO entry_block(entry_id, page_no, block_index, row_start, row_end, y0, y1, is_continuation, raw_header)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        page_no,
                        block.block_index,
                        block.start_row_no,
                        block.end_row_no,
                        min(word["top"] for row in block_rows for word in row.words),
                        max(word["bottom"] for row in block_rows for word in row.words),
                        int(metadata.is_continuation),
                        metadata.raw_header,
                    ),
                )

                body_end = (block.refs_start_row_no - 1) if block.refs_start_row_no is not None else block.end_row_no
                current_phase: str | None = None
                current_temperature: float | None = None
                pending_transition_id: int | None = None
                for row in block_rows:
                    if row.row_no <= block.header_end_row_no or row.row_no > body_end:
                        continue

                    phase_label = extract_phase_label_from_words(row.words)
                    if phase_label:
                        current_phase = phase_label
                        if pending_transition_id is not None:
                            conn.execute("UPDATE transition SET to_phase_label = ? WHERE id = ?", (phase_label, pending_transition_id))
                            pending_transition_id = None

                    assigned = assign_words_to_columns(row.words, column_centers)
                    fallback_values = fallback_row_values(row.words)
                    values = {}
                    for key in DATA_COLUMNS:
                        values[key] = assigned.get(key, "") or fallback_values.get(key, "")

                    temperature = to_float(values["T_K"])
                    numeric_count = sum(1 for key in DATA_COLUMNS[1:] if to_float(values[key]) is not None)
                    numeric_groups = extract_numeric_groups(row.words, skip_phase=bool(phase_label))
                    if temperature is not None and current_phase and numeric_count >= 5:
                        phase_id = ensure_phase(conn, phase_state, entry_id, current_phase)
                        conn.execute(
                            """
                            INSERT INTO datum(
                                phase_id, page_no, row_no, T_K, Cp_J_molK, S_J_molK,
                                minus_G_minus_H298_over_T_J_molK, H_kJ_mol, H_minus_H298_kJ_mol,
                                G_kJ_mol, dHf_kJ_mol, dGf_kJ_mol, logKf, raw_row_text
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                phase_id,
                                page_no,
                                row.row_no,
                                temperature,
                                to_float(values["Cp_J_molK"]),
                                to_float(values["S_J_molK"]),
                                to_float(values["minus_G_minus_H298_over_T_J_molK"]),
                                to_float(values["H_kJ_mol"]),
                                to_float(values["H_minus_H298_kJ_mol"]),
                                to_float(values["G_kJ_mol"]),
                                to_float(values["dHf_kJ_mol"]),
                                to_float(values["dGf_kJ_mol"]),
                                to_float(values["logKf"]),
                                row.text,
                            ),
                        )
                        current_temperature = temperature
                        row_kinds[row.row_no] = "datum"
                        continue

                    if current_phase and numeric_count >= 5 and len(numeric_groups) >= 9:
                        phase_id = ensure_phase(conn, phase_state, entry_id, current_phase)
                        conn.execute(
                            """
                            INSERT INTO datum(
                                phase_id, page_no, row_no, T_K, Cp_J_molK, S_J_molK,
                                minus_G_minus_H298_over_T_J_molK, H_kJ_mol, H_minus_H298_kJ_mol,
                                G_kJ_mol, dHf_kJ_mol, dGf_kJ_mol, logKf, raw_row_text
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                phase_id,
                                page_no,
                                row.row_no,
                                temperature,
                                to_float(values["Cp_J_molK"]),
                                to_float(values["S_J_molK"]),
                                to_float(values["minus_G_minus_H298_over_T_J_molK"]),
                                to_float(values["H_kJ_mol"]),
                                to_float(values["H_minus_H298_kJ_mol"]),
                                to_float(values["G_kJ_mol"]),
                                to_float(values["dHf_kJ_mol"]),
                                to_float(values["dGf_kJ_mol"]),
                                to_float(values["logKf"]),
                                row.text,
                            ),
                        )
                        row_kinds[row.row_no] = "datum"
                        insert_parse_issue(
                            conn,
                            document_id,
                            page_no=page_no,
                            entry_id=entry_id,
                            block_index=block.block_index,
                            severity="warning",
                            issue_type="malformed_temperature",
                            message="Datum row was stored with a null temperature because the temperature token could not be parsed",
                            raw_context=row.text,
                        )
                        continue

                    if current_phase and current_temperature is not None and len(numeric_groups) in {1, 2}:
                        entropy_value = to_float(numeric_groups[0]) if numeric_groups else None
                        enthalpy_value = to_float(numeric_groups[1]) if len(numeric_groups) > 1 else None
                        if entropy_value is not None or enthalpy_value is not None:
                            cursor = conn.execute(
                                """
                                INSERT INTO transition(
                                    entry_id, page_no, row_no, from_phase_label, to_phase_label,
                                    T_K, dH_transition_kJ_mol, dS_transition_J_molK, note, raw_row_text
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    entry_id,
                                    page_no,
                                    row.row_no,
                                    current_phase,
                                    None,
                                    current_temperature,
                                    enthalpy_value,
                                    entropy_value,
                                    None,
                                    row.text,
                                ),
                            )
                            pending_transition_id = int(cursor.lastrowid)
                            row_kinds[row.row_no] = "transition"
                            continue

                    if phase_label or numeric_groups:
                        row_kinds[row.row_no] = "parse_issue"
                        insert_parse_issue(
                            conn,
                            document_id,
                            page_no=page_no,
                            entry_id=entry_id,
                            block_index=block.block_index,
                            severity="warning",
                            issue_type="unparsed_body_row",
                            message="Body row could not be parsed as datum or transition",
                            raw_context=row.text,
                        )

                if block.refs_start_row_no is not None:
                    ref_data_start = (block.refs_header_end_row_no or block.refs_start_row_no) + 1
                    for row in block_rows:
                        if row.row_no < ref_data_start or row.row_no > block.end_row_no:
                            continue
                        phase_label, hs_ref, cp_ref, remarks = parse_reference_row_text(row.text)
                        conn.execute(
                            """
                            INSERT INTO reference_row(entry_id, page_no, phase_label, hs_ref, cp_ref, remarks, raw_row_text)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (entry_id, page_no, phase_label, hs_ref, cp_ref, remarks, row.text),
                        )
                        row_kinds[row.row_no] = "reference"

            for row in rows:
                block = row_to_block.get(row.row_no)
                conn.execute(
                    """
                    INSERT INTO raw_row(document_id, page_no, block_index, row_no, y_center, section_kind, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        page_no,
                        block.block_index if block else None,
                        row.row_no,
                        row.y_center,
                        row_kinds[row.row_no],
                        row.text,
                    ),
                )

            for word in words:
                block = row_to_block.get(word["row_no"])
                conn.execute(
                    """
                    INSERT INTO raw_word(
                        document_id, page_no, block_index, row_no, word_index, text,
                        x0, x1, top, bottom, upright, fontname, size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        page_no,
                        block.block_index if block else None,
                        word["row_no"],
                        word["word_index"],
                        word["text"],
                        float(word["x0"]),
                        float(word["x1"]),
                        float(word["top"]),
                        float(word["bottom"]),
                        int(bool(word.get("upright", True))),
                        word.get("fontname"),
                        float(word["size"]) if word.get("size") is not None else None,
                    ),
                )

    validate_phase_temperatures(conn, document_id)
    validate_smoothness(conn, document_id)
    conn.commit()
    tables = ("page", "entry_block", "entry", "phase", "datum", "transition", "reference_row", "raw_word", "raw_row", "parse_issue")
    result = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
    conn.close()
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a Barin PDF into a SQLite database.")
    parser.add_argument("--pdf", required=True, help="Path to the Barin PDF.")
    parser.add_argument("--db", required=True, help="Path to the output SQLite database.")
    parser.add_argument("--pages", help="Optional comma-separated page list/ranges such as 1-6 or 1,2,5-7.")
    parser.add_argument("--max-pages", type=int, help="Optional maximum page number to parse from the beginning of the document.")
    parser.add_argument("--replace", action="store_true", help="Overwrite the SQLite file if it already exists.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    summary = parse_pdf_to_sqlite(
        args.pdf,
        args.db,
        replace=args.replace,
        max_pages=args.max_pages,
        page_numbers=sorted(parse_page_numbers(args.pages) or []),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
