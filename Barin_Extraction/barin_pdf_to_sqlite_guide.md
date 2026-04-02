# Barin PDF → SQLite extraction guide

This document is a practical blueprint for extracting *Thermochemical Data of Pure Substances* into a SQLite database using `pdfplumber`, while preserving enough provenance to debug and repair parser errors later.

It is written specifically for the Barin layout shown in the provided excerpt, including:

- entries that span multiple pages,
- multiple entries on one page,
- multiple phase sections inside one entry,
- explicit phase-transition values between sections,
- separate references subsections,
- gas entries that must remain distinct from similarly named species.

---

## 1. Core idea

Do **not** try to go directly from PDF pages to one “final clean dataframe”.

Use a two-layer pipeline:

1. **Raw geometric extraction**
   - page blocks
   - words with coordinates
   - row reconstruction
   - raw row text
2. **Normalized thermochemical tables**
   - entries
   - phases
   - temperature rows
   - transitions
   - references

That design makes the parser repairable. If you only store the final parsed numbers, every bug becomes archaeology.

---

## 2. What makes Barin tricky

From the excerpt, the parser must handle all of the following:

### 2.1 Entry continuation across pages
Example: HAFNIUM continues from one page to the next, with a `[continued]` header.

### 2.2 Multiple entries on one page
Example: one page can contain `HfB2` and `HfBr4` separated by a horizontal rule.

### 2.3 Multiple phases inside one entry
Examples:
- `SOL-A`
- `SOL-B`
- `LIQ`
- `GAS`

### 2.4 Transition rows between phase sections
Some entries contain standalone values between phase blocks. These are not normal temperature rows and should be stored separately.

### 2.5 References subsections
Each entry can end with a mini-table like:

- `Phase`
- `H / S`
- `Cp`
- `Remarks`

### 2.6 Formula/name ambiguity
A species must **not** be keyed by English name alone. Cases like atomic vs diatomic nitrogen can share essentially the same name while corresponding to different formulas.

The stable identity should come primarily from the **formula string** in the header.

---

## 3. Why use `pdfplumber`

The original Barin PDF is machine-readable enough that OCR should not be the primary extraction method.

`pdfplumber` is a good fit because it gives you:

- characters and words with coordinates,
- page cropping,
- deduplication of overlapping text,
- access to page lines/rectangles,
- visual debugging when needed.

For Barin, geometry matters more than simple text extraction order.

---

## 4. Recommended output architecture

### 4.1 High-level objects

Your pipeline should produce these logical entities:

- **document**
- **page**
- **entry**
- **entry block**
- **phase**
- **datum**
- **transition**
- **reference row**
- **raw word**
- **parse issue**

### 4.2 Why separate entry and block

An entry can span multiple pages.
A page can contain multiple entries.

So:
- **entry** = the chemical substance record
- **entry block** = one visible section of that entry on one page

---

## 5. SQLite schema

```sql
CREATE TABLE document (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT
);

CREATE TABLE page (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id),
    page_no INTEGER NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    UNIQUE(document_id, page_no)
);

CREATE TABLE entry (
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
    raw_header TEXT,
    UNIQUE(document_id, formula_key, name_key, COALESCE(molar_mass, -1))
);

CREATE TABLE entry_block (
    id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES entry(id),
    page_no INTEGER NOT NULL,
    block_index INTEGER NOT NULL,
    y0 REAL NOT NULL,
    y1 REAL NOT NULL,
    is_continuation INTEGER NOT NULL DEFAULT 0,
    raw_header TEXT
);

CREATE TABLE phase (
    id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES entry(id),
    phase_label TEXT NOT NULL,
    phase_order INTEGER NOT NULL,
    UNIQUE(entry_id, phase_label, phase_order)
);

CREATE TABLE datum (
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

CREATE TABLE transition (
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

CREATE TABLE reference_row (
    id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES entry(id),
    page_no INTEGER NOT NULL,
    phase_label TEXT,
    hs_ref TEXT,
    cp_ref TEXT,
    remarks TEXT,
    raw_row_text TEXT
);

CREATE TABLE raw_word (
    id INTEGER PRIMARY KEY,
    page_no INTEGER NOT NULL,
    block_index INTEGER,
    text TEXT NOT NULL,
    x0 REAL NOT NULL,
    x1 REAL NOT NULL,
    top REAL NOT NULL,
    bottom REAL NOT NULL,
    fontname TEXT,
    size REAL
);

CREATE TABLE parse_issue (
    id INTEGER PRIMARY KEY,
    page_no INTEGER,
    entry_id INTEGER,
    severity TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    message TEXT NOT NULL,
    raw_context TEXT
);
```

---

## 6. Stable identifiers

### 6.1 Formula is primary identity
Use the header formula as the primary species key.

Examples:
- `Hf`
- `Hf[g]`
- `HfBr4`
- `HfBr4[g]`

These should remain distinct even if the names look very similar.

### 6.2 Suggested normalization

```python
import re

def normalize_formula(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = s.replace(" ", "")
    return s or None


def normalize_name(s: str) -> str:
    return " ".join(s.split()).upper()
```

Do not normalize away gas qualifiers like `[g]`.

---

## 7. Page parsing strategy

### 7.1 Crop away obvious noise
Barin pages often include a vertical Wiley notice on the far right margin.

Before doing anything else, crop the page to remove that margin and possibly a little top/bottom noise.

Example idea:

```python
cropped = page.crop((20, 20, page.width - 35, page.height - 20))
```

Tune these margins after inspecting a few pages.

### 7.2 Deduplicate text
PDFs sometimes contain duplicated characters from overlays.

Use:

```python
page = page.dedupe_chars()
```

### 7.3 Split pages into entry blocks
A page may contain one entry or multiple entries.

Use long horizontal rules to identify splits.

Algorithm:
1. inspect `page.lines`
2. keep lines that are nearly horizontal
3. keep only long ones
4. use those y-values as potential block separators

If no separator exists, treat the page as a single block.

---

## 8. Why not rely on `extract_table()`

The Barin layout is table-like, but not a single clean rectangular table:

- the header metadata sits above the main table,
- transition rows can have different structure,
- the references section uses a different schema,
- continuation headers change slightly.

So the safest route is:

1. extract words with coordinates,
2. cluster into rows,
3. assign each word to a semantic column by x-position.

`extract_table()` may still help for debugging, but it should not be the primary parser.

---

## 9. Word extraction

Use `extract_words` with coordinate information and text styling:

```python
words = block.extract_words(
    x_tolerance=2,
    y_tolerance=2,
    use_text_flow=False,
    keep_blank_chars=False,
    extra_attrs=["fontname", "size"],
    return_chars=True,
)
```

Important choice:
- use `use_text_flow=False`

For Barin, PDF internal reading order is less trustworthy than actual geometry.

---

## 10. Reconstruct rows from coordinates

Cluster words into rows by their `top` coordinate.

```python
def cluster_rows(words, y_tol=2.0):
    words = sorted(words, key=lambda w: (round(w["top"], 2), w["x0"]))
    rows = []
    current = []
    last_top = None

    for w in words:
        if last_top is None or abs(w["top"] - last_top) <= y_tol:
            current.append(w)
        else:
            rows.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
        last_top = w["top"]

    if current:
        rows.append(sorted(current, key=lambda x: x["x0"]))
    return rows


def row_text(row):
    return " ".join(w["text"] for w in row).strip()
```

This is the central primitive for the parser.

---

## 11. Detect the main table header and references section

In each block, search the reconstructed rows for the row containing the main thermochemical header.

Typical tokens include:

- `Phase`
- `T`
- `Cp`
- `S`
- `H-H298`
- `log`

Then search for the start of the references subsection, usually beginning with `References`.

```python
def detect_header_and_body(rows):
    header_idx = None
    refs_idx = None

    for i, row in enumerate(rows):
        txt = row_text(row)
        if "Phase" in txt and "Cp" in txt and "H-H298" in txt:
            header_idx = i
        if txt.startswith("References"):
            refs_idx = i
            break

    return header_idx, refs_idx
```

---

## 12. Infer column anchors from the header row

Do not hard-code absolute x coordinates.

For each block, infer column centers from the detected header row.

Expected semantic columns:

- `phase`
- `T_K`
- `Cp_J_molK`
- `S_J_molK`
- `minus_G_minus_H298_over_T_J_molK`
- `H_kJ_mol`
- `H_minus_H298_kJ_mol`
- `G_kJ_mol`
- `dHf_kJ_mol`
- `dGf_kJ_mol`
- `logKf`

```python
def infer_column_centers(header_row):
    wanted = {
        "phase": ["Phase"],
        "T_K": ["T"],
        "Cp_J_molK": ["Cp"],
        "S_J_molK": ["S"],
        "minus_G_minus_H298_over_T_J_molK": ["-(G-H298)/T", "-(G-H298)rT"],
        "H_kJ_mol": ["H"],
        "H_minus_H298_kJ_mol": ["H-H298"],
        "G_kJ_mol": ["G"],
        "dHf_kJ_mol": ["AHf", "ΔHf"],
        "dGf_kJ_mol": ["AGf", "ΔGf"],
        "logKf": ["log", "logKf"],
    }

    centers = {}
    for w in header_row:
        t = w["text"]
        xmid = 0.5 * (w["x0"] + w["x1"])
        for key, aliases in wanted.items():
            if any(a in t for a in aliases):
                centers[key] = xmid
    return centers
```

---

## 13. Assign words in each row to columns

Once you have column centers, assign each word to the nearest semantic column.

```python
def assign_columns(row, col_centers):
    cols = {k: [] for k in col_centers}
    keys = list(col_centers)
    centers = [col_centers[k] for k in keys]

    for w in row:
        xmid = 0.5 * (w["x0"] + w["x1"])
        j = min(range(len(centers)), key=lambda i: abs(xmid - centers[i]))
        cols[keys[j]].append(w["text"])

    return {k: " ".join(v).strip() for k, v in cols.items()}
```

This is much more robust than trying to parse by whitespace alone.

---

## 14. Parse block header metadata

The top of each block contains metadata such as:

- molar mass,
- centered name,
- formula,
- possibly `[continued]`.

These are not always in identical positions, so parse them separately from the body.

### 14.1 Suggested heuristics

Inspect only the rows above the main table header.

Use heuristics such as:
- longest centered uppercase phrase → likely `name_raw`
- token that looks like chemical formula → likely `formula_raw`
- floating-point number near top → likely `molar_mass`
- presence of `[continued]` → continuation flag

### 14.2 Formula regex

```python
FORMULA_RE = re.compile(r"^[A-Z][A-Za-z0-9\[\]\-\+\(\)]*$")
```

That is intentionally permissive.

---

## 15. Continuation handling

If the block header contains `[continued]`, do **not** create a new entry automatically.

Instead, link it to the most recent open entry with matching normalized header identity.

Suggested key:

```python
entry_key = (
    formula_key,
    name_key,
    round(molar_mass or -1, 3),
)
```

If `is_continuation` is true and that key exists in the current parsing session, reuse the existing `entry_id` and extend `page_end`.

---

## 16. Distinguish normal data rows from transition rows

### 16.1 Normal thermochemical row
A row is a normal `datum` if it has:

- a valid temperature, and
- several numeric columns, and
- an active current phase.

### 16.2 Transition row
A row is likely a `transition` if:

- it appears between two phase sections,
- it has no temperature in the temperature column,
- it contains one or two standalone numeric values,
- its structure does not match the full data table.

Store these rows separately first, then interpret them in a cleanup pass.

Do not force them into the same schema as ordinary temperature rows.

---

## 17. Parse phases explicitly

Maintain `current_phase` while walking the main body rows.

When a row has a recognized phase token such as:

- `GAS`
- `LIQ`
- `SOL`
- `SOL-A`
- `SOL-B`

then either:
- create a new phase record if not seen before,
- or switch the current phase pointer.

Suggested phase regex:

```python
PHASE_RE = re.compile(r"^(GAS|LIQ|SOL(?:-[A-Z0-9]+)?)$")
```

You may expand this later if other labels appear.

---

## 18. Parse references subsection separately

Once `References` is reached, switch parser mode.

The rows underneath do not follow the main thermochemical schema.

Typical fields are:

- `phase_label`
- `hs_ref`
- `cp_ref`
- `remarks`

At first, it is perfectly fine to store the full row as `raw_row_text` and only weakly parse the fields.

You can tighten this later.

---

## 19. Numeric conversion

Be conservative with numeric parsing.

```python
def to_float(s):
    if s is None:
        return None
    s = s.strip().replace("−", "-").replace("—", "-")
    s = s.replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None
```

Keep raw text around even when conversion succeeds.

---

## 20. Validation and QA

You will need this. Even with a machine-readable PDF, some rows will parse badly.

### 20.1 Structural checks
Flag issues when:

- no header row is found,
- no references boundary is found where expected,
- too few columns are detected,
- formula or name are missing,
- continuation block cannot be linked.

### 20.2 Thermodynamic sanity checks
For each phase:

- `T` should be strictly increasing
- `H-H298` should usually increase with `T`
- `S` should usually increase with `T`
- `Cp` should vary smoothly

### 20.3 Derivative consistency check
A very strong check is:

\[
1000 \cdot \frac{\Delta(H-H_{298})}{\Delta T} \approx C_p
\]

Use it as an anomaly detector, not as an exact identity.

### 20.4 Continuity checks
Within each phase:
- no duplicated temperature rows,
- no abrupt impossible jumps,
- values should not randomly disappear in the middle of a block.

### 20.5 Store issues in the database
Do not just print warnings.
Store them in `parse_issue`.

---

## 21. Minimal end-to-end structure

A reasonable implementation order is:

1. open PDF
2. insert document metadata
3. iterate pages
4. crop and dedupe page
5. split into blocks
6. extract words with coordinates
7. persist raw words
8. reconstruct rows
9. detect header and references
10. parse block header
11. resolve or create entry
12. parse body rows
13. parse transitions
14. parse references
15. run a second validation pass

---

## 22. Starter implementation skeleton

### 22.1 Setup

```python
import re
import sqlite3
from pathlib import Path

import pdfplumber
```

### 22.2 Database init

```python
def init_db(conn):
    conn.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS entry (
        id INTEGER PRIMARY KEY,
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

    CREATE TABLE IF NOT EXISTS phase (
        id INTEGER PRIMARY KEY,
        entry_id INTEGER NOT NULL,
        phase_label TEXT NOT NULL,
        phase_order INTEGER NOT NULL,
        UNIQUE(entry_id, phase_label, phase_order)
    );

    CREATE TABLE IF NOT EXISTS datum (
        id INTEGER PRIMARY KEY,
        phase_id INTEGER NOT NULL,
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
        entry_id INTEGER NOT NULL,
        page_no INTEGER NOT NULL,
        row_no INTEGER NOT NULL,
        from_phase_label TEXT,
        to_phase_label TEXT,
        T_K REAL,
        dH_transition_kJ_mol REAL,
        dS_transition_J_mol REAL,
        note TEXT,
        raw_row_text TEXT
    );

    CREATE TABLE IF NOT EXISTS reference_row (
        id INTEGER PRIMARY KEY,
        entry_id INTEGER NOT NULL,
        page_no INTEGER NOT NULL,
        phase_label TEXT,
        hs_ref TEXT,
        cp_ref TEXT,
        remarks TEXT,
        raw_row_text TEXT
    );

    CREATE TABLE IF NOT EXISTS parse_issue (
        id INTEGER PRIMARY KEY,
        page_no INTEGER,
        entry_id INTEGER,
        severity TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        message TEXT NOT NULL,
        raw_context TEXT
    );
    """)
    conn.commit()
```

### 22.3 Split page into blocks

```python
def find_long_horizontal_splits(page, min_frac=0.65):
    min_len = page.width * min_frac
    ys = []
    for line in page.lines:
        if abs(line["top"] - line["bottom"]) < 1.5:
            if (line["x1"] - line["x0"]) >= min_len:
                ys.append(line["top"])
    ys = sorted(set(round(y, 1) for y in ys))
    return ys


def split_into_blocks(page):
    cropped = page.crop((20, 20, page.width - 35, page.height - 20))
    ys = find_long_horizontal_splits(cropped)

    if not ys:
        return [(0, cropped)]

    bounds = [20] + ys + [cropped.height - 20]
    blocks = []
    for i in range(len(bounds) - 1):
        y0, y1 = bounds[i], bounds[i + 1]
        if y1 - y0 < 80:
            continue
        block = cropped.crop((20, y0, cropped.width - 20, y1))
        blocks.append((len(blocks), block))
    return blocks
```

### 22.4 Row reconstruction and body parsing

```python
def parse_pdf_to_sqlite(pdf_path: str, db_path: str):
    conn = sqlite3.connect(db_path)
    init_db(conn)

    open_entries = {}

    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            page = page.dedupe_chars()

            for block_index, block in split_into_blocks(page):
                words = block.extract_words(
                    x_tolerance=2,
                    y_tolerance=2,
                    use_text_flow=False,
                    keep_blank_chars=False,
                    extra_attrs=["fontname", "size"],
                    return_chars=True,
                )

                rows = cluster_rows(words, y_tol=2.2)
                if not rows:
                    continue

                header_idx, refs_idx = detect_header_and_body(rows)
                if header_idx is None:
                    conn.execute(
                        "INSERT INTO parse_issue(page_no, severity, issue_type, message) VALUES (?, ?, ?, ?)",
                        (pageno, "warning", "missing_header", "Could not find main table header")
                    )
                    continue

                top_rows = rows[:header_idx]
                top_text = " | ".join(row_text(r) for r in top_rows)

                # crude first pass: refine later
                formula = None
                molar_mass = None
                name = None
                is_cont = "[continued]" in top_text

                for r in top_rows:
                    txt = row_text(r)
                    parts = txt.split()

                    for part in parts:
                        mm = to_float(part)
                        if molar_mass is None and mm is not None and mm > 1:
                            molar_mass = mm

                    if name is None and len(txt) > 5 and txt.upper() == txt:
                        if "Phase" not in txt and "KI" not in txt:
                            name = txt

                    for part in parts:
                        if FORMULA_RE.match(part) and any(ch.isalpha() for ch in part):
                            if part not in {"Phase", "References"}:
                                formula = part

                if name is None:
                    name = top_text

                formula_key = normalize_formula(formula)
                name_key = normalize_name(name)
                entry_key = (formula_key, name_key, round(molar_mass or -1, 3))

                if is_cont and entry_key in open_entries:
                    entry_id = open_entries[entry_key]
                    conn.execute("UPDATE entry SET page_end=? WHERE id=?", (pageno, entry_id))
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO entry(formula_raw, formula_key, name_raw, name_key, molar_mass,
                                          page_start, page_end, is_gas_header, raw_header)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            formula,
                            formula_key,
                            name,
                            name_key,
                            molar_mass,
                            pageno,
                            pageno,
                            int("(GAS)" in name.upper() or (formula or "").endswith("[g]")),
                            top_text,
                        ),
                    )
                    entry_id = cur.lastrowid
                    open_entries[entry_key] = entry_id

                col_centers = infer_column_centers(rows[header_idx])
                body_rows = rows[header_idx + 1 : refs_idx if refs_idx is not None else len(rows)]

                current_phase = None
                phase_order = 0
                phase_ids = {}

                for row_no, r in enumerate(body_rows, start=1):
                    txt = row_text(r)
                    if not txt:
                        continue

                    assigned = assign_columns(r, col_centers)
                    phase_txt = assigned.get("phase", "").strip()

                    if PHASE_RE.match(phase_txt):
                        current_phase = phase_txt
                        if current_phase not in phase_ids:
                            phase_order += 1
                            cur = conn.execute(
                                "INSERT INTO phase(entry_id, phase_label, phase_order) VALUES (?, ?, ?)",
                                (entry_id, current_phase, phase_order),
                            )
                            phase_ids[current_phase] = cur.lastrowid

                    T = to_float(assigned.get("T_K"))
                    numeric_count = sum(
                        to_float(assigned.get(k)) is not None
                        for k in col_centers
                        if k != "phase"
                    )

                    if T is not None and numeric_count >= 3 and current_phase is not None:
                        conn.execute(
                            """
                            INSERT INTO datum(
                                phase_id, page_no, row_no, T_K, Cp_J_molK, S_J_molK,
                                minus_G_minus_H298_over_T_J_molK, H_kJ_mol, H_minus_H298_kJ_mol,
                                G_kJ_mol, dHf_kJ_mol, dGf_kJ_mol, logKf, raw_row_text
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                phase_ids[current_phase],
                                pageno,
                                row_no,
                                T,
                                to_float(assigned.get("Cp_J_molK")),
                                to_float(assigned.get("S_J_molK")),
                                to_float(assigned.get("minus_G_minus_H298_over_T_J_molK")),
                                to_float(assigned.get("H_kJ_mol")),
                                to_float(assigned.get("H_minus_H298_kJ_mol")),
                                to_float(assigned.get("G_kJ_mol")),
                                to_float(assigned.get("dHf_kJ_mol")),
                                to_float(assigned.get("dGf_kJ_mol")),
                                to_float(assigned.get("logKf")),
                                txt,
                            ),
                        )
                    else:
                        vals = [to_float(v) for v in assigned.values()]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            conn.execute(
                                """
                                INSERT INTO transition(
                                    entry_id, page_no, row_no, from_phase_label, raw_row_text
                                ) VALUES (?, ?, ?, ?, ?)
                                """,
                                (entry_id, pageno, row_no, current_phase, txt),
                            )

                if refs_idx is not None:
                    for rr in rows[refs_idx + 1 :]:
                        txt = row_text(rr)
                        if not txt:
                            continue
                        parts = txt.split()
                        phase_label = parts[0] if parts else None
                        conn.execute(
                            "INSERT INTO reference_row(entry_id, page_no, phase_label, raw_row_text) VALUES (?, ?, ?, ?)",
                            (entry_id, pageno, phase_label, txt),
                        )

        conn.commit()
    conn.close()
```

This is deliberately a first-pass parser, not the final polished version.

---

## 23. Second-pass cleanup

After the first pass, run a second pass that:

- fills `to_phase_label` in transitions,
- merges entries across continuation pages,
- normalizes formula spacing,
- detects duplicate rows,
- checks monotonic temperature,
- checks rough `Cp` / `ΔH` consistency,
- populates `parse_issue` with anomalies.

This second pass is not optional.

---

## 24. Recommended development order

### Phase 1: prove geometry works
Use 5–10 pages and verify:
- block splitting
- row clustering
- header detection
- column inference

### Phase 2: parse one family well
Pick one chemical family, such as the Hf pages, and make sure:
- multi-phase entries work,
- continuation works,
- multiple entries per page work,
- gas entries stay distinct.

### Phase 3: persist everything
Write all raw words and parsed rows to SQLite.

### Phase 4: build validators
Add `parse_issue` generation.

### Phase 5: scale to full document
Run the whole book and inspect only flagged entries.

---

## 25. Practical advice

### 25.1 Don’t overfit too early
First build something that parses 80–90% of rows and stores enough provenance to fix the rest.

### 25.2 Never throw away raw context
Keep:
- raw header text,
- raw row text,
- raw words with coordinates.

### 25.3 Treat the PDF as semi-structured geometry
Do not think of this as a plain text parsing task.

### 25.4 Expect a repair loop
The first pass will not be perfect. That is normal.

---

## 26. Final recommendation

The right approach for Barin is:

- **pdfplumber** for geometric text extraction,
- **SQLite** for durable normalized storage,
- **two-layer architecture** for provenance and repair,
- **second-pass validation** for quality control.

That is the path that scales to the whole book without becoming unmaintainable.
