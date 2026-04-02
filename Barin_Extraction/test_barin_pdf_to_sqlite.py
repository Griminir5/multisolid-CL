from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pdfplumber

from Barin_Extraction.barin_pdf_to_sqlite import (
    cluster_rows,
    detect_blocks,
    extract_page_words,
    parse_pdf_to_sqlite,
    to_float,
)


HERE = Path(__file__).resolve().parent
EXCERPT_PDF = HERE / "excerpt.pdf"


class ParserHelperTests(unittest.TestCase):
    def test_extract_page_words_filters_vertical_watermark(self) -> None:
        with pdfplumber.open(EXCERPT_PDF) as pdf:
            words = extract_page_words(pdf.pages[1])
        texts = {word["text"] for word in words}
        self.assertNotIn("Downloaded", texts)
        self.assertTrue(all(word.get("upright", True) for word in words))

    def test_cluster_rows_and_block_detection_match_excerpt_layout(self) -> None:
        with pdfplumber.open(EXCERPT_PDF) as pdf:
            words = extract_page_words(pdf.pages[4])
        rows = cluster_rows(5, words)
        blocks = detect_blocks(rows)
        self.assertEqual(2, len(blocks))
        self.assertEqual(19, blocks[0].refs_start_row_no)
        self.assertEqual(31, blocks[1].refs_start_row_no)
        texts = [row.text for row in rows]
        self.assertIn("HfBr4 HAFNIUM TETRABROMIDE 498.1 06", texts)

    def test_to_float_handles_split_numeric_tokens(self) -> None:
        self.assertEqual(1100.0, to_float("1 100.00"))
        self.assertEqual(-716.028, to_float("- 71 6.028"))


class ExcerptIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tempdir.name) / "excerpt.sqlite"
        cls.summary = parse_pdf_to_sqlite(EXCERPT_PDF, cls.db_path, replace=True)
        cls.conn = sqlite3.connect(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()
        cls.tempdir.cleanup()

    def test_excerpt_counts(self) -> None:
        self.assertEqual(7, self.summary["entry_block"])
        self.assertEqual(6, self.summary["entry"])
        self.assertEqual(8, self.summary["phase"])
        self.assertEqual(158, self.summary["datum"])
        self.assertEqual(2, self.summary["transition"])
        self.assertEqual(8, self.summary["reference_row"])

    def test_continuation_entry_spans_pages_and_keeps_all_phases(self) -> None:
        row = self.conn.execute(
            """
            SELECT id, page_start, page_end
            FROM entry
            WHERE name_key = 'HAFNIUM' AND is_gas_header = 0
            """
        ).fetchone()
        self.assertIsNotNone(row)
        entry_id, page_start, page_end = row
        self.assertEqual((2, 3), (page_start, page_end))
        phases = {
            phase_label
            for (phase_label,) in self.conn.execute(
                "SELECT phase_label FROM phase WHERE entry_id = ? ORDER BY phase_order",
                (entry_id,),
            )
        }
        self.assertEqual({"SOL-A", "SOL-B", "LIQ"}, phases)

    def test_gas_and_condensed_entries_stay_distinct(self) -> None:
        hafnium_variants = self.conn.execute(
            "SELECT COUNT(1) FROM entry WHERE name_key = 'HAFNIUM'"
        ).fetchone()[0]
        self.assertEqual(2, hafnium_variants)

        tetrabromide_variants = self.conn.execute(
            """
            SELECT COUNT(1)
            FROM entry
            WHERE molar_mass = 498.106
            """
        ).fetchone()[0]
        self.assertEqual(2, tetrabromide_variants)

    def test_temperatures_increase_within_each_phase_for_non_null_rows(self) -> None:
        rows = self.conn.execute(
            """
            SELECT p.id, d.T_K
            FROM phase p
            JOIN datum d ON d.phase_id = p.id
            WHERE d.T_K IS NOT NULL
            ORDER BY p.id, d.T_K
            """
        ).fetchall()
        last_by_phase: dict[int, float] = {}
        for phase_id, temperature in rows:
            previous = last_by_phase.get(phase_id)
            if previous is not None:
                self.assertGreater(temperature, previous)
            last_by_phase[phase_id] = temperature

    def test_parse_issues_capture_low_confidence_and_malformed_rows(self) -> None:
        issue_types = {
            issue_type
            for (issue_type,) in self.conn.execute("SELECT issue_type FROM parse_issue")
        }
        self.assertIn("low_confidence_formula", issue_types)
        self.assertIn("malformed_temperature", issue_types)


if __name__ == "__main__":
    unittest.main()
