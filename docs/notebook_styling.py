"""Notebook helper utilities for diachronic_analysis.ipynb."""

from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


def escape_html_special_chars(value: Any) -> str:
    """Escape angle brackets for HTML table rendering."""
    if isinstance(value, str):
        return value.replace("<", "&lt;").replace(">", "&gt;")
    return str(value)


def style_centered_table(df: pd.DataFrame):
    """Apply centered style and hide index."""
    return df.style.hide(axis="index").set_properties(**{"text-align": "center"})


def style_decade_stats_table(df: pd.DataFrame):
    """Style decade summary table with wider sample column."""
    return (
        df.style
        .hide(axis="index")
        .set_properties(
            subset=["Sample N-grams"],
            **{"min-width": "560px", "white-space": "normal", "text-align": "left"},
        )
        .set_properties(subset=["Decade"], **{"min-width": "90px"})
    )


def style_vocab_table(
    df: pd.DataFrame,
    escape_cols: Optional[List[str]] = None,
    column_widths: Optional[Dict[str, str]] = None,
):
    """Style vocab table with escaped token strings and configurable widths."""
    df_escaped = df.copy()

    if escape_cols is None:
        escape_cols = df_escaped.select_dtypes(include=["object"]).columns.tolist()
    for col in escape_cols:
        if col in df_escaped.columns:
            df_escaped[col] = df_escaped[col].apply(escape_html_special_chars)

    if column_widths is None:
        column_widths = {}

    table_styles = [{"selector": "th.col_heading", "props": "text-align: center;"}]
    for col_key, width in column_widths.items():
        table_styles.append({
            "selector": f"th.{col_key}, td.{col_key}",
            "props": f"min-width: {width};",
        })

    return (
        df_escaped.style
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
        .set_table_styles(table_styles, overwrite=False)
    )


def style_coverage_table(df: pd.DataFrame):
    """Style vocabulary coverage table."""
    table_styles = [
        {"selector": "th.col_heading", "props": "text-align: center;"},
        {"selector": "th.col0, td.col0", "props": "min-width: 90px;"},
        {"selector": "th.col1, td.col1", "props": "min-width: 130px;"},
        {"selector": "th.col2, td.col2, th.col3, td.col3", "props": "min-width: 120px;"},
    ]

    return (
        df.style
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
        .set_table_styles(table_styles, overwrite=False)
    )


def collect_changed_preprocessing_rows(
    raw_file: Union[str, Path],
    processed_dir: Union[str, Path],
    ngram_type: int,
    get_decade_from_year: Callable[[int], str],
    normalize_text: Callable[[str], str],
    tokenize_and_clean: Callable[[str], List[str]],
    max_scan_lines: int = 1_500_000,
    wanted_kept: int = 8,
    wanted_discarded: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Collect representative rows where preprocessing alters the source n-gram."""
    raw_file = Path(raw_file)
    processed_dir = Path(processed_dir)

    processed_cache: Dict[str, set] = {}

    def in_processed_decade(decade: str, ngram: str) -> bool:
        if decade not in processed_cache:
            decade_path = processed_dir / f"{decade}.txt"
            if decade_path.exists():
                with open(decade_path, "r", encoding="utf-8") as f:
                    processed_cache[decade] = {line.strip() for line in f if line.strip()}
            else:
                processed_cache[decade] = set()
        return ngram in processed_cache[decade]

    kept_rows: List[Dict[str, Any]] = []
    discarded_rows: List[Dict[str, Any]] = []
    seen_kept = set()
    seen_discarded = set()
    changed_rows_scanned = 0

    with open(raw_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            raw_ngram = parts[0]
            year = int(parts[1])
            decade = get_decade_from_year(year)

            normalized = normalize_text(raw_ngram)
            normalized_tokens = re.findall(r"\b[\w]+\b", normalized)
            cleaned_tokens = tokenize_and_clean(raw_ngram)

            if raw_ngram == normalized and normalized_tokens == cleaned_tokens:
                continue

            changed_rows_scanned += 1

            if len(cleaned_tokens) < ngram_type and len(discarded_rows) < wanted_discarded:
                processed_ngram = " ".join(cleaned_tokens) if cleaned_tokens else "[filtered]"
                if processed_ngram not in seen_discarded:
                    seen_discarded.add(processed_ngram)
                    discarded_rows.append({
                        "Year": year,
                        "Raw N-gram": raw_ngram,
                        "Processed N-gram": processed_ngram,
                        "Outcome": f"Discarded (<{ngram_type} tokens)",
                        "Result": "<span style='color:#c62828;font-weight:700;'>X</span>",
                    })
            elif len(cleaned_tokens) >= ngram_type and len(kept_rows) < wanted_kept:
                processed_ngram = " ".join(cleaned_tokens[:ngram_type])
                if in_processed_decade(decade, processed_ngram) and processed_ngram not in seen_kept:
                    seen_kept.add(processed_ngram)
                    kept_rows.append({
                        "Year": year,
                        "Raw N-gram": raw_ngram,
                        "Processed N-gram": processed_ngram,
                        "Outcome": "Kept",
                        "Result": "<span style='color:#2e7d32;font-weight:700;'>OK</span>",
                    })

            if len(kept_rows) >= wanted_kept and len(discarded_rows) >= wanted_discarded:
                break

    return kept_rows, discarded_rows, changed_rows_scanned


def build_changed_rows_dataframe(
    kept_rows: List[Dict[str, Any]], discarded_rows: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Build a sorted dataframe for changed row samples."""
    return (
        pd.DataFrame(kept_rows + discarded_rows)[
            ["Year", "Raw N-gram", "Processed N-gram", "Outcome", "Result"]
        ]
        .sort_values("Year")
        .reset_index(drop=True)
    )
