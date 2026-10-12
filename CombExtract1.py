"""
================================================================================
END-TO-END WEB SCRAPING & ALGORITHMIC NEWS HEADLINE EXTRACTION PIPELINE
================================================================================
Pipeline Overview:
- Stage 1 (HTML Normalization & Noise Removal):

      Takes raw downloaded HTML pages or browser view-source dumps, removes
      scripts, styles, and structural clutter, and extracts clean text into
      structured datasets.
- Stage 2 (Heuristic & NLP News Headline Refiner):

      Ingests the cleaned structured data and isolates genuine news headlines
      using heuristic patterns and syntactic grammar rules (spaCy POS tagging).

Key Prerequisites:
    pip install beautifulsoup4 pandas openpyxl spacy
    python -m spacy download en_core_web_sm
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS & THIRD-PARTY DEPENDENCIES
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Loads all standard Python utilities and third-party libraries required for
#   HTML parsing, file management, data manipulation, and natural language processing.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Runs first at script startup so all subsequent data cleaning, file handling,
#   and linguistic analysis functions have immediate access to their dependencies.
#
# KEY ITEMS INITIALIZED / IMPORTED:
#   - 'BeautifulSoup' & 'NavigableString' (bs4): Parse and decompose HTML markup.
#   - 'pandas': Builds, manipulates, and exports tabular data frames.
#   - 'spacy': Provides syntactic parsing and Part-of-Speech (POS) tagging.
#   - 'Path' (pathlib): Handles operating-system-independent folder and file paths.
# ==============================================================================

import html
import re
from pathlib import Path
from typing import Optional, Union

from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import spacy

# ==============================================================================
# SECTION 2: CONFIGURATION, PATHS & RUNTIME TOGGLES
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Centralizes all operational settings, directory locations, processing toggles,
#   and extraction rules in one easily editable location.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Provides the global parameters used by the batch processor in Section 5 to
#   locate source files, decide which formats to generate, and determine whether
#   Stage 2 runs automatically.
#
# KEY ITEMS CONFIGURED:
#   - 'INPUT_HTML_DIR' / 'OUTPUT_DATA_DIR': Folder paths for source HTML files
#     and output tables.
#   - 'PROCESS_ALL_FILES': Boolean switch. Set to True to scan and batch-process
#     every HTML file in the input directory; set to False to limit execution to
#     the specific files listed in 'INPUT_FILES'.
#   - 'RUN_STAGE_2_FILTER': Boolean switch. Set to True to immediately send Stage 1
#     output into the Stage 2 headline filter.
#   - 'OUTPUT_EXCEL_FILENAME': Name of the consolidated multi-tab Excel workbook
#     ('cleaned_output_summary2.xlsx') populated with cleaned data.
#   - 'REFINED_EXCEL_FILENAME': Name of the consolidated multi-tab Excel workbook
#     ('refined_output_summary2.xlsx') populated with refined headlines.
#   - 'HEADLINE_TAG_WHITELIST': Permitted HTML tag types (e.g., 'h1', 'p', 'span')
#     inspected for potential headlines.
# ==============================================================================

# --- DIRECTORIES ---
INPUT_HTML_DIR = Path(r"C:\Stevedrive\BusNewsAn\InputhtmlSources")
OUTPUT_DATA_DIR = Path(r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM")

# --- USER TOGGLES ---
# True: Process all HTML files in INPUT_HTML_DIR | False: Process only INPUT_FILES list
PROCESS_ALL_FILES = True

# Specific files used when PROCESS_ALL_FILES is False
INPUT_FILES = [
    INPUT_HTML_DIR / "view-source_https___www.afr.com_companies.html",
]

# Output format mode: True for plain text/CSV (lowest LLM token cost), False for simplified HTML
TEXT_ONLY_MODE = True

# Automatic pipeline hand-off: Run Stage 2 headline filtering immediately after Stage 1
RUN_STAGE_2_FILTER = True

# --- OUTPUT & NLP CONFIGURATION ---
HTML_EXTENSIONS = (".html", ".htm")
OUTPUT_EXCEL_FILENAME = "cleaned_output_summary2.xlsx"
REFINED_EXCEL_FILENAME = "refined_output_summary2.xlsx"
SPACY_MODEL_NAME = "en_core_web_sm"

# Permitted HTML container and heading tags for candidate news headlines
HEADLINE_TAG_WHITELIST = {"span", "section", "h1", "h2", "h3", "h4", "p", "article", "a"}


# ==============================================================================
# SECTION 3: STAGE 1 — HTML CLEANING & STRUCTURE EXTRACTOR
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Takes raw web page text, decodes browser-specific source dumps back into valid
#   HTML, removes noise (scripts, styles, tracking tags, and menus), and extracts
#   clean textual elements.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Serves as the initial data intake step. It cleans messy web data so downstream
#   processes receive only readable text without markup bloat or layout code.
#
# KEY ITEMS GENERATED & POPULATED:
#   - 'unescape_chrome_source()': Detects table-wrapped Chrome 'view-source' lines
#     ('<td class="line-content">') and unpacks them into standard HTML strings.
#   - 'clean_html_for_llm()': Strips boilerplate tags ('script', 'style', 'nav',
#     'header', 'footer') and returns two items:
#       1. A continuous block of clean text suitable for LLM prompts.
#       2. 'structured_rows': A list of paired values in (Tag, Content) format
#          (e.g., ('h1', 'Market Closes Up 2%')) used to build data frames.
# ==============================================================================

def unescape_chrome_source(raw_content: str) -> str:
    """Normalizes DevTools 'view-source' HTML wraps back to raw HTML markup."""
    soup = BeautifulSoup(raw_content, "html.parser")
    line_cells = soup.find_all("td", class_="line-content")

    # Reassemble table lines if Chrome source-view wrapper is detected
    raw_markup = "\n".join(td.get_text() for td in line_cells) if line_cells else soup.get_text()
    return html.unescape(raw_markup)


def clean_html_for_llm(html_content: str, extract_text_only: bool = True):
    """Strips boilerplate tags/attributes and extracts structured tag-content pairs."""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Remove non-content, functional, and structural boilerplate
    noise_tags = [
        "script", "style", "noscript", "svg", "iframe",
        "header", "footer", "nav", "form", "head"
    ]
    for tag in soup(noise_tags):
        tag.decompose()

    # 2. Retain only semantic attributes if maintaining HTML layout
    allowed_attrs = {"href", "src", "alt", "title", "datetime"}
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k.lower() in allowed_attrs}

    # 3. Extract direct textual leaves to avoid duplicated ancestor content
    structured_rows = []
    for element in soup.find_all(True):
        direct_text = "".join(
            child for child in element.children if isinstance(child, NavigableString)
        ).strip()
        if direct_text:
            structured_rows.append((element.name.lower(), direct_text))

    if extract_text_only:
        text = soup.get_text(separator="\n")
        cleaned_text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return cleaned_text, structured_rows

    clean_html = re.sub(r"\n\s*\n+", "\n", soup.prettify()).strip()
    return clean_html, structured_rows


# ==============================================================================
# SECTION 4: STAGE 2 — HEURISTIC & NLP HEADLINE FILTER
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Filters text down to genuine news headlines by weeding out fragments, legal
#   notices, timestamps, and navigation labels, then verifying that remaining
#   phrases follow natural headline grammar.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Acts as the semantic refinement layer directly following Stage 1. It operates
#   either on in-memory tables passed from Section 5 or on standalone CSV files
#   saved from earlier runs.
#
# KEY ITEMS GENERATED & POPULATED:
#   - 'HeuristicNewsFilter': An NLP engine powered by the 'en_core_web_sm' spaCy model.
#   - Shape & Pattern Filter: Rejects short phrases (< 3 words or < 15 characters),
#     clock timestamps (e.g., '14:30'), and legal lines ('all rights reserved').
#   - Syntactic Grammar Validator: Checks for active verbs/auxiliaries ('VERB', 'AUX')
#     or substantive title structures containing prepositions and connectives.
#   - Refined Output Data: Produces filtered tables containing three standardized
#     columns: ['Item_No', 'Tag', 'Content'].
# ==============================================================================

class HeuristicNewsFilter:
    """Isolates genuine news headlines using pattern heuristics and spaCy POS analysis."""

    def __init__(self, spacy_model: str = SPACY_MODEL_NAME):
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner"])
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model, disable=["ner"])

    @staticmethod
    def _is_boilerplate_shape(text: str) -> bool:
        """Identifies durations, pure symbols/dates, short phrases, and legal notices."""
        t = text.strip()

        # Timestamps / durations (e.g., '12:45', '1:02:30')
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t):
            return True

        # Pure numbers, dates, or non-word symbols
        if re.fullmatch(r"[\d\s\W_]+", t):
            return True

        # Legal disclaimers and navigation boilerplate
        boilerplate_pattern = r"[©®™]|\b(all rights reserved|terms of service|privacy policy|cookie settings|sign in|subscribe)\b"
        if re.search(boilerplate_pattern, t, re.IGNORECASE):
            return True

        # Valid headlines require minimum length (>= 3 words and >= 15 characters)
        words = t.split()
        return len(words) < 3 or len(t) < 15

    @staticmethod
    def _has_headline_grammar(doc: spacy.tokens.Doc) -> bool:
        """Determines if the parsed token sequence represents headline syntax."""
        # 1. Verbal elements (finite verbs, auxiliaries, or participles)
        if any(token.pos_ in ("VERB", "AUX") for token in doc):
            return True

        # 2. Substantive noun phrases (>= 4 words containing clausal/prepositional depth)
        words = doc.text.split()
        if len(words) >= 4:
            connective_tokens = {":", "—", "-", "’s", "'s"}
            if any(token.pos_ in ("ADP", "CCONJ", "SCONJ") or token.text in connective_tokens for token in doc):
                return True

        return False

    def refine_dataframe(
            self,
            df: pd.DataFrame,
            content_col: str = "Content",
            tag_col: str = "Tag",
            tag_whitelist: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Filters an in-memory DataFrame down to candidate headlines."""
        if df.empty:
            return pd.DataFrame(columns=["Item_No", tag_col, content_col])

        df_work = df.copy()
        df_work.columns = [c.strip().capitalize() for c in df_work.columns]
        content_col = content_col.capitalize()
        tag_col = tag_col.capitalize()

        # Standardize strings
        df_work[content_col] = df_work[content_col].astype(str).str.strip().str.strip("'\"")
        df_work[tag_col] = df_work[tag_col].astype(str).str.strip().str.lower()

        # Filter by allowed HTML tags
        whitelist = tag_whitelist or HEADLINE_TAG_WHITELIST
        df_work = df_work[df_work[tag_col].isin(whitelist)]

        # Filter boilerplate shape and drop duplicates
        df_work = df_work[~df_work[content_col].apply(self._is_boilerplate_shape)]
        df_work = df_work.drop_duplicates(subset=[content_col])

        if df_work.empty:
            return pd.DataFrame(columns=["Item_No", tag_col, content_col])

        # Batch NLP parsing
        docs = list(self.nlp.pipe(df_work[content_col].tolist(), batch_size=64))
        valid_mask = [self._has_headline_grammar(doc) for doc in docs]

        refined_df = df_work[valid_mask].copy().reset_index(drop=True)
        refined_df.insert(0, "Item_No", refined_df.index + 1)

        return refined_df[["Item_No", tag_col, content_col]]

    def refine(
            self,
            file_path: Union[str, Path],
            content_col: str = "Content",
            tag_col: str = "Tag",
            tag_whitelist: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Reads a cleaned CSV directly and applies the headline filter."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return self.refine_dataframe(pd.read_csv(path), content_col, tag_col, tag_whitelist)


# ==============================================================================
# SECTION 5: PIPELINE ORCHESTRATION & FILE EXPORT
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Controls the end-to-end processing loop across all target files. It reads raw
#   HTML, executes Stage 1 cleaning, records audit entries in multi-sheet Excel
#   workbooks (for both cleaned data and refined headlines), and hands data
#   directly to Stage 2 for headline extraction.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Acts as the operational engine that connects Stage 1 and Stage 2, coordinating
#   reading, cleaning, analyzing, and writing data to disk.
#
# KEY OUTPUT FILES CREATED & POPULATED:
#   - Intermediate Cleaned CSVs: Populates files named '<source_name>_cleaned.csv'
#     containing two columns: 'Tag' and 'Content'.
#   - Consolidated Cleaned Excel Workbook: Populates 'cleaned_output_summary2.xlsx' with a
#     dedicated sheet ('C1', 'C2', ...) per processed source file. Cell A1 records
#     the source file name, and table data begins at Row 3.
#   - Final Refined News CSVs: When Stage 2 is enabled, populates files named
#     '<source_name>_refined.csv' containing candidate headlines organized by
#     'Item_No', 'Tag', and 'Content'.
#   - Consolidated Refined Excel Workbook: Populates 'refined_output_summary2.xlsx' with a
#     dedicated sheet ('R1', 'R2', ...) per processed source file matching the layout of
#     the cleaned workbook.
# ==============================================================================

def process_pipeline(
        input_dir: Path = INPUT_HTML_DIR,
        output_dir: Path = OUTPUT_DATA_DIR,
        process_all: bool = PROCESS_ALL_FILES,
        file_list: Optional[list] = None,
        text_only: bool = TEXT_ONLY_MODE,
        run_stage2: bool = RUN_STAGE_2_FILTER,
):
    """Executes Stage 1 HTML sanitization and optional Stage 2 headline extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaner = HeuristicNewsFilter() if run_stage2 else None

    # Determine files to process
    if process_all:
        target_files = [f for f in input_dir.iterdir() if f.suffix.lower() in HTML_EXTENSIONS]
        print(f"[Mode: ALL FILES] Scanning '{input_dir.resolve()}' ({len(target_files)} found)")
    else:
        target_files = [Path(f) for f in (file_list or INPUT_FILES)]
        print(f"[Mode: SPECIFIED LIST] Selected {len(target_files)} file(s)")

    if not target_files:
        print("[!] No target files found.")
        return

    excel_cleaned_path = output_dir / OUTPUT_EXCEL_FILENAME
    excel_refined_path = output_dir / REFINED_EXCEL_FILENAME

    with pd.ExcelWriter(excel_cleaned_path, engine="openpyxl") as clean_writer:
        refined_writer = (
            pd.ExcelWriter(excel_refined_path, engine="openpyxl")
            if (cleaner and text_only)
            else None
        )

        try:
            for tab_idx, file_path in enumerate(target_files, start=1):
                if not file_path.exists():
                    print(f"[✘] Missing: {file_path.name}")
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_data = f.read()

                    # Stage 1: Clean HTML & extract tag-content rows
                    normalized_html = unescape_chrome_source(raw_data)
                    cleaned_text, structured_rows = clean_html_for_llm(normalized_html, extract_text_only=text_only)
                    df_clean = pd.DataFrame(structured_rows, columns=["Tag", "Content"])

                    # Output Stage 1 CSV / HTML
                    output_ext = ".csv" if text_only else file_path.suffix
                    output_file = output_dir / f"{file_path.stem}_cleaned{output_ext}"
                    is_overwrite = output_file.exists()

                    if text_only:
                        df_clean.to_csv(output_file, index=False, encoding="utf-8")
                    else:
                        output_file.write_text(cleaned_text, encoding="utf-8")

                    # Append to Cleaned Excel summary sheet (C1, C2, ...)
                    sheet_name = f"C{tab_idx}"
                    pd.DataFrame([f"Source: {file_path.name}"]).to_excel(
                        clean_writer, sheet_name=sheet_name, startrow=0, startcol=0, header=False, index=False
                    )
                    df_clean.to_excel(clean_writer, sheet_name=sheet_name, startrow=2, index=False)

                    status = "[⇄ Overwrite]" if is_overwrite else "[✔ Created]"
                    reduction = ((len(raw_data) - len(cleaned_text)) / len(raw_data) * 100) if raw_data else 0
                    print(f"{status} {file_path.name} -> {output_file.name} (Size reduction: {reduction:.1f}%)")

                    # Stage 2: In-memory headline refinement
                    if cleaner and text_only:
                        refined_df = cleaner.refine_dataframe(df_clean)
                        refined_file = output_dir / f"{file_path.stem}_refined.csv"
                        refined_df.to_csv(refined_file, index=False, encoding="utf-8")

                        # Append to Refined Excel summary sheet (R1, R2, ...)
                        if refined_writer:
                            refined_sheet_name = f"R{tab_idx}"
                            pd.DataFrame([f"Source: {file_path.name}"]).to_excel(
                                refined_writer, sheet_name=refined_sheet_name, startrow=0, startcol=0, header=False, index=False
                            )
                            refined_df.to_excel(refined_writer, sheet_name=refined_sheet_name, startrow=2, index=False)

                        print(f"    ↳ [Stage 2] {len(refined_df)} headlines saved -> {refined_file.name}")

                except Exception as e:
                    print(f"[✘] Error processing {file_path.name}: {e}")
        finally:
            if refined_writer:
                refined_writer.close()


# ==============================================================================
# SECTION 6: MAIN EXECUTION ENTRY POINT
# ==============================================================================
# WHAT THIS SECTION ACHIEVES:
#   Provides the primary starting gate when running this script directly from
#   the terminal or an IDE.
#
# HOW IT FITS INTO THE PROCESS FLOW:
#   Protects the pipeline code from running accidentally when imported into other
#   scripts as a module, triggering 'process_pipeline()' only when launched
#   as the main program.
#
# KEY ACTIONS EXECUTED:
#   - Calls 'process_pipeline()' using the configuration settings from Section 2.
#   - Coordinates the entire process: reading HTML files, cleaning them into
#     '_cleaned.csv' files, recording summary tabs in both Excel workbooks, and
#     exporting filtered headlines to '_refined.csv' files.
# ==============================================================================

if __name__ == "__main__":
    process_pipeline(
        input_dir=INPUT_HTML_DIR,
        output_dir=OUTPUT_DATA_DIR,
        process_all=PROCESS_ALL_FILES,
        file_list=INPUT_FILES,
        text_only=TEXT_ONLY_MODE,
        run_stage2=RUN_STAGE_2_FILTER,
    )