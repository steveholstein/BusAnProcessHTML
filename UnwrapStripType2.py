"""
Cleaning process to make web pages useful for LLM processing
operates in two generic stages:
Unwrap Browser Markup: Standardize the code back into raw HTML.
Boilerplate & Noise Removal:
Strip out scripts, styles, embedded SVGs, navigation links, and structural bloat
using BeautifulSoup, leaving only clean semantic markup or plain text.
NOTE that sample file for this code came from:
view-source_https___www.afr.com_companies30Jul26.html

pip install beautifulsoup4 pandas openpyxl
"""

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================

import html
import re
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import pandas as pd

# ==============================================================================
# SECTION 2: CONFIGURATION / PARAMETERS & TOGGLES (Edit these in PyCharm)
# ==============================================================================

# Input and Output Directories
INPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\InputhtmlSources"
OUTPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM"

# --- USER TOGGLES ---

# Mode Toggle for Input File Selection:
# True  -> Automatically process EVERY file of type .html (or .htm) inside INPUT_DIRECTORY
# False -> Process ONLY the explicitly specified files in the INPUT_FILES list below
# DIRECTLY INCORPORATED IN: Section 5 (Main execution -> process_file_list parameter 'process_all')
#                           Section 4 (Controls file selection branching in process_file_list)
#

PROCESS_ALL_FILES = True
#
# Specify individual files located inside INPUT_DIRECTORY (Used when PROCESS_ALL_FILES = False)
INPUT_FILES = [
    Path(INPUT_DIRECTORY) / "view-source_https___www.afr.com_companies.html",
    #Path(INPUT_DIRECTORY) / "another_page.html",
    #Path(INPUT_DIRECTORY) / "sample_article.html",
]
#
# Output format mode:
# True  -> Output clean plain text/markdown (Recommended for LLMs - lowest token cost)
# False -> Output simplified HTML structure (Stripped of CSS/JS/Nav/Ads)
# DIRECTLY INCORPORATED IN: Section 5 (Main execution -> process_file_list parameter 'text_only')
#                           Section 4 (Passed into clean_html_for_llm in Section 3)
TEXT_ONLY_MODE = True

# --- OTHER CONFIGURATION ---

# Allowed file extensions when scanning the input directory in PROCESS_ALL_FILES mode
HTML_EXTENSIONS = (".html", ".htm")

# ==============================================================================
# SECTION 3: CORE CLEANING & NORMALIZATION FUNCTIONS
# ==============================================================================

def unescape_chrome_source(raw_content: str) -> str:
    """Converts Chrome/DevTools source view HTML back into valid raw HTML."""
    soup = BeautifulSoup(raw_content, "html.parser")

    # If file contains table-based line content (Chrome view-source format)
    line_contents = soup.find_all("td", class_="line-content")
    if line_contents:
        raw_text_lines = [td.get_text() for td in line_contents]
        reassembled = "\n".join(raw_text_lines)
    else:
        reassembled = soup.get_text()

    # Unescape entities (e.g., convert &lt;div&gt; back to <div>)
    return html.unescape(reassembled)


def clean_html_for_llm(
    html_content: str, extract_text_only: bool = True
):
    """Generically strips noise (scripts, styles, SVGs, nav, headers, footers)

    from an HTML document to optimize context for LLMs.
    Returns clean text/html string AND structured tag-data pairs for Excel generation.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Remove non-content and boilerplate tags completely
    noise_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "header",
        "footer",
        "nav",
        "form",
        "head",
    ]
    for tag in soup(noise_tags):
        tag.decompose()

    # 2. Remove cosmetic attributes if retaining HTML layout
    for tag in soup.find_all(True):
        allowed_attrs = ["href", "src", "alt", "title", "datetime"]
        tag.attrs = {
            k: v for k, v in tag.attrs.items() if k.lower() in allowed_attrs
        }

    # Extract structured content for Excel export (Tag Name in Column A, Text in Column B)
    structured_rows = []
    for element in soup.find_all(True):
        # Only take direct non-empty text strings to prevent duplicate ancestor text
        direct_text = "".join(
            child for child in element.children if isinstance(child, NavigableString)
        ).strip()

        if direct_text:
            # HTML tag name acts as the consistent identifying class/type tag
            tag_name = element.name.lower()
            structured_rows.append((tag_name, direct_text))

    if extract_text_only:
        # Extract pure readable text line by line
        text = soup.get_text(separator="\n")
        # Collapse multiple blank lines
        cleaned_text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return cleaned_text, structured_rows

    # Return clean HTML structure without clutter
    clean_html = soup.prettify()
    return re.sub(r"\n\s*\n+", "\n", clean_html).strip(), structured_rows

# ==============================================================================
# SECTION 4: FILE BATCH PROCESSING PIPELINE
# ==============================================================================

def process_file_list(
    input_dir: str,
    output_dir: str,
    process_all: bool = True,
    file_list: list = None,
    text_only: bool = True,
    extensions: tuple = HTML_EXTENSIONS,
):
    """Processes either ALL HTML files in input_dir or an explicit list of file paths,
    and outputs each to a separate cleaned file in output_dir along with an Excel file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine input files based on the process_all toggle mode
    if process_all:
        target_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in extensions
        ]
        print(f"[Mode: ALL FILES] Scanning '{input_path.resolve()}' for HTML sources...")
    else:
        target_files = [Path(f) for f in (file_list or [])]
        print(f"[Mode: SPECIFIED LIST] Selected {len(target_files)} file(s) for processing.")

    if not target_files:
        print("[!] No valid target files found to process.")
        return

    print(
        f"Starting batch process for {len(target_files)} file(s). Output target: '{output_path.resolve()}'\n"
    )

    for file_path in target_files:
        if not file_path.exists():
            print(f"[✘] Skipping (File not found): {file_path.name}")
            continue

        # Construct separate output file path
        ext = ".txt" if text_only else file_path.suffix
        output_file = output_path / f"{file_path.stem}_cleaned{ext}"
        excel_file = output_path / f"{file_path.stem}_cleaned.xlsx"

        # Pre-check if the destination output file already exists prior to writing
        file_already_exists = output_file.exists()

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_data = f.read()

            # Step 1: Normalize source-view wrapper format
            normalized_html = unescape_chrome_source(raw_data)

            # Step 2: Strip HTML noise & extract structured data for Excel
            cleaned_content, excel_data = clean_html_for_llm(
                normalized_html, extract_text_only=text_only
            )

            # Step 3: Write out to its own separate file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            # Step 4: Write out to Excel file (Tag in Column A, Data in Column B)
            df = pd.DataFrame(excel_data, columns=["Tag", "Content"])
            df.to_excel(excel_file, index=False)

            reduction = (
                ((len(raw_data) - len(cleaned_content)) / len(raw_data)) * 100
                if raw_data
                else 0
            )

            # Explicitly log whether the file was newly created or overwritten
            if file_already_exists:
                print(
                    f"[⇄ Overwrite] {file_path.name} -> {output_file.name} & {excel_file.name} ({reduction:.1f}% size reduction)"
                )
            else:
                print(
                    f"[✔ Created]   {file_path.name} -> {output_file.name} & {excel_file.name} ({reduction:.1f}% size reduction)"
                )

        except Exception as e:
            print(f"[✘] Error processing {file_path.name}: {e}")

# ==============================================================================
# SECTION 5: MAIN EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    process_file_list(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        process_all=PROCESS_ALL_FILES,
        file_list=INPUT_FILES,
        text_only=TEXT_ONLY_MODE,
        extensions=HTML_EXTENSIONS,
    )