"""

Cleaning process to make web pages useful for LLM processing
operates in two generic stages:
Unwrap Browser Markup: Standardize the code back into raw HTML.
Boilerplate & Noise Removal:
Strip out scripts, styles, embedded SVGs, navigation links, and structural bloat
using BeautifulSoup, leaving only clean semantic markup or plain text.
NOTE that sample file for this code came from:
view-source_https___www.afr.com_companies30Jul26.html

pip install beautifulsoup4
"""




import html
import re
from pathlib import Path
from bs4 import BeautifulSoup

# ==============================================================================
# CONFIGURATION / PARAMETERS (Edit these in PyCharm)
# ==============================================================================

# Input and Output Directories
INPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\InputhtmlSources"
OUTPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM"

# Specify individual files located inside INPUT_DIRECTORY
INPUT_FILES = [
    Path(INPUT_DIRECTORY) / "view-source_https___www.afr.com_companies30Jul26.html",
    #Path(INPUT_DIRECTORY) / "another_page.html",
    #Path(INPUT_DIRECTORY) / "sample_article.html",
]

# Output format mode:
# True  -> Output clean plain text/markdown (Recommended for LLMs - lowest token cost)
# False -> Output simplified HTML structure (Stripped of CSS/JS/Nav/Ads)
TEXT_ONLY_MODE = True
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
) -> str:
    """Generically strips noise (scripts, styles, SVGs, nav, headers, footers)

    from an HTML document to optimize context for LLMs.
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

    if extract_text_only:
        # Extract pure readable text line by line
        text = soup.get_text(separator="\n")
        # Collapse multiple blank lines
        cleaned_text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return cleaned_text

    # Return clean HTML structure without clutter
    clean_html = soup.prettify()
    return re.sub(r"\n\s*\n+", "\n", clean_html).strip()


def process_file_list(
    file_list: list, output_dir: str, text_only: bool = True
):
    """Processes a explicit list of file paths and outputs each to a separate cleaned file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting batch process for {len(file_list)} file(s). Output target: '{output_path.resolve()}'\n"
    )

    for file_item in file_list:
        file_path = Path(file_item)

        if not file_path.exists():
            print(f"[✘] Skipping (File not found): {file_item}")
            continue

        # Construct separate output file path
        ext = ".txt" if text_only else file_path.suffix
        output_file = output_path / f"{file_path.stem}_cleaned{ext}"

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_data = f.read()

            # Step 1: Normalize source-view wrapper format
            normalized_html = unescape_chrome_source(raw_data)

            # Step 2: Strip HTML noise
            cleaned_content = clean_html_for_llm(
                normalized_html, extract_text_only=text_only
            )

            # Step 3: Write out to its own separate file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            reduction = (
                ((len(raw_data) - len(cleaned_content)) / len(raw_data)) * 100
                if raw_data
                else 0
            )
            print(
                f"[✔] Success: {file_path.name} -> {output_file.name} ({reduction:.1f}% size reduction)"
            )

        except Exception as e:
            print(f"[✘] Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    process_file_list(
        file_list=INPUT_FILES,
        output_dir=OUTPUT_DIRECTORY,
        text_only=TEXT_ONLY_MODE,
    )