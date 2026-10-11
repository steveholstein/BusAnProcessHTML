"""
algorithmic_news_refiner.py

A generalised pipeline to isolate news headlines from raw scraped CSV/HTML data
without using hardcoded website-specific keyword blacklists.

Requires:
    pip install pandas spacy
    python -m spacy download en_core_web_sm
"""

from pathlib import Path
from typing import Optional
import re
import pandas as pd
import spacy


# Input and Output Directories
INPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM"
OUTPUT_DIRECTORY = r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM"



class HeuristicNewsFilter:
    def __init__(self, spacy_model="en_core_web_sm"):
        # Auto-download the lightweight English model if it is not already installed locally
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner"])
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model, disable=["ner"])

    def process(self, text):
        # Do NOT load the model here; reuse self.nlp
        return self.nlp(text)

    @staticmethod
    def _is_boilerplate_shape(text: str) -> bool:
        """
        Filters out pure boilerplate, metadata, and timestamps.
        """
        t = text.strip()

        # Duration / time stamps (e.g., '38:21', '2:33')
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t):
            return True

        # Pure numbers, dates, or symbols
        if re.fullmatch(r"[\d\s\W_]+", t):
            return True

        # Boilerplate / legal phrases
        if re.search(
                r"[©®™]|\b(all rights reserved|terms of service|privacy policy|cookie settings|sign in|subscribe)\b", t,
                re.IGNORECASE):
            return True

        # Headlines must be at least 3 words and at least 15 characters
        words = t.split()
        if len(words) < 3 or len(t) < 15:
            return True

        return False

    def _has_headline_grammar(self, doc: spacy.tokens.Doc) -> bool:
        """
        Generic headline recognition:
        - Catches finite verbs, participles, or auxiliaries in standard or lowercased text.
        - Accepts substantive headline phrases (>= 4 words) with clausal or prepositional depth.
        """
        # 1. Any verbal or clausal elements in original or lowercased form
        has_verbal_element = any(
            token.pos_ in ("VERB", "AUX")
            for token in doc
        )
        if has_verbal_element:
            return True

        # Check lowercased tokens (resolves Title Case bias where verbs get tagged as PROPN)
        doc_lower = self.nlp(doc.text.lower())
        if any(token.pos_ in ("VERB", "AUX") for token in doc_lower):
            return True

        # 2. Substantive noun phrase headlines (e.g., "The Killing of Bob Lee E5", "The Future of Gene Editing...")
        # If it has 4+ words and includes prepositions/conjunctions/colons, it's a substantive title, not a menu.
        words = doc.text.split()
        if len(words) >= 4:
            has_connective = any(
                token.pos_ in ("ADP", "CCONJ", "SCONJ") or token.text in (":", "—", "-", "’s", "'s")
                for token in doc
            )
            if has_connective:
                return True

        return False

    def refine(
            self,
            file_path: str | Path,
            content_col: str = "Content",
            tag_col: str = "Tag",
            tag_whitelist: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

        col_map = {c: c.strip().capitalize() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        content_col = content_col.capitalize()
        tag_col = tag_col.capitalize()

        # Clean whitespace and strip stray surrounding quotes
        df[content_col] = df[content_col].astype(str).str.strip().str.strip("'\"")
        df[tag_col] = df[tag_col].astype(str).str.strip().str.lower()

        # 1. Tag Filtering (Include standard container and heading tags)
        if tag_whitelist is None:
            tag_whitelist = {"span", "section", "h1", "h2", "h3", "h4", "p", "article", "a"}

        df = df[df[tag_col].isin(tag_whitelist)].copy()

        # 2. Heuristic Shape Filter
        df = df[~df[content_col].apply(self._is_boilerplate_shape)].copy()

        # 3. Deduplication (preserves first occurrence)
        df.drop_duplicates(subset=[content_col], inplace=True)

        # 4. Syntactic Structure Parsing
        docs = list(self.nlp.pipe(df[content_col].tolist(), batch_size=64))

        valid_indices = []
        for (idx, _), doc in zip(df.iterrows(), docs):
            if self._has_headline_grammar(doc):
                valid_indices.append(idx)

        refined_df = df.loc[valid_indices].copy()
        refined_df.reset_index(drop=True, inplace=True)
        refined_df["Item_No"] = refined_df.index + 1

        return refined_df[["Item_No", tag_col, content_col]]

if __name__ == "__main__":
    input_csv = Path(OUTPUT_DIRECTORY) /"view-source_https___www.bloomberg.com_technology_cleaned.csv"

    cleaner = HeuristicNewsFilter()
    result = cleaner.refine(input_csv)

    print(f"Algorithmically isolated {len(result)} candidate news items:\n")
    for _, row in result.iterrows():
        print(f"{row['Item_No']}. [{row['Tag']}] {row['Content']}")

result.to_csv(Path(OUTPUT_DIRECTORY) /"refined_output.csv", index=False)
