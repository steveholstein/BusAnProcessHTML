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
        Filters out timestamps, legal notices, single tokens, and pure symbols.
        """
        # Duration stamps (e.g., '38:21', '2:33')
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", text):
            return True

        # Copyright, disclaimers, and legal notices
        if re.search(r"[©®™]|\b(all rights reserved|terms of service|privacy policy|cookie settings)\b", text,
                     re.IGNORECASE):
            return True

        words = text.split()
        # Single-token or two-word site navigation labels
        if len(words) < 3:
            return True

        # Pure numbers, dates, or symbols
        if re.fullmatch(r"[\d\s\W_]+", text):
            return True

        return False

    def _has_headline_grammar(self, doc: spacy.tokens.Doc) -> bool:
        """
        Determines whether candidate text exhibits headline characteristics:
        1. Contains finite verbs, auxiliaries, or participles (evaluates both Title Case and lowercase).
        2. OR represents a substantive headline phrase (>= 5 words with rich noun-phrase structure).
        """

        def doc_has_verb(d: spacy.tokens.Doc) -> bool:
            return any(
                token.pos_ in ("VERB", "AUX")
                for token in d
            )

        # Pass 1: Standard token check
        if doc_has_verb(doc):
            return True

        # Pass 2: Overcome Title-Case POS tagging bias (e.g. 'Turkey Bans...' -> 'bans' as VERB)
        doc_lower = self.nlp(doc.text.lower())
        if doc_has_verb(doc_lower):
            return True

        # Pass 3: Long-form / headlinese noun phrases (e.g. 'The Future of Gene Editing with Jennifer Doudna')
        words = doc.text.split()
        if len(words) >= 5:
            # Must contain at least one preposition, conjunction, or punctuation indicating depth
            has_connective = any(
                token.pos_ in ("ADP", "CCONJ", "SCONJ") or token.text in (":", "—", "-") for token in doc)
            if has_connective:
                return True

        return False

    def _has_headline_grammar(self, doc: spacy.tokens.Doc) -> bool:
        """
        Evaluates whether the candidate phrase exhibits headline grammar.
        News headlines almost always contain a ROOT verb, auxiliary, or passive participle.
        Section labels and menu items (e.g., 'Big Tech', 'The Circuit') are typically pure noun phrases.
        """
        # Check for presence of finite verbs, auxiliaries, or past participles acting as predicates
        has_verb = any(
            token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "advcl", "relcl", "ccomp")
            for token in doc
        )
        return has_verb

    def refine(
            self,
            file_path: str | Path,
            content_col: str = "Content",
            tag_col: str = "Tag",
            tag_whitelist: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """
        Generically extracts news items and produces dynamically filtered content.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

        # Standardize columns
        col_map = {c: c.strip().capitalize() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        content_col = content_col.capitalize()
        tag_col = tag_col.capitalize()

        # Clean string formatting
        df[content_col] = df[content_col].astype(str).str.strip()
        df[tag_col] = df[tag_col].astype(str).str.strip().str.lower()

        # 1. Structural Tag Filtering
        # Default target tags where headlines and substantive content reside
        if tag_whitelist is None:
            tag_whitelist = {"span", "section", "h1", "h2", "h3", "p", "article"}

        df = df[df[tag_col].isin(tag_whitelist)].copy()

        # 2. Heuristic Shape Filter
        df = df[~df[content_col].apply(self._is_boilerplate_shape)].copy()

        # 3. Deduplication (preserves first occurrence)
        df.drop_duplicates(subset=[content_col], inplace=True)

        # 4. Syntactic Structure Parsing (distinguish headlines from section headers)
        # Process in batches for performance
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
