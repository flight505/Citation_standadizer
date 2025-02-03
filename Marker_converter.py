"""
marker_converter.py

A clean implementation for:
1. Converting PDF to Markdown using Marker
2. Standardizing citations and references using Gemini
3. Saving both original and standardized versions

Author: Jesper
"""

import os
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from termcolor import colored
import google.generativeai as genai
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# --------------------------------------------------------------------
# Constants & Configuration
# --------------------------------------------------------------------
PDF_DIR = Path("pdfs")
OUTPUT_DIR = Path(__file__).parent / "converted_pdfs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
GEMINI_MODELS = {
    "flash": {
        "name": "gemini-1.5-flash",
        "rpm": 2000,  # 2000 RPM for Flash
        "pricing": {
            "input": 0.00001875 / 1000,  # $0.01875 per 1M tokens
            "output": 0.000075 / 1000,  # $0.075 per 1M tokens
        },
    },
    "flash-8b": {
        "name": "gemini-1.5-flash-8b",
        "rpm": 4000,  # 4000 RPM for Flash-8B
        "pricing": {
            "input": 0.0000375 / 1000,  # $0.0375 per 1M tokens
            "output": 0.00015 / 1000,  # $0.15 per 1M tokens
            "cached": 0.00001 / 1000,  # $0.01 per 1M tokens for cached prompts
        },
    },
}

# Select model to use - change this to switch models
SELECTED_MODEL = "flash"  # or "flash-8b"
GEMINI_MODEL = GEMINI_MODELS[SELECTED_MODEL]["name"]
GEMINI_RPM = GEMINI_MODELS[SELECTED_MODEL]["rpm"]

# Marker configuration
MARKER_CONFIG = {
    "use_llm": False,
    "output_format": "markdown",
    "extract_images": False,
    "paginate_output": False,
    "page_separator": "",
    "debug": False,
}

# Gemini configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print(colored("Warning: GEMINI_API_KEY is not set. LLM calls may fail.", "red"))


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------
def remove_span_tags(markdown_content: str) -> str:
    """Remove page markers from Marker output."""
    markdown_content = re.sub(r'<span id="page-\d+-\d+"/>', "", markdown_content)
    markdown_content = re.sub(r"\(#page-\d+-\d+\)", "", markdown_content)
    return markdown_content


def get_token_count(text: str) -> int:
    """Simple token counter based on word splits."""
    return len(text.split())


def calculate_gemini_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate approximate cost for Gemini API usage."""
    model_config = next(
        (cfg for name, cfg in GEMINI_MODELS.items() if cfg["name"] == model), None
    )
    if not model_config:
        raise ValueError(
            f"Invalid model name. Choose from: {[cfg['name'] for cfg in GEMINI_MODELS.values()]}"
        )

    pricing = model_config["pricing"]
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]
    return input_cost + output_cost


# --------------------------------------------------------------------
# Gemini Processing
# --------------------------------------------------------------------
async def process_section_async(
    model: genai.GenerativeModel, section: str, prompt: str, generation_config: Dict
) -> Tuple[str, Dict]:
    """Process a single section asynchronously and return result with usage data."""
    try:
        response = await model.generate_content_async(
            prompt + "\n\nText to process:\n" + section,
            generation_config=generation_config,
            stream=False,
        )
        return response.text, response.usage_metadata
    except Exception as e:
        print(colored(f"Error in async processing: {str(e)}", "red"))
        return section, None


async def process_batch_async(
    model: genai.GenerativeModel,
    sections: List[str],
    prompts: List[str],
    generation_config: Dict,
    batch_size: int = GEMINI_RPM,
) -> List[Tuple[str, Dict]]:
    """Process a batch of sections using full RPM capacity."""
    print(colored(f"Processing all {len(sections)} sections in parallel...", "blue"))

    # Process all sections in parallel - no need to batch with 4000 RPM limit
    results = await asyncio.gather(
        *[
            process_section_async(model, section, prompt, generation_config)
            for section, prompt in zip(sections, prompts)
        ]
    )

    return results


def standardize_citations_gemini(text: str) -> Tuple[str, float]:
    """Use Gemini to standardize citations and references while preserving structure."""
    model = genai.GenerativeModel(GEMINI_MODEL)  # Use configured model
    citation_tracker = CitationTracker()
    total_cost = 0.0

    # Find and extract references section
    refs_match = re.search(
        r"^#+\s*References\s*\n(.*?)(?=^#|\Z)", text, re.MULTILINE | re.DOTALL
    )
    references_section = refs_match.group(0) if refs_match else ""
    main_text = text[: refs_match.start()] if refs_match else text

    # Split main text into sections
    sections = re.split(r"(^#+\s.*?\n)", main_text, flags=re.MULTILINE)
    sections = [s for s in sections if s.strip()]

    print(colored(f"Processing document in {len(sections)} sections...", "blue"))

    # Prepare prompts for all sections
    prompts = []
    for section in sections:
        citation_tracker.update_from_section(section)
        prompt = f"""
        Standardize the citations in this section to numeric format [n] while preserving the exact document structure.
        Rules:
        1. Handle cases where citations have mistakes due to conversion [e1], [c:1], "In another wor[k22]," etc. should be fixed to [1], [1], "In another work[22]," and so on. 
        2. Merges fragmented citations (e.g. [Wu] [et] [al.] [(2021)]) into a single citation [Wu et al., 2021]
        3. Remove any accidental double brackets and escapes (e.g. [[n]], [[n1,n2,n3]], [/[n]/], [/n/])
        4. Convert citations like (Author et al., 2020), [Author et al., 2020], Author et al., (2020), etc. to [n] format
        5. Format multiple citations as [1,2,3], if a range is present, always format as [1,2,3]
        6. Format figure/table citations/references as 'Fig. (1)', 'Table. (1)', etc. so that they are not mistaken for citations. Any mistaking of citations in the original text should be corrected, and errors in the original text should be corrected.
        7. Use these existing citation numbers:
        {citation_tracker.get_mapping_str()}
        
        IMPORTANT: Only modify citation formats. Do not add any other text or commentary.
        """
        prompts.append(prompt)

    # Process sections in parallel batches
    generation_config = {"temperature": 0.1, "top_p": 0.8, "top_k": 40}

    # Run async processing
    results = asyncio.run(
        process_batch_async(model, sections, prompts, generation_config)
    )

    # Collect results and calculate costs
    standardized_sections = []
    for result, usage in results:
        standardized_sections.append(result)
        if usage:
            cost = calculate_gemini_cost(
                usage.prompt_token_count, usage.candidates_token_count, GEMINI_MODEL
            )
            total_cost += cost
            print(colored(f"Section cost: ${cost:.6f}", "yellow"))

    # Process references section last
    if references_section:
        print(colored("Processing references section...", "blue"))
        try:
            refs_prompt = f"""
            Format the references section using these citation numbers while preserving the exact document structure.
            Rules:
            1. Format each reference as: [n] - <original reference text>
            2. Use these citation numbers:
            {citation_tracker.get_mapping_str()}
            3. Keep references in their original order
            4. Do not modify reference content beyond adding [n]
            
            IMPORTANT: Only add citation numbers. Do not modify the references or add commentary.
            """

            response = model.generate_content(
                refs_prompt + "\n\nReferences section:\n" + references_section,
                generation_config=generation_config,
            )

            if response.usage_metadata:
                cost = calculate_gemini_cost(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count,
                    GEMINI_MODEL,
                )
                total_cost += cost
                print(colored(f"References section cost: ${cost:.6f}", "yellow"))

            standardized_sections.append(response.text)

        except Exception as e:
            print(colored(f"Error processing references: {str(e)}", "red"))
            standardized_sections.append(references_section)

    print(colored(f"Total processing cost: ${total_cost:.6f}", "green"))
    return "\n".join(standardized_sections), total_cost


class CitationTracker:
    """Track and manage citation numbering across document sections."""

    def __init__(self):
        self.citation_map = {}  # Maps author-year to number
        self.next_number = 1

    def update_from_section(self, text: str):
        """Extract citations from text and update mapping."""
        # Match common citation patterns
        patterns = [
            r"\(([^)]+?)\s*,\s*\d{4}[a-z]?\)",  # (Author et al., 2020)
            r"\[([^]]+?)\s*,\s*\d{4}[a-z]?\]",  # [Author et al., 2020]
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = match.group(1)
                if citation not in self.citation_map:
                    self.citation_map[citation] = self.next_number
                    self.next_number += 1

    def get_mapping_str(self) -> str:
        """Get citation mapping as formatted string."""
        return "\n".join(
            f"{author} -> [{num}]" for author, num in self.citation_map.items()
        )


# --------------------------------------------------------------------
# Core Conversion Logic
# --------------------------------------------------------------------
def convert_and_standardize(pdf_path: Path) -> bool:
    """
    Convert PDF to markdown and standardize citations/references.
    Returns True if successful, False otherwise.
    """
    try:
        t0 = time.time()
        print(colored(f"\nProcessing: {pdf_path.name}", "blue"))

        # Initialize Marker converter
        config_parser = ConfigParser(MARKER_CONFIG)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        # Convert PDF to markdown
        print(colored("Converting PDF to markdown...", "blue"))
        rendered = converter(str(pdf_path))
        markdown_content, _, _ = text_from_rendered(rendered)
        markdown_content = remove_span_tags(markdown_content)

        # Save original markdown
        pdf_stem = pdf_path.stem
        original_path = OUTPUT_DIR / f"{pdf_stem}_original.md"
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(colored(f"✓ Saved original markdown: {original_path}", "green"))

        # Standardize citations and references
        print(colored("Standardizing citations and references...", "blue"))
        standardized, cost = standardize_citations_gemini(markdown_content)

        # Save standardized version
        standardized_path = OUTPUT_DIR / f"{pdf_stem}_standardized.md"
        with open(standardized_path, "w", encoding="utf-8") as f:
            f.write(standardized)
        print(colored(f"✓ Saved standardized version: {standardized_path}", "green"))

        # Log completion
        duration = round(time.time() - t0, 2)
        print(colored(f"✓ Completed in {duration}s", "green"))
        print(colored(f"✓ Total cost for {pdf_path.name}: ${cost:.6f}", "green"))
        return True, cost

    except Exception as e:
        print(colored(f"Error processing {pdf_path.name}: {str(e)}", "red"))
        return False, 0.0


# --------------------------------------------------------------------
# Main Entry Point
# --------------------------------------------------------------------
def main():
    """Process all PDFs in the input directory."""
    print(colored("Starting PDF processing...", "cyan"))

    # Check for PDF files
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(colored("No PDF files found in 'pdfs' folder.", "red"))
        return

    # Process each PDF
    print(colored(f"Found {len(pdf_files)} PDF file(s) to process...", "blue"))
    success_count = 0
    total_cost = 0.0

    for pdf_path in pdf_files:
        success, cost = convert_and_standardize(pdf_path)
        if success:
            success_count += 1
            total_cost += cost

    # Final summary
    print(colored("\nProcessing Summary:", "cyan"))
    print(colored(f"Total files: {len(pdf_files)}", "yellow"))
    print(colored(f"Successfully processed: {success_count}", "green"))
    print(colored(f"Failed: {len(pdf_files) - success_count}", "red"))
    print(colored(f"Total cost for all PDFs: ${total_cost:.6f}", "green"))


if __name__ == "__main__":
    main()
