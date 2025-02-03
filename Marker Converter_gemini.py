"""
marker_converter.py

A clean implementation for:
1. Converting PDF to Markdown using Marker
2. Standardizing citations and references using Gemini
3. Saving both original and standardized versions
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

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

# Marker configuration
MARKER_CONFIG = {
    "use_llm": False,  # LLM is handled externally
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
    pricing = {
        "Gemini 1.5 Pro": {"input": 0.00125 / 1000, "output": 0.00375 / 1000},
        "Gemini 1.5 Flash": {"input": 0.00001875 / 1000, "output": 0.000075 / 1000},
    }

    if model not in pricing:
        raise ValueError("Invalid model name. Choose 'Gemini 1.5 Pro' or 'Gemini 1.5 Flash'.")

    input_cost = input_tokens * pricing[model]["input"]
    output_cost = output_tokens * pricing[model]["output"]
    return input_cost + output_cost

# --------------------------------------------------------------------
# Gemini Processing
# --------------------------------------------------------------------
def standardize_citations_gemini(text: str) -> str:
    """
    Use Gemini to standardize citations and references while preserving structure.
    """
    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name=model_name)
    
    # Enhanced prompt with explicit reference preservation
    prompt = """
    Standardize the citations and references in this academic text while maintaining the document's structure.
    Follow these rules carefully:

    Citation Rules:
    1. Convert all citations to numeric format [n]
    2. Format multiple citations as [1,2,3] (not [1-3])
    3. Keep figure/table references as 'Fig. 1', 'Table 1', etc.
    4. Maintain citation-reference number consistency
    
    Reference Rules:
    1. Preserve the entire References section in its original location
    2. Format each reference as: [n] - 
    3. Keep existing reference numbers if present
    4. Number references sequentially if unnumbered
    5. Ensure reference numbers match their citations
    6. Do not remove or reorder sections
    7. Do not add fields (like DOI) if not present in original
    
    **IMPORTANT**
    - Preserve all section headings and content structure
    - Keep the References section intact and in its original location
    - Only modify citation formats and reference formatting
    """
    
    try:
        response = model.generate_content([prompt, text])
        standardized = response.text

        # Calculate and log cost
        input_tokens = get_token_count(prompt)
        output_tokens = get_token_count(standardized)
        usage_cost = calculate_gemini_cost(input_tokens, output_tokens, "Gemini 1.5 Flash")
        print(colored(f"💰 Gemini usage cost: ${usage_cost:.6f}", "yellow"))

        return standardized

    except Exception as e:
        print(colored(f"Error in Gemini processing: {str(e)}", "red"))
        return text  # Return original text if processing fails

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
        standardized = standardize_citations_gemini(markdown_content)

        # Save standardized version
        standardized_path = OUTPUT_DIR / f"{pdf_stem}_standardized.md"
        with open(standardized_path, "w", encoding="utf-8") as f:
            f.write(standardized)
        print(colored(f"✓ Saved standardized version: {standardized_path}", "green"))

        # Log completion
        duration = round(time.time() - t0, 2)
        print(colored(f"✓ Completed in {duration}s", "green"))
        return True

    except Exception as e:
        print(colored(f"Error processing {pdf_path.name}: {str(e)}", "red"))
        return False

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
    
    for pdf_path in pdf_files:
        if convert_and_standardize(pdf_path):
            success_count += 1

    # Final summary
    print(colored("\nProcessing Summary:", "cyan"))
    print(colored(f"Total files: {len(pdf_files)}", "yellow"))
    print(colored(f"Successfully processed: {success_count}", "green"))
    print(colored(f"Failed: {len(pdf_files) - success_count}", "red"))

if __name__ == "__main__":
    main()
