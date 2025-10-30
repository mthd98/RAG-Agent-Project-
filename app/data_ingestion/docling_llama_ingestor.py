"""
docling_llama_ingestor.py

Minimal class to:
 - convert files with Docling (OCR optional) -> returns Markdown + Docling JSON
 - parse the Markdown with LlamaIndex MarkdownNodeParser
 - (recommended) parse the Docling JSON with DoclingNodeParser to preserve page/bbox/char/ocr metadata

Install required packages (example):
     pip install docling llama-index llama-index-readers-docling

Notes:
 - This class keeps things simple and pragmatic. For strict per-element metadata mapping
   prefer `parse_with_docling_node_parser()` which uses Docling JSON directly.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from core.config import settings
import logging
logger = logging.getLogger(__name__)
# Docling converter

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
   
# LlamaIndex Document + node parsers
try:
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.readers.docling import DoclingReader
    from llama_index.node_parser.docling import DoclingNodeParser
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.node_parser import SentenceSplitter
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.otel import register

    tracer_provider = register(
        project_name="rag-app", # Default is 'default'
        auto_instrument=True ,# Auto-instrument your app based on installed OI dependencies
        endpoint=  settings.PHOENIX_COLLECTOR_ENDPOINT+"/v1/traces", # Phoenix Collector endpoint
        )

    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)



except Exception as e:
    #if llama_index (or docling node parser extension) missing
    raise ImportError(
        "llama_index (and optionally the llama-index-readers-docling extension) are required. "
        "Install via `pip install llama-index` and `pip install llama-index-readers-docling`."
    ) from e


class DoclingLlamaIngestor:
    """
    Minimal, class to convert files with Docling and parse them with LlamaIndex node parsers.

    Methods:
    - convert_to_markdown_and_json(path) -> (markdown_str, docling_json)
    - parse_with_markdown_node_parser(path) -> list of Nodes (header-based)
    - parse_markdown_file(path) -> list of Nodes (from existing .md file)
    - parse_with_docling_node_parser(path) -> list of Nodes (typed nodes with rich metadata)

    Use parse_with_docling_node_parser when you need precise per-chunk metadata (page, bbox, char offsets, ocr_confidence).
    """

    def __init__(self, converter: Optional[DocumentConverter] = None):
        # allow passing a pre-configured Docling DocumentConverter
        if converter:
            self.converter = converter
        else:
            # Set up a default, configurable converter as per your example
            
            # 1. Define OCR options to use (e.g., Tesseract CLI)
            # We'll use TesseractCliOcrOptions as a robust default.
            # Note: Tesseract must be installed and accessible in your system's PATH
            # or specified directly with tesseract_cmd.
            ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
            #ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")

            
            # 2. Define PDF options
            pdf_pipeline_options = PdfPipelineOptions()
            # We don't set do_ocr=True here; we let the 'ocr' flag in the convert method control it.
            # But we specify *how* OCR should be done if it's requested.
            pdf_pipeline_options.ocr_options = ocr_options
            pdf_pipeline_options.do_table_structure = True
            pdf_pipeline_options.table_structure_options.do_cell_matching = True

        

            # 5. Build format_options dictionary for different file types
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                )
            }
            
            # 6. Initialize the converter with these specific format options
            self.converter = DocumentConverter(format_options=format_options,)

    def convert_to_markdown_and_json(
         self, file_path: str
     ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a file with Docling and return (markdown_text, docling_json_dict).

        - file_path: local path or URL to file
        Returns: (markdown string, docling JSON dict)"""
        path = str(file_path)
        logger.info(f"Converting file: {path}")     
        result = self.converter.convert(path)
        
        try:
            md = result.document.export_to_markdown()
        except Exception:
            # fallback: try to stringify if export_to_markdown missing
            md = str(result.document)

        try:
            doc_json = result.document.to_dict()
        except Exception:
            # some Docling versions provide `.to_dict()` or `.export()`; try both
            try:
                doc_json = result.document.export_to_json()
            except Exception:
                # last resort: store what we can
                doc_json = {"raw": repr(result.document)}
        
        logger.info(f"Conversion successful. Markdown length: {len(md)}, JSON keys: {list(doc_json.keys())}")
        return md, doc_json

    def parse_with_markdown_node_parser(
         self, file_path: str, include_metadata: bool = True
     ):
         """
         Convert file -> markdown -> parse markdown with LlamaIndex MarkdownNodeParser.

         Returns: list of Node objects (as returned by parser.get_nodes_from_documents([...]))
         Note: MarkdownNodeParser is header-aware. It will preserve document-level metadata,
               but it won't magically reconstruct Docling's page/bbox/char offsets for each node.
         """
         md, doc_json = self.convert_to_markdown_and_json(file_path)

         # Prepare a LlamaIndex Document that holds markdown text + doc-level metadata
         doc_meta = {
             "source_file": str(file_path),
             "docling_export": True, 
         }
         

         llama_doc = LlamaDocument(text=md, metadata=doc_meta)

         parser = MarkdownNodeParser(include_metadata=include_metadata)

         # get_nodes_from_documents expects a sequence
         nodes = parser.get_nodes_from_documents([llama_doc])
         return nodes

    def parse_markdown_file(
        self, markdown_file_path: str, include_metadata: bool = True
    ):
        """
        Read an *existing* Markdown file and parse it with LlamaIndex MarkdownNodeParser.

        This is different from `parse_with_markdown_node_parser`, which first converts
        a file (like a PDF) *to* Markdown. This method assumes you already have a
        Markdown file.

        Args:
            markdown_file_path: Path to the .md file.
            include_metadata: Passed to MarkdownNodeParser.

        Returns:
            list of Node objects (as returned by parser.get_nodes_from_documents([...]))
        """
        logger.info(f"Reading Markdown file: {markdown_file_path}")
        
        # 1. Read the Markdown file content
        try:
            with open(markdown_file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
        except Exception as e:
            logger.exception(f"Error reading Markdown file {markdown_file_path}: {e}", file=sys.stderr)
            return [] # Return empty list on failure

        # 2. Prepare a LlamaIndex Document
        doc_meta = {
            "source_file": str(markdown_file_path),
            "docling_export": False, 
        }

        llama_doc = LlamaDocument(text=md_content, metadata=doc_meta)

        # 3. Initialize the parser (already imported)
        parser = MarkdownNodeParser(include_metadata=include_metadata)

        # 4. Get nodes
        nodes = parser.get_nodes_from_documents([llama_doc])
        logger.info(f"Markdown file parser returned {len(nodes)} nodes.")
        return nodes
     
    async def parse_with_docling_node_parser(
        self, file_path: str, node_parser: Optional[Any] = DoclingNodeParser()
    ):
        """
            Convert file -> docling JSON -> parse with DoclingNodeParser (recommended).

            Robustly handles ConversionResult or dict-like returns from DocumentConverter,
            and ensures the payload we pass to LlamaIndex is JSON-serializable.
            """
        # 1 Load and convert your document using DoclingReader
        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON,doc_converter=self.converter)

        # You can load PDFs, Word docs, etc.
        documents = reader.load_data(file_path=str(file_path))

        # 3️ Convert documents into LlamaIndex nodes
        nodes = node_parser.get_nodes_from_documents(documents)

        return nodes
