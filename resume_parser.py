"""Resume parser module for extracting text and structured metadata from PDF files.

Supports:
1. Native embedded-text PDF extraction via PyPDF
2. Scanned / image-only PDF OCR fallback via pdf2image & pytesseract
3. In-memory buffer processing (no permanent file storage)
4. Section identification, structural health score, and contact info extraction
"""

import os
import io
import shutil
import glob
from typing import Dict, Any, Union, BinaryIO, Optional
from pypdf import PdfReader
from utils.text_processing import (
    clean_text,
    extract_contact_info,
    segment_resume_sections,
    calculate_text_metrics
)

# Optional OCR libraries
try:
    import pdf2image
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class ResumeParseError(Exception):
    """Custom exception raised when resume parsing fails."""
    pass


class ResumeParser:
    """Extracts text and structured metadata from resume PDF files with OCR fallback."""

    MIN_NATIVE_CHAR_COUNT = 80
    MAX_OCR_PAGES = 5
    OCR_DPI = 300

    @classmethod
    def _find_tesseract_cmd(cls) -> Optional[str]:
        """Locates Tesseract executable on PATH or common Windows installation directories."""
        # 1. Check if already configured or on PATH
        if shutil.which("tesseract"):
            return "tesseract"
        
        # 2. Check environment variable
        env_path = os.environ.get("TESSERACT_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path
            
        # 3. Check common Windows installation paths
        common_win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            r"C:\Tesseract-OCR\tesseract.exe"
        ]
        for path in common_win_paths:
            if os.path.isfile(path):
                return path
        return None

    @classmethod
    def _find_poppler_path(cls) -> Optional[str]:
        """Locates Poppler binary directory on PATH or common Windows directories."""
        # 1. Check if pdftoppm is on PATH
        if shutil.which("pdftoppm"):
            return None  # None means pdf2image will use system PATH

        # 2. Check environment variable
        env_path = os.environ.get("POPPLER_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path

        # 3. Check common Windows installation directories, including WinGet installs
        common_win_dirs = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\Library\bin",
            r"C:\Program Files (x86)\poppler\bin",
            r"C:\poppler\Library\bin",
            r"C:\poppler\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\poppler\bin"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\poppler\Library\bin"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages")
        ]
        for p_dir in common_win_dirs:
            if os.path.isdir(p_dir) and os.path.isfile(os.path.join(p_dir, "pdftoppm.exe")):
                return p_dir

            if os.path.isdir(p_dir):
                matches = glob.glob(os.path.join(p_dir, "**", "pdftoppm.exe"), recursive=True)
                if matches:
                    return os.path.dirname(matches[0])
        return None

    @classmethod
    def _extract_ocr_text(cls, pdf_bytes: bytes, total_pages: int) -> Dict[str, Any]:
        """Converts PDF pages to images in-memory and extracts text using Tesseract OCR.

        Args:
            pdf_bytes: Raw in-memory PDF bytes.
            total_pages: Total number of pages in the PDF document.

        Returns:
            Dict containing extracted text, extraction_method, warnings, or error details.
        """
        if not PDF2IMAGE_AVAILABLE:
            return {
                "success": False,
                "text": "",
                "error": "OCR module 'pdf2image' is not installed. Please install it via 'pip install pdf2image'."
            }

        if not PYTESSERACT_AVAILABLE:
            return {
                "success": False,
                "text": "",
                "error": "OCR module 'pytesseract' is not installed. Please install it via 'pip install pytesseract Pillow'."
            }

        # Configure Tesseract binary path if on Windows
        tesseract_cmd = cls._find_tesseract_cmd()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            os.environ["TESSERACT_PATH"] = tesseract_cmd
        else:
            return {
                "success": False,
                "text": "",
                "error": (
                    "Tesseract OCR engine was not found on your system.\n\n"
                    "**Installation Instructions:**\n"
                    "- **Windows:** Download the installer from https://github.com/UB-Mannheim/tesseract/wiki "
                    "and install to default directory `C:\\Program Files\\Tesseract-OCR` or add to PATH.\n"
                    "- **Ubuntu/Debian:** Run `sudo apt-get update && sudo apt-get install -y tesseract-ocr`\n"
                    "- **macOS:** Run `brew install tesseract`"
                )
            }

        poppler_dir = cls._find_poppler_path()
        if poppler_dir:
            os.environ["POPPLER_PATH"] = poppler_dir

        # Check poppler availability
        if poppler_dir is None and not shutil.which("pdftoppm") and os.name == "nt":
            # On Windows, if pdftoppm is not on PATH and not in common folders
            return {
                "success": False,
                "text": "",
                "error": (
                    "Poppler utility was not found on your system (required to convert PDF to images).\n\n"
                    "**Installation Instructions:**\n"
                    "- **Windows:** Download binary from https://github.com/oschwartz10612/poppler-windows/releases, "
                    "extract and add the `bin/` folder to PATH, or set `POPPLER_PATH` environment variable.\n"
                    "- **Ubuntu/Debian:** Run `sudo apt-get install -y poppler-utils`\n"
                    "- **macOS:** Run `brew install poppler`"
                )
            }

        pages_to_process = min(total_pages, cls.MAX_OCR_PAGES)
        ocr_warning = None
        if total_pages > cls.MAX_OCR_PAGES:
            ocr_warning = f"The uploaded PDF has {total_pages} pages. OCR was limited to the first {cls.MAX_OCR_PAGES} pages for optimal performance."

        try:
            # Convert PDF pages to PIL images entirely in-memory
            images = convert_from_bytes(
                pdf_bytes,
                dpi=cls.OCR_DPI,
                first_page=1,
                last_page=pages_to_process,
                poppler_path=poppler_dir
            )

            if not images:
                return {
                    "success": False,
                    "text": "",
                    "error": "No image frames could be extracted from the scanned PDF."
                }

            page_texts = []
            for i, img in enumerate(images):
                page_str = pytesseract.image_to_string(img)
                if page_str.strip():
                    page_texts.append(page_str.strip())

            full_ocr_text = "\n\n".join(page_texts).strip()

            if len(full_ocr_text) < 30:
                return {
                    "success": False,
                    "text": "",
                    "error": "No readable text could be recovered from this PDF. Upload a text-based PDF or a clear, high-resolution scanned resume."
                }

            return {
                "success": True,
                "text": full_ocr_text,
                "ocr_warning": ocr_warning,
                "error": None
            }

        except Exception as e:
            err_msg = str(e)
            if "tesseract" in err_msg.lower():
                return {
                    "success": False,
                    "text": "",
                    "error": f"Tesseract OCR failed: {err_msg}. Ensure Tesseract is properly installed."
                }
            elif "poppler" in err_msg.lower() or "pdftoppm" in err_msg.lower():
                return {
                    "success": False,
                    "text": "",
                    "error": f"Poppler conversion failed: {err_msg}. Ensure Poppler is installed."
                }
            return {
                "success": False,
                "text": "",
                "error": f"OCR processing failed: {err_msg}"
            }

    @classmethod
    def extract_text_from_pdf(cls, file_source: Union[str, BinaryIO, bytes]) -> Dict[str, Any]:
        """Extracts text from a PDF file using native PyPDF with automatic OCR fallback for scanned PDFs.

        Args:
            file_source: File path, file-like object, or raw in-memory bytes.

        Returns:
            Dict containing:
                - text: Extracted plain text string
                - raw_text: Original extracted text
                - page_count: Total number of pages
                - extraction_method: "native" or "ocr"
                - ocr_used: Boolean flag indicating if OCR was used
                - ocr_warning: Optional warning string (e.g. if page count > 5)
                - is_scanned_or_empty: Boolean flag
                - error: Error message string if any, else None
        """
        try:
            if isinstance(file_source, bytes):
                pdf_bytes = file_source
                reader = PdfReader(io.BytesIO(file_source))
            elif isinstance(file_source, str):
                with open(file_source, "rb") as f:
                    pdf_bytes = f.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
            else:
                # File-like object (e.g. UploadedFile)
                if hasattr(file_source, "getvalue"):
                    pdf_bytes = file_source.getvalue()
                elif hasattr(file_source, "read"):
                    pdf_bytes = file_source.read()
                    if hasattr(file_source, "seek"):
                        file_source.seek(0)
                else:
                    pdf_bytes = bytes(file_source)
                reader = PdfReader(io.BytesIO(pdf_bytes))

            page_count = len(reader.pages)
            if page_count == 0:
                return {
                    "text": "",
                    "raw_text": "",
                    "page_count": 0,
                    "extraction_method": "native",
                    "ocr_used": False,
                    "ocr_warning": None,
                    "is_scanned_or_empty": True,
                    "error": "The uploaded PDF has 0 pages."
                }

            # Attempt native embedded-text extraction
            extracted_pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)

            full_native_text = "\n\n".join(extracted_pages).strip()

            # Check if native extraction yielded sufficient meaningful characters
            if len(full_native_text) >= cls.MIN_NATIVE_CHAR_COUNT:
                return {
                    "text": full_native_text,
                    "raw_text": full_native_text,
                    "page_count": page_count,
                    "extraction_method": "native",
                    "ocr_used": False,
                    "ocr_warning": None,
                    "is_scanned_or_empty": False,
                    "error": None
                }

            # Less than 80 characters extracted: Scanned / image-only PDF detected -> Trigger OCR Fallback
            ocr_res = cls._extract_ocr_text(pdf_bytes, page_count)

            if ocr_res["success"] and len(ocr_res["text"].strip()) >= 30:
                return {
                    "text": ocr_res["text"],
                    "raw_text": ocr_res["text"],
                    "page_count": page_count,
                    "extraction_method": "ocr",
                    "ocr_used": True,
                    "ocr_warning": ocr_res.get("ocr_warning"),
                    "is_scanned_or_empty": False,
                    "error": None
                }
            else:
                # OCR failed or returned empty
                ocr_error = ocr_res.get("error")
                if ocr_error and ("Tesseract OCR engine was not found" in ocr_error or "Poppler utility was not found" in ocr_error):
                    return {
                        "text": "",
                        "raw_text": "",
                        "page_count": page_count,
                        "extraction_method": "ocr",
                        "ocr_used": True,
                        "ocr_warning": None,
                        "is_scanned_or_empty": True,
                        "error": ocr_error
                    }
                else:
                    return {
                        "text": "",
                        "raw_text": "",
                        "page_count": page_count,
                        "extraction_method": "ocr",
                        "ocr_used": True,
                        "ocr_warning": None,
                        "is_scanned_or_empty": True,
                        "error": "No readable text could be recovered from this PDF. Upload a text-based PDF or a clear, high-resolution scanned resume."
                    }

        except Exception as e:
            return {
                "text": "",
                "raw_text": "",
                "page_count": 0,
                "extraction_method": "native",
                "ocr_used": False,
                "ocr_warning": None,
                "is_scanned_or_empty": True,
                "error": f"Failed to parse PDF: {str(e)}"
            }

    @classmethod
    def parse(cls, file_source: Union[str, BinaryIO, bytes]) -> Dict[str, Any]:
        """Performs full extraction and analysis of resume structure and metadata.

        Args:
            file_source: File path, file-like object, or raw in-memory bytes.

        Returns:
            Dict containing full analysis details, sections, contacts, metrics, and extraction method.
        """
        pdf_res = cls.extract_text_from_pdf(file_source)
        if pdf_res["error"]:
            return {
                "success": False,
                "error": pdf_res["error"],
                "text": "",
                "extraction_method": pdf_res.get("extraction_method", "native"),
                "ocr_used": pdf_res.get("ocr_used", False),
                "ocr_warning": pdf_res.get("ocr_warning"),
                "metadata": {}
            }

        raw_text = pdf_res["raw_text"]
        contacts = extract_contact_info(raw_text)
        sections = segment_resume_sections(raw_text)
        metrics = calculate_text_metrics(raw_text)

        # Section health evaluation
        section_status = {
            "has_contact_info": bool(contacts.get("email") or contacts.get("phone")),
            "has_summary": bool(sections.get("summary") and len(sections["summary"]) > 40),
            "has_skills_section": bool(sections.get("skills") and len(sections["skills"]) > 20),
            "has_experience_section": bool(sections.get("experience") and len(sections["experience"]) > 50),
            "has_education_section": bool(sections.get("education") and len(sections["education"]) > 30),
            "has_projects_section": bool(sections.get("projects") and len(sections["projects"]) > 40),
            "has_certifications": bool(sections.get("certifications") and len(sections["certifications"]) > 20),
            "has_github": bool(contacts.get("github")),
            "has_linkedin": bool(contacts.get("linkedin")),
            "has_portfolio": bool(contacts.get("portfolio"))
        }

        # Calculate structure health score (0-100)
        core_checks = [
            section_status["has_contact_info"],
            section_status["has_skills_section"],
            section_status["has_experience_section"],
            section_status["has_education_section"],
            section_status["has_projects_section"],
            section_status["has_summary"]
        ]
        structure_health_score = int((sum(core_checks) / len(core_checks)) * 100)

        return {
            "success": True,
            "error": None,
            "text": raw_text,
            "cleaned_text": clean_text(raw_text),
            "page_count": pdf_res["page_count"],
            "extraction_method": pdf_res["extraction_method"],
            "ocr_used": pdf_res["ocr_used"],
            "ocr_warning": pdf_res.get("ocr_warning"),
            "contacts": contacts,
            "sections": sections,
            "metrics": metrics,
            "section_status": section_status,
            "structure_health_score": structure_health_score
        }
