from fastapi import UploadFile, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
import io
import docx

class DocumentProcessor:
    """Handles file parsing and text chunking for various document formats."""

    async def process_file(self, file: UploadFile) -> str:
        """Extracts text content from an uploaded file.

        Supports PDF, DOCX, and TXT formats.

        Args:
            file (UploadFile): The uploaded file.

        Returns:
            str: Extracted text content.

        Raises:
            HTTPException: If file format is unsupported or parsing fails.

        Examples:
            >>> text = await processor.process_file(uploaded_file)
            >>> print(text[:50])
            'Sample document text...'
        """
        content = await file.read()
        filename = file.filename.lower()
        
        text = ""
        
        try:
            if filename.endswith(".pdf"):
                text = self._parse_pdf(content)
            elif filename.endswith(".docx"):
                text = self._parse_docx(content)
            elif filename.endswith(".txt"):
                text = content.decode("utf-8")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, DOCX, or TXT.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")
            
        return text

    def _parse_pdf(self, content: bytes) -> str:
        """Parses PDF content and extracts text.

        Args:
            content (bytes): PDF file content.

        Returns:
            str: Extracted text from all pages.

        Examples:
            >>> text = processor._parse_pdf(pdf_bytes)
            >>> isinstance(text, str)
            True
        """
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def _parse_docx(self, content: bytes) -> str:
        """Parses DOCX content and extracts text.

        Args:
            content (bytes): DOCX file content.

        Returns:
            str: Extracted text from all paragraphs.

        Examples:
            >>> text = processor._parse_docx(docx_bytes)
            >>> isinstance(text, str)
            True
        """
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    def chunk_text(self, text: str) -> list[str]:
        """Splits text into semantic chunks using recursive character splitting.

        Uses LangChain's RecursiveCharacterTextSplitter with chunk size 1000
        and overlap 200 for better semantic coherence.

        Args:
            text (str): The full text to chunk.

        Returns:
            list[str]: List of text chunks.

        Examples:
            >>> chunks = processor.chunk_text("Long document text...")
            >>> len(chunks) > 0
            True
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)