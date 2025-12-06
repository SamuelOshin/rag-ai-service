from fastapi import UploadFile, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
import io
import docx

class DocumentProcessor:
    """
    Handles file parsing and text chunking.
    Separated to allow easy addition of new file types (e.g., HTML, Markdown).
    """
    
    async def process_file(self, file: UploadFile) -> str:
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
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def _parse_docx(self, content: bytes) -> str:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    def chunk_text(self, text: str) -> list[str]:
        """
        Splits text into semantic chunks using LangChain's splitter.
        Size: 1000 chars, Overlap: 200 chars.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)