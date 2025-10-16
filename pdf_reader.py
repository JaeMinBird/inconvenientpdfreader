import PyPDF2
from PIL import Image
import io
import os

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages = []
        self.current_page = 0
        self.load_pdf()
    
    def load_pdf(self):
        """Extract pages from PDF and convert to images"""
        try:
            import fitz  # PyMuPDF for better image conversion
            pdf_document = fitz.open(self.pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                self.pages.append(img)
            
            pdf_document.close()
            
        except ImportError:
            # Fallback to PyPDF2 (text-only, requires additional setup for images)
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    # For now, create a placeholder image with page text
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    img = self.create_text_image(text, page_num + 1)
                    self.pages.append(img)
    
    def create_text_image(self, text, page_num):
        """Create an image from text (fallback method)"""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', (800, 1000), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        
        # Add page number
        draw.text((10, 10), f"Page {page_num}", fill='black', font=font)
        
        # Add text content
        lines = text.split('\n')
        y_position = 50
        for line in lines[:50]:  # Limit to 50 lines
            if y_position > 950:
                break
            draw.text((10, y_position), line[:80], fill='black', font=font)
            y_position += 20
        
        return img
    
    def get_current_page(self):
        """Get the current page image"""
        if 0 <= self.current_page < len(self.pages):
            return self.pages[self.current_page]
        return None
    
    def next_page(self):
        """Move to next 2-page spread (turn page forward)"""
        # Move by 2 to get to the next spread
        if self.current_page < len(self.pages) - 1:
            self.current_page = min(self.current_page + 2, len(self.pages) - 1)
            return True
        return False

    def previous_page(self):
        """Move to previous 2-page spread (turn page backward)"""
        # Move by 2 to get to the previous spread
        if self.current_page > 0:
            self.current_page = max(self.current_page - 2, 0)
            return True
        return False
    
    def get_page_count(self):
        """Get total number of pages"""
        return len(self.pages)