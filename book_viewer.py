import pygame
import sys
import cv2
from PIL import Image
import numpy as np

class BookViewer:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PDF Book Reader")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (220, 220, 220)
        
        # Book display settings
        self.book_margin = 50
        self.page_margin = 20
        self.book_width = width - 2 * self.book_margin
        self.book_height = height - 2 * self.book_margin
        self.page_width = (self.book_width - 3 * self.page_margin) // 2
        self.page_height = self.book_height - 2 * self.page_margin
        
        # Font for text
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Page turn animation
        self.turning_page = False
        self.turn_progress = 0
        self.turn_speed = 5
        
    def pil_to_pygame(self, pil_image):
        """Convert PIL image to pygame surface"""
        # Convert PIL image to RGB if it's not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize image to fit page
        pil_image = pil_image.resize((self.page_width, self.page_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and then to pygame surface
        img_array = np.array(pil_image)
        pygame_image = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        
        return pygame_image
    
    def draw_book_background(self):
        """Draw the book background"""
        # Clear screen
        self.screen.fill(self.LIGHT_GRAY)
        
        # Draw book shadow
        shadow_rect = pygame.Rect(
            self.book_margin + 5, self.book_margin + 5,
            self.book_width, self.book_height
        )
        pygame.draw.rect(self.screen, self.GRAY, shadow_rect)
        
        # Draw book
        book_rect = pygame.Rect(
            self.book_margin, self.book_margin,
            self.book_width, self.book_height
        )
        pygame.draw.rect(self.screen, self.WHITE, book_rect)
        pygame.draw.rect(self.screen, self.BLACK, book_rect, 2)
        
        # Draw spine (center line)
        spine_x = self.book_margin + self.book_width // 2
        pygame.draw.line(
            self.screen, self.BLACK,
            (spine_x, self.book_margin),
            (spine_x, self.book_margin + self.book_height),
            2
        )
    
    def draw_page(self, page_image, is_left=True):
        """Draw a single page"""
        if page_image is None:
            return
        
        pygame_surface = self.pil_to_pygame(page_image)
        
        if is_left:
            # Left page
            page_x = self.book_margin + self.page_margin
        else:
            # Right page
            page_x = self.book_margin + self.book_width // 2 + self.page_margin
        
        page_y = self.book_margin + self.page_margin
        
        # Draw page background
        page_rect = pygame.Rect(page_x, page_y, self.page_width, self.page_height)
        pygame.draw.rect(self.screen, self.WHITE, page_rect)
        pygame.draw.rect(self.screen, self.BLACK, page_rect, 1)
        
        # Draw the page content
        self.screen.blit(pygame_surface, (page_x, page_y))
    
    def draw_page_numbers(self, left_page_num, right_page_num):
        """Draw page numbers"""
        if left_page_num is not None:
            left_text = self.font.render(str(left_page_num), True, self.BLACK)
            left_x = self.book_margin + self.page_margin + self.page_width // 2 - left_text.get_width() // 2
            left_y = self.book_margin + self.book_height - 30
            self.screen.blit(left_text, (left_x, left_y))
        
        if right_page_num is not None:
            right_text = self.font.render(str(right_page_num), True, self.BLACK)
            right_x = self.book_margin + self.book_width // 2 + self.page_margin + self.page_width // 2 - right_text.get_width() // 2
            right_y = self.book_margin + self.book_height - 30
            self.screen.blit(right_text, (right_x, right_y))
    
    def draw_instructions(self):
        """Draw usage instructions"""
        instructions = [
            "Wave your hand left or right to turn pages",
            "Press SPACE to toggle webcam view",
            "Press ESC to exit"
        ]
        
        y_offset = 10
        for instruction in instructions:
            text = self.font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
    
    def draw_webcam_view(self, webcam_frame, show_webcam=True):
        """Draw webcam view in corner"""
        if not show_webcam or webcam_frame is None:
            return
        
        # Resize webcam frame
        cam_width, cam_height = 200, 150
        cam_frame = cv2.resize(webcam_frame, (cam_width, cam_height))
        
        # Convert to pygame surface
        cam_frame_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(cam_frame_rgb.swapaxes(0, 1))
        
        # Position in top-right corner
        cam_x = self.width - cam_width - 10
        cam_y = 10
        
        # Draw webcam view
        self.screen.blit(cam_surface, (cam_x, cam_y))
        pygame.draw.rect(self.screen, self.BLACK, (cam_x, cam_y, cam_width, cam_height), 2)
    
    def render_book(self, pdf_processor, webcam_frame=None, show_webcam=True):
        """Render the complete book view"""
        # Draw book background
        self.draw_book_background()
        
        # Get current pages
        current_page = pdf_processor.current_page
        total_pages = pdf_processor.get_page_count()
        
        # For book format, show two pages side by side
        left_page = None
        right_page = None
        left_page_num = None
        right_page_num = None
        
        # Determine which pages to show
        if current_page % 2 == 0:  # Even page on right
            if current_page > 0:
                left_page = pdf_processor.pages[current_page - 1]
                left_page_num = current_page
            if current_page < total_pages:
                right_page = pdf_processor.pages[current_page]
                right_page_num = current_page + 1
        else:  # Odd page on left
            left_page = pdf_processor.pages[current_page]
            left_page_num = current_page + 1
            if current_page + 1 < total_pages:
                right_page = pdf_processor.pages[current_page + 1]
                right_page_num = current_page + 2
        
        # Draw pages
        if left_page:
            self.draw_page(left_page, is_left=True)
        if right_page:
            self.draw_page(right_page, is_left=False)
        
        # Draw page numbers
        self.draw_page_numbers(left_page_num, right_page_num)
        
        # Draw instructions
        self.draw_instructions()
        
        # Draw webcam view
        if webcam_frame is not None:
            self.draw_webcam_view(webcam_frame, show_webcam)
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    return 'toggle_webcam'
        return True
    
    def quit(self):
        """Clean up and quit"""
        pygame.quit()