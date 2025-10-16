import cv2
import sys
import os
import argparse
from pdf_reader import PDFProcessor
from gesture_detector import GestureDetector
from book_viewer import BookViewer

class PDFBookReader:
    def __init__(self, pdf_path):
        # Initialize components
        self.pdf_processor = PDFProcessor(pdf_path)
        self.gesture_detector = GestureDetector()
        self.book_viewer = BookViewer()
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
        
        # App state
        self.running = True
        self.show_webcam = True
        
        print(f"Loaded PDF with {self.pdf_processor.get_page_count()} pages")
        print("Use hand gestures to turn pages:")
        print("- Swipe LEFT to advance to next page")
        print("- Swipe RIGHT to go back to previous page")
        print("- Press SPACE to toggle webcam view")
        print("- Press ESC to exit")
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            gesture, processed_frame = self.gesture_detector.detect_gesture(frame.copy())
            
            # Handle gestures
            # Left swipe = advance (turn page forward, like flipping left to right)
            # Right swipe = go back (turn page backward, like flipping right to left)
            if gesture == 'left':
                if self.pdf_processor.next_page():
                    print(f"Next page: {self.pdf_processor.current_page + 1}")
            elif gesture == 'right':
                if self.pdf_processor.previous_page():
                    print(f"Previous page: {self.pdf_processor.current_page + 1}")
            
            # Handle pygame events
            event_result = self.book_viewer.handle_events()
            if event_result == False:
                self.running = False
            elif event_result == 'toggle_webcam':
                self.show_webcam = not self.show_webcam
                print(f"Webcam view: {'ON' if self.show_webcam else 'OFF'}")
            
            # Render the book
            webcam_frame = processed_frame if self.show_webcam else None
            self.book_viewer.render_book(
                self.pdf_processor, 
                webcam_frame, 
                self.show_webcam
            )
            
            # Control frame rate
            clock.tick(30)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.book_viewer.quit()
        print("Application closed")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='PDF Book Reader with Gesture Control')
    parser.add_argument('pdf_path', help='Path to the PDF file to read')
    parser.add_argument('--width', type=int, default=1200, help='Window width (default: 1200)')
    parser.add_argument('--height', type=int, default=800, help='Window height (default: 800)')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        sys.exit(1)
    
    # Check if it's a PDF file
    if not args.pdf_path.lower().endswith('.pdf'):
        print("Error: File must be a PDF")
        sys.exit(1)
    
    try:
        # Create and run the application
        app = PDFBookReader(args.pdf_path)
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Import pygame here to avoid issues if not installed
    try:
        import pygame
    except ImportError:
        print("Error: pygame is required. Install it with: pip install pygame")
        sys.exit(1)
    
    main()