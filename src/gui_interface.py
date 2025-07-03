import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from real_time_detection import RealTimeDetector


class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition System")
        self.root.geometry("800x600")

        self.detector = RealTimeDetector()
        self.cap = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Sign Language Recognition System",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Detection",
                                       command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Detection",
                                      command=self.stop_detection)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear Text",
                                       command=self.clear_text)
        self.clear_button.grid(row=0, column=2, padx=5)

        # Text display
        text_frame = ttk.LabelFrame(main_frame, text="Recognized Text", padding="5")
        text_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.text_display = tk.Text(text_frame, height=8, width=70)
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_display.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.text_display.configure(yscrollcommand=scrollbar.set)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

    def start_detection(self):
        """Start the detection process"""
        if not self.detector.load_model():
            messagebox.showerror("Error", "Could not load model. Please train the model first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Detection running...")

        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        if self.cap:
            self.cap.release()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Detection stopped")

    def clear_text(self):
        """Clear the text display"""
        self.text_display.delete(1.0, tk.END)
        if hasattr(self.detector, 'sentence'):
            self.detector.sentence = ""
            self.detector