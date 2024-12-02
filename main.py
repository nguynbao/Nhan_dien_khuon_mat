import os
import tkinter as tk
import cv2
import numpy as np
from tkinter import Frame, Label, Button, Entry, filedialog, messagebox, ttk
from PIL import Image, ImageTk
from queue import Queue
import threading
from FaceRecognizer import FaceRecognizer
from NewUser import DatabaseManager
from TrainModel import TrainModel
from FaceRecognitionFromImage import FaceRecognitionFromImage


class ModernFaceRecognitionApp:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.face_trainer = TrainModel()
        self.face_recognizer = FaceRecognizer()
        self.face_recognition_from_image = None


        # GUI configuration
        self.root = tk.Tk()
        self.root.title("Hệ thống nhận diện khuôn mặt")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f4f8")

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TLabel", background="#ffffff", font=("Segoe UI", 12))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))

        self.camera_thread = None
        self.is_recognizing_video = False

        self.create_widgets()

    def create_widgets(self):
        # Main container
        self.main_container = Frame(self.root, bg="#ffffff", padx=20, pady=20)
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)

        # Title
        title_frame = Frame(self.main_container, bg="#ffffff")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT", 
                  style="Title.TLabel", foreground="#2c3e50").pack()

        # Main content frame
        content_frame = Frame(self.main_container, bg="#ffffff")
        content_frame.pack(expand=True, fill=tk.BOTH)

        # Left panel
        left_panel = Frame(content_frame, bg="#ffffff", width=400)
        left_panel.pack(side=tk.LEFT, padx=(0, 20), fill=tk.Y)

        ttk.Label(left_panel, text="Nhập tên:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.name_entry = Entry(left_panel, font=("Segoe UI", 12), borderwidth=2, relief=tk.SOLID)
        self.name_entry.pack(fill=tk.X, pady=(0, 10))

        button_frame = Frame(left_panel, bg="#ffffff")
        button_frame.pack(fill=tk.X, pady=10)

        button_style = {
            "font": ("Segoe UI", 10),
            "bg": "#3498db", 
            "fg": "white", 
            "activebackground": "#2980b9",
            "relief": tk.FLAT,
            "padx": 15,
            "pady": 8
        }

        Button(button_frame, text="Add/Update", 
               command=self.add_or_update, **button_style).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        Button(button_frame, text="Train Model", 
               command=self.train_model, **button_style).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        Button(left_panel, text="Upload Image", 
               command=self.upload_image, **button_style).pack(fill=tk.X, pady=10)

        Button(left_panel, text="Recognize Face", 
               command=self.start_recognition, **button_style).pack(fill=tk.X, pady=10)

        self.result_label = ttk.Label(left_panel, text="", style="TLabel", foreground="#27ae60")
        self.result_label.pack(pady=10)

        # Right panel
        right_panel = Frame(content_frame, bg="#f1f3f5", width=600)
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.image_label = Label(right_panel, bg="#f1f3f5")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.recognize_button = Button(right_panel, text="Nhận diện từ ảnh", 
                               command=self.recognize_from_image, **button_style)
        self.recognize_button.pack(fill=tk.X, padx=20, pady=10)

        # Đặt nút ban đầu là disabled
        self.recognize_button.config(state=tk.DISABLED)

    def update_frame(self, frame):
        # Chuyển đổi khung hình OpenCV sang định dạng phù hợp với Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def update_image_label(self, frame):
        """Cập nhật khung hình lên giao diện chính với tỷ lệ đúng mà không đổi màu"""
        img = Image.fromarray(frame)  # Sử dụng frame gốc từ OpenCV (BGR)
        
        # Tính toán tỷ lệ resize ảnh để không bị méo
        img_width, img_height = img.size
        max_width = 600
        max_height = 400
        aspect_ratio = img_width / img_height

        # Điều chỉnh kích thước ảnh
        if img_width > img_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
            if new_width > max_width:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def start_recognition(self):
        if not self.is_recognizing_video:
            self.is_recognizing_video = True
            self.camera_thread = threading.Thread(
                target=self.face_recognizer.recognize_from_camera, 
                args=(self.update_frame,)
            )
            self.camera_thread.start()
            self.result_label.config(text="Nhận diện video đã bắt đầu...")
        else:
            messagebox.showinfo("Thông báo", "Đang chụp ảnh. Vui lòng chờ.")

    def add_or_update(self):
        name_input = self.name_entry.get()
        if name_input:
            new_id = self.db_manager.insert_or_update(name_input)
            user_folder = self.db_manager.create_dataset_folder(new_id, name_input)
            
            # Khởi tạo luồng chụp ảnh
            if not self.db_manager.is_capturing:
                self.db_manager.is_capturing = True
                threading.Thread(
                    target=self.db_manager.capture_faces,
                    args=(user_folder, new_id)
                ).start()
                self.result_label.config(text="Đang chụp ảnh...")
            else:
                messagebox.showinfo("Thông báo", "Đang chụp ảnh. Vui lòng chờ.")
        else:
            messagebox.showerror("Lỗi", "Vui lòng nhập tên.")

    def train_model(self):
        self.face_trainer.train()
        self.result_label.config(text="Mô hình huấn luyện thành công!")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh", 
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            img = Image.open(file_path)
            img_width, img_height = img.size  

            max_width = 600
            max_height = 400
            aspect_ratio = img_width / img_height

            # Nếu tỷ lệ chiều rộng lớn hơn tỷ lệ chiều cao
            if img_width > img_height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(max_height * aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(max_width / aspect_ratio)

            # Resize ảnh giữ tỷ lệ gốc
            img = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.result_label.config(text=f"Ảnh đã tải: {os.path.basename(file_path)}")

            # Lưu file_path vào thuộc tính của đối tượng
            self.uploaded_image_path = file_path

            # Kích hoạt nút "Recognize Face"
            self.recognize_button.config(state=tk.NORMAL)

    def recognize_from_image(self):
        if hasattr(self, 'uploaded_image_path'):
            file_path = self.uploaded_image_path
            
            if file_path:
                frame = cv2.imread(file_path)
                self.face_recognizer.recognize_from_frame(frame)  # Nhận diện khuôn mặt từ ảnh

                # Resize ảnh để không bị biến dạng
                frame_resized = self.resize_image(frame, 600, 400)
                self.update_image_label(frame_resized)  # Cập nhật ảnh lên giao diện
                self.result_label.config(text=f"Ảnh đã nhận diện: {os.path.basename(file_path)}")

    def resize_image(self, frame, max_width, max_height):
        """Resize ảnh giữ tỷ lệ gốc và phù hợp với kích thước khung"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height

        if img_width > img_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
            if new_width > max_width:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        return np.array(img)  # Trả về ảnh đã resize dưới dạng numpy array

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ModernFaceRecognitionApp()
    app.run()
