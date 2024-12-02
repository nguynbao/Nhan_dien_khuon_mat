import tkinter as tk
from tkinter import Label, Button
import numpy as np
import cv2
import sqlite3
from define import * 

class FaceRecognitionFromImage:
    def __init__(self, parent_frame, result_label):
        self.parent_frame = parent_frame
        self.result_label = result_label  # Sử dụng nhãn được truyền từ giao diện chính

        self.recognize_button = Button(parent_frame, text="Nhận diện khuôn mặt", command=self.recognize_face)
        self.recognize_button.pack(pady=10)
        self.recognize_button.pack_forget()  # Ẩn nút này khi chưa tải hình ảnh

        # Tải Haar Cascade cho phát hiện khuôn mặt
        self.faceDetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
        if self.faceDetect.empty():
            print("Error: Haar Cascade file not found.")
            return

        # Khởi tạo bộ nhận diện khuôn mặt
        self.model = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.model.read('model/trainner.yml')
        except Exception as e:
            print(f"Error loading the model: {e}")

    # Hàm lấy thông tin người dùng qua ID
    def getProfile(self, id):
        conn = sqlite3.connect("FaceBaseNew.db")
        cursor = conn.execute("SELECT * FROM People WHERE ID=?", (id,))
        profile = cursor.fetchone()
        conn.close()
        return profile

    def recognize_face(self):
        # Kiểm tra nếu có đường dẫn hình ảnh đã tải lên
        if hasattr(self.parent_frame.master, 'uploaded_image_path'):
            image_path = self.parent_frame.master.uploaded_image_path
            if image_path:
                self.recognize_from_file(image_path)
            else:
                self.result_label.config(text="Chưa có hình ảnh nào được tải lên!", fg="red")
        else:
            self.result_label.config(text="Chưa có hình ảnh nào được tải lên!", fg="red")

    # Hàm nhận diện khuôn mặt từ file ảnh
    def recognize_from_file(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            self.result_label.config(text="Không thể đọc file ảnh.", fg="red")
            return

        # Chuyển ảnh về xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện các khuôn mặt trong ảnh
        faces = self.faceDetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            self.result_label.config(text="Không phát hiện khuôn mặt nào trong ảnh.", fg="red")
            return

        # Lặp qua các khuôn mặt nhận được để hiện thông tin
        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật quanh mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Nhận diện khuôn mặt
            id, dist = self.model.predict(gray[y:y + h, x:x + w])

            profile = None

            # Nếu độ sai khác < numMatch thì lấy profile
            if dist < numMatch:
                profile = self.getProfile(id)  # Sửa lỗi gọi hàm getProfile

            # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
            if profile is not None:
                name_text = f"Name: {profile[1]}"
                cv2.putText(img, name_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Name: Unknown", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Hiển thị kết quả nhận diện khuôn mặt
        cv2.imshow('Face Recognition from File', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
