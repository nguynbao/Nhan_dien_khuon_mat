import cv2
import numpy as np
import sqlite3
from define import * 

class FaceRecognizer:
    def __init__(self):
        # Khởi tạo bộ phát hiện khuôn mặt
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

        # Set text style
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 1
        self.fontcolor = Green
        self.fontcolor1 = Blue

        # Khởi tạo camera
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            print("Error: Camera not accessible.")
            return

    # Hàm lấy thông tin người dùng qua ID
    def getProfile(self, id):
        conn = sqlite3.connect("FaceBaseNew.db")
        cursor = conn.execute("SELECT * FROM People WHERE ID=?", (id,))
        profile = cursor.fetchone()
        conn.close()
        return profile

    def recognize_from_camera(self, update_callback):
        while True:
            # Đọc ảnh từ camera
            ret, img = self.cam.read()
            if not ret:
                print("Không thể đọc từ camera. Vui lòng kiểm tra kết nối.")
                break

            # Lật ảnh cho đỡ bị ngược
            img = cv2.flip(img, 1)

            # Vẽ khung chữ nhật để định vị vùng người dùng đưa mặt vào
            centerH = img.shape[0] // 2
            centerW = img.shape[1] // 2
            sizeboxW = 300
            sizeboxH = 400
            cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                        (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

            # Chuyển ảnh về xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Phát hiện các khuôn mặt trong ảnh camera
            faces = self.faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Vẽ hình chữ nhật quanh mặt
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Nhận diện khuôn mặt
                id, dist = self.model.predict(gray[y:y + h, x:x + w])

                profile = None
                if dist < numMatch:
                    profile = self.getProfile(id)

                # Hiển thị thông tin tên người hoặc Unknown
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor, 2)
                else:
                    cv2.putText(img, "Name: Unknown", (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor1, 2)

            # Gửi khung hình đã xử lý về GUI qua callback
            update_callback(img)

            # Thoát nếu cần
            if cv2.waitKey(1) == ord('q'):
                break

        self.cam.release()

    def recognize_from_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Nhận diện khuôn mặt
            id, dist = self.model.predict(gray[y:y + h, x:x + w])

            profile = None
            if dist < numMatch:
                profile = self.getProfile(id)

            if profile is not None:
                name_text = "Name: " + str(profile[1])
                cv2.putText(frame, name_text, (x, y - 10), self.fontface, 1.5, (0, 255, 0), 4)  # Tăng kích thước chữ
            else:
                cv2.putText(frame, "Name: Unknown", (x, y - 10), self.fontface, 1.5, (0, 0, 255), 2)  # Màu đỏ

            # Vẽ hình chữ nhật quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame  # Trả về khung hình đã được xử lý


