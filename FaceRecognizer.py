import cv2
import numpy as np
import sqlite3

from define import *  # Đảm bảo bạn đã định nghĩa Green và Blue trong file define.py

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Khởi tạo bộ nhận diện khuôn mặt
model = cv2.face.LBPHFaceRecognizer_create()
model.read('model/trainner.yml')

# Set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = Green
fontcolor1 = Blue

# Hàm lấy thông tin người dùng qua ID
def getProfile(id):
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.execute("SELECT * FROM People WHERE ID=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

# Hàm nhận diện khuôn mặt từ file ảnh
def recognize_from_file(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc file ảnh.")
        return
    
    # Chuyển ảnh về xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    # Lặp qua các khuôn mặt nhận được để hiện thông tin
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Nhận diện khuôn mặt
        id, dist = model.predict(gray[y:y + h, x:x + w])

        profile = None

        # Nếu độ sai khác < 70 thì lấy profile
        if dist < 1000:
            profile = getProfile(id)

        # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
        if profile is not None:
            cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontface, fontscale, fontcolor, 2)
        else:
            cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)

    # Hiển thị kết quả
    cv2.imshow('Face Recognition from File', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Menu cho người dùng
while True:
    choice = input("Nhập 'c' để sử dụng camera, 'f' để nhận diện từ file, hoặc 'q' để thoát: ")
    
    if choice == 'c':
        while True:
            # Đọc ảnh từ camera
            ret, img = cam.read()
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
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            # Lặp qua các khuôn mặt nhận được để hiện thông tin
            for (x, y, w, h) in faces:
                # Vẽ hình chữ nhật quanh mặt
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Nhận diện khuôn mặt
                id, dist = model.predict(gray[y:y + h, x:x + w])

                profile = None

                # Nếu độ sai khác < 70 thì lấy profile
                if dist < 70:
                    profile = getProfile(id)

                # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontface, fontscale, fontcolor, 2)
                else:
                    cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)

            cv2.imshow('Face Recognition', img)
            
            # Nếu nhấn q thì thoát
            if cv2.waitKey(1) == ord('q'):
                break

    elif choice == 'f':
        image_path = input("Nhập đường dẫn đến file ảnh: ")
        recognize_from_file(image_path)

    elif choice == 'q':
        break

cam.release()
cv2.destroyAllWindows()
