import cv2
import os
import numpy as np
from PIL import Image

# Khởi tạo mô hình nhận diện khuôn mặt
model = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Danh sách để lưu trữ các khuôn mặt và ID
    faceSamples = []
    Ids = []
    
    # Duyệt qua tất cả các thư mục và tệp con trong đường dẫn đã cho
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                imagePath = os.path.join(root, file)
                print(f"Processing file: {imagePath}")

                # Đọc hình ảnh và chuyển đổi sang ảnh xám
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')

                # Lấy ID từ tên thư mục cha (giả sử tên thư mục là "Name_ID")
                folder_name = os.path.basename(root)
                try:
                    Id = int(folder_name.split("_")[1])  # Giả sử định dạng là Name_ID
                except ValueError:
                    print(f"Invalid folder name format for {folder_name}. Skipping...")
                    continue

                # Phát hiện khuôn mặt trong ảnh
                faces = detector.detectMultiScale(imageNp)

                # Nếu có khuôn mặt, thêm vào danh sách
                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y+h, x:x+w])
                    Ids.append(Id)

    return faceSamples, Ids

# Lấy các khuôn mặt và ID từ thư mục dataSet
faceSamples, Ids = getImagesAndLabels('dataSet')

# Huấn luyện mô hình với các khuôn mặt và ID
model.train(faceSamples, np.array(Ids))

# Lưu mô hình
model.save('model/trainner.yml')

print("Trained!")
