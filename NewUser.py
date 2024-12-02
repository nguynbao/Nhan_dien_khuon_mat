import cv2
import sqlite3
import os
import time

from queue import Queue
from define import *  

class DatabaseManager:
    def __init__(self, db_file="FaceBaseNew.db"):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()
        self.queue = Queue()  # Queue để lưu trữ khung hình
        self.is_capturing = False  # Biến trạng thái chụp
        self.camera_thread = None  # Để lưu thread camera
        self.result_label = None  # Label để thông báo trạng thái
        self.label = None  # Label để hiển thị hình ảnh
        self.user_folder = None  # Thư mục lưu ảnh

    def create_table(self):
        self.cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS People ( 
                ID INTEGER PRIMARY KEY AUTOINCREMENT, 
                Name TEXT 
            ) 
        ''')
        self.conn.commit()

    def insert_or_update(self, name):
        # Chỉ cần thêm tên mới và tự động tăng ID
        self.cursor.execute("INSERT INTO People (Name) VALUES (?)", (name,))
        self.conn.commit()

        # Lấy ID tự động tăng của bản ghi vừa thêm
        return self.cursor.lastrowid
   
    def close(self):
        self.conn.close()

    def create_dataset_folder(self, id, name):
        dataset_folder = 'dataSet'
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        user_folder = os.path.join(dataset_folder, f"{name}_{id}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        return user_folder

    def capture_faces(self, user_folder, id): 
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
        if not cam.isOpened():
            print("Error: Camera not accessible.")
            return

        # Chia số lượng ảnh cần chụp cho từng hướng
        images_per_direction = numI // 5  # Số lượng ảnh cho mỗi hướng
        sampleNum = 0
        direction_counts = {  # Số lượng ảnh cần chụp cho mỗi hướng
            "thang": 0,
            "trai": 0,
            "phai": 0,
            "len": 0,
            "xuong": 0
        }
        directions = ["thang", "trai", "phai", "len", "xuong"]
        direction_index = 0  # Bắt đầu từ hướng đầu tiên
        current_direction = directions[direction_index] # Hướng hiện tại

        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)

            # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
            centerH = img.shape[0] // 2
            centerW = img.shape[1] // 2
            sizeboxW = sW
            sizeboxH = sH
            cv2.rectangle(
                img,
                (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                (centerW + sizeboxW // 2, centerH + sizeboxH // 2),
                White,
                5,
            )

            # Đưa ảnh về ảnh xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Hiển thị hướng dẫn xoay mặt
            cv2.putText(
                img,
                current_direction,
                (centerW - 100, centerH - sizeboxH // 2 - 20),  # Vị trí hiển thị hướng dẫn
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Màu đỏ để làm nổi bật
                2,
                cv2.LINE_AA,
            )

            # Nhận diện khuôn mặt
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                # Vẽ hình chữ nhật quanh mặt nhận được
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Kiểm tra nếu số lượng ảnh cho hướng hiện tại chưa đủ
                if direction_counts[current_direction] < images_per_direction:
                    direction_counts[current_direction] += 1
                    sampleNum += 1

                    # Ghi dữ liệu khuôn mặt vào thư mục user_folder
                    cv2.imwrite(os.path.join(user_folder, f"User.{id}.{sampleNum}.jpg"), gray[y:y + h, x:x + w])

                    # Khi đạt số ảnh yêu cầu cho hướng hiện tại, tạm dừng 3 giây và chuyển sang hướng tiếp theo
                    if direction_counts[current_direction] >= images_per_direction:
                        # Hiển thị thông báo chuyển hướng
                        next_direction = directions[(direction_index + 1) % len(directions)]
                        cv2.putText(
                            img,
                            f"Hay nhin {next_direction}",
                            (centerW - 200, img.shape[0] - 60),  # Vị trí thông báo
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),  # Màu vàng để thông báo
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("frame", img)
                        cv2.waitKey(1)  # Cập nhật khung hình

                        # Tạm dừng 3 giây
                        time.sleep(3)

                        # Chuyển sang hướng tiếp theo
                        direction_index = (direction_index + 1) % len(directions)
                        current_direction = directions[direction_index]

            # Hiển thị số lượng ảnh đã chụp ở phía dưới màn hình
            cv2.putText(
                img,
                f"Images Captured: {sampleNum}/{numI}",
                (10, img.shape[0] - 20),  # Vị trí hiển thị phía dưới
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Kích thước chữ
                (0, 255, 0),  # Màu chữ (màu xanh lá cây)
                2,  # Độ dày chữ
                cv2.LINE_AA
            )

            cv2.imshow("frame", img)
            if cv2.waitKey(100) & 0xFF == ord("q") or sampleNum >= numI:
                break

        cam.release()
        cv2.destroyAllWindows()

    

        
