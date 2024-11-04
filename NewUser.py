import cv2
import sqlite3
import os
from define import *

# Kết nối đến file SQLite và tạo bảng "People" nếu chưa tồn tại
conn = sqlite3.connect("FaceBaseNew.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS People (
        ID INTEGER PRIMARY KEY,
        Name TEXT
    )
''')
conn.commit()
conn.close()

print("Cơ sở dữ liệu và bảng 'People' đã được tạo thành công.")

# Hàm cập nhật tên và ID vào CSDL
def insertOrUpdate(id, name):
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.cursor()

    # Kiểm tra xem ID đã tồn tại trong bảng chưa
    cursor.execute("SELECT * FROM People WHERE ID = ?", (id,))
    isRecordExist = cursor.fetchone() is not None

    # Nếu bản ghi đã tồn tại, cập nhật thông tin
    if isRecordExist:
        cursor.execute("UPDATE People SET Name = ? WHERE ID = ?", (name, id))
    else:
        # Nếu chưa tồn tại, thêm bản ghi mới
        cursor.execute("INSERT INTO People (ID, Name) VALUES (?, ?)", (id, name))

    # Lưu thay đổi và đóng kết nối
    conn.commit()
    conn.close()

# Hàm tạo thư mục để lưu ảnh
def create_dataset_folder(id, name):
    # Tạo folder dataset nếu chưa có
    dataset_folder = 'dataSet'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Tạo folder cho người dùng
    user_folder = os.path.join(dataset_folder, f"{name}_{id}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    return user_folder

# Bắt đầu xử lý nhận diện khuôn mặt
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

while True:
    id = input("Nhập mã nhân viên (hoặc nhập 'exit' để thoát): ")
    if id.lower() == 'exit':  # Kiểm tra nếu người dùng muốn thoát
        break

    name = input("Nhập tên nhân viên: ")
    print("Bắt đầu chụp ảnh nhân viên, nhấn q để thoát!")
    
    # Cập nhật hoặc thêm thông tin vào CSDL
    insertOrUpdate(id, name)

    # Tạo thư mục để lưu ảnh
    user_folder = create_dataset_folder(id, name)

    sampleNum = 0  # Đặt lại sampleNum cho mỗi người dùng mới

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

        # Nhận diện khuôn mặt
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            # Vẽ hình chữ nhật quanh mặt nhận được
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1

            # Ghi dữ liệu khuôn mặt vào thư mục user_folder
            cv2.imwrite(os.path.join(user_folder, f"User.{id}.{sampleNum}.jpg"), gray[y:y + h, x:x + w])

        cv2.imshow("frame", img)
        if cv2.waitKey(100) & 0xFF == ord("q") or sampleNum >= numI:
            break

# Giải phóng camera và đóng các cửa sổ
cam.release()
cv2.destroyAllWindows()
