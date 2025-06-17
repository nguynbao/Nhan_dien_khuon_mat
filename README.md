# Hệ Thống Nhận Diện Khuôn Mặt (Face Recognition System)

## Giới thiệu
Đây là một ứng dụng web được xây dựng bằng Django framework, tập trung vào việc nhận diện và xác thực khuôn mặt. Hệ thống này cung cấp các chức năng nhận diện khuôn mặt thời gian thực và có thể được sử dụng trong nhiều lĩnh vực như bảo mật, điểm danh, và xác thực người dùng.

## Phân tích hệ thống

### 1. Các thành phần chính của hệ thống

#### 1.1. Quản lý người dùng (User Management)
- **DatabaseManager**: Quản lý dữ liệu người dùng trong cơ sở dữ liệu
  - Thêm người dùng mới
  - Tạo và quản lý thư mục dataset cho từng người dùng
  - Đếm số lượng ảnh của mỗi người dùng
  - Xóa người dùng và dữ liệu liên quan

#### 1.2. Xử lý và nhận diện khuôn mặt
- **FaceRecognizer**: Module xử lý nhận diện khuôn mặt thời gian thực
  - Nhận diện khuôn mặt từ video stream
  - Xử lý frame và trả về kết quả nhận diện
  
- **FaceRecognitionFromImage**: Module xử lý nhận diện từ ảnh tĩnh
  - Upload và xử lý ảnh
  - Nhận diện khuôn mặt từ ảnh đã upload
  - Hiển thị kết quả nhận diện

- **TrainModel**: Module huấn luyện mô hình
  - Huấn luyện mô hình từ dataset người dùng
  - Lưu trữ mô hình đã train

### 2. Luồng xử lý chính

#### 2.1. Quy trình thêm người dùng mới
1. Nhập thông tin người dùng (tên)
2. Tạo ID và thư mục dataset riêng
3. Chụp ảnh khuôn mặt với các hướng khác nhau:
   - Hướng thẳng (mặc định)
   - Các góc nghiêng khác nhau
4. Xử lý và lưu trữ ảnh:
   - Lật ảnh ngang (flip)
   - Vẽ khung hướng dẫn (guide rectangle)
   - Chuyển đổi sang ảnh xám
   - Phát hiện và lưu khuôn mặt

#### 2.2. Quy trình nhận diện thời gian thực
1. Capture frame từ video stream
2. Chuyển đổi frame sang base64
3. Xử lý frame:
   - Phát hiện khuôn mặt
   - So sánh với mô hình đã train
4. Trả về kết quả:
   - Ảnh đã xử lý
   - Thông tin người được nhận diện

#### 2.3. Quy trình nhận diện từ ảnh
1. Upload ảnh
2. Lưu ảnh tạm thời
3. Xử lý nhận diện
4. Hiển thị kết quả:
   - Ảnh gốc
   - Ảnh đã xử lý
   - Thông tin nhận diện

### 3. Cấu trúc dữ liệu

#### 3.1. Models
- **Person**: Thông tin người dùng
- **CapturedImage**: Lưu trữ ảnh đã chụp
- **RecognizedFace**: Kết quả nhận diện

#### 3.2. Cấu trúc thư mục
- **/media/dataSet/**: Chứa dataset của người dùng
- **/media/temp/**: Lưu trữ ảnh tạm thời
- **/model/**: Chứa file mô hình đã train (trainner.yml)

### 4. API Endpoints

#### 4.1. Quản lý người dùng
- `POST /add_user/`: Thêm người dùng mới
- `POST /delete_person/<person_id>/`: Xóa người dùng
- `POST /reset_database/`: Reset toàn bộ dữ liệu

#### 4.2. Xử lý ảnh và nhận diện
- `POST /capture_image/`: API chụp và lưu ảnh
- `POST /recognize_from_video/`: Nhận diện từ video
- `POST /upload_image/`: Upload ảnh để nhận diện
- `GET /recognize_from_image/`: Xử lý nhận diện từ ảnh đã upload

### 5. Yêu cầu kỹ thuật

#### 5.1. Thư viện chính
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- Django
- Base64 (xử lý ảnh)

#### 5.2. Cấu hình hệ thống
- Frame Width: Định nghĩa trong FRAME_WIDTH
- Frame Height: Định nghĩa trong FRAME_HEIGHT
- Hỗ trợ định dạng ảnh: JPEG, PNG

### 6. Bảo mật
- CSRF protection cho các form
- Xác thực cho API endpoints
- Kiểm tra và xử lý file upload an toàn
- Quản lý session cho upload ảnh

## Cài đặt và Triển khai
[Chi tiết hướng dẫn cài đặt...]

## Đóng góp
Mọi đóng góp cho dự án đều được hoan nghênh. Vui lòng tạo pull request hoặc báo cáo lỗi thông qua mục Issues.

## Giấy phép
[MIT License](LICENSE)

## Tác giả và Liên hệ
[Thông tin liên hệ...]

## Yêu cầu hệ thống
- Python 3.6+
- OpenCV và OpenCV-contrib
- Django
- Pillow
- NumPy

## Cài đặt

### 1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị cơ sở dữ liệu:
```bash
python manage.py migrate
```

### 3. Khởi chạy ứng dụng:
```bash
python manage.py runserver
```

Sau đó truy cập vào địa chỉ: http://127.0.0.1:8000/

## Hướng dẫn sử dụng

### Thêm người dùng mới
1. Truy cập vào "Thêm người dùng" trong menu
2. Nhập tên người dùng
3. Nhấn "Tiếp tục" để chụp ảnh khuôn mặt
4. Cho phép truy cập camera
5. Nhấn nút "Chụp ảnh" để bắt đầu chụp khuôn mặt
6. Di chuyển khuôn mặt qua các hướng khác nhau để có đủ dữ liệu
7. Sau khi chụp đủ ảnh, nhấn "Hoàn thành"

### Huấn luyện mô hình
1. Truy cập vào "Huấn luyện mô hình" trong menu
2. Nhấn "Bắt đầu huấn luyện"
3. Đợi quá trình huấn luyện hoàn tất
4. Sau khi huấn luyện xong, có thể tiến hành nhận diện

### Nhận diện khuôn mặt từ webcam
1. Truy cập vào "Nhận diện" trong menu
2. Cho phép truy cập camera
3. Nhấn "Bắt đầu nhận diện" để bắt đầu
4. Kết quả nhận diện sẽ hiển thị bên cạnh

### Nhận diện khuôn mặt từ ảnh
1. Truy cập vào "Tải ảnh lên" trong menu
2. Chọn ảnh từ máy tính
3. Nhấn "Tải lên"
4. Nhấn "Nhận diện" để tiến hành nhận diện khuôn mặt trong ảnh
5. Kết quả sẽ hiển thị cùng với ảnh đã xử lý

## Cấu trúc thư mục
- `face_app/`: Thư mục chứa mã nguồn ứng dụng
  - `utils/`: Chứa các tệp tiện ích xử lý khuôn mặt
  - `templates/`: Chứa các tệp HTML giao diện
  - `model/`: Chứa các mô hình nhận diện và Haar Cascade
- `media/`: Thư mục lưu trữ hình ảnh người dùng
  - `dataSet/`: Thư mục chứa dữ liệu khuôn mặt người dùng
- `face_recognition_project/`: Thư mục cấu hình Django

## Lưu ý
- Đảm bảo ánh sáng đầy đủ khi chụp ảnh và nhận diện
- Nên chụp ít nhất 50 ảnh cho mỗi người dùng để tăng độ chính xác
- Huấn luyện lại mô hình sau khi thêm người dùng mới

---

# Face Recognition System

## Introduction
Face recognition system using OpenCV and Django, allowing user addition, face image capture, model training, and face recognition from webcam or static images.

## System Requirements
- Python 3.6+
- OpenCV and OpenCV-contrib
- Django
- Pillow
- NumPy

## Installation

### 1. Install required libraries:
```bash
pip install -r requirements.txt
```

### 2. Prepare database:
```bash
python manage.py migrate
```

### 3. Run the application:
```bash
python manage.py runserver
```

Then access: http://127.0.0.1:8000/

## User Guide

### Add New User
1. Access "Add User" in the menu
2. Enter user's name
3. Click "Continue" to capture face images
4. Allow camera access
5. Click "Capture" button to start capturing face
6. Move your face in different directions to get enough data
7. After capturing enough images, click "Finish"

### Train Model
1. Access "Train Model" in the menu
2. Click "Start Training"
3. Wait for the training process to complete
4. After training is done, you can proceed with recognition

### Face Recognition from Webcam
1. Access "Recognition" in the menu
2. Allow camera access
3. Click "Start Recognition" to begin
4. Recognition results will be displayed on the side

### Face Recognition from Image
1. Access "Upload Image" in the menu
2. Select image from computer
3. Click "Upload"
4. Click "Recognize" to perform face recognition on the image
5. Results will be displayed along with the processed image

## Directory Structure
- `face_app/`: Contains application source code
  - `utils/`: Contains face processing utility files
  - `templates/`: Contains HTML interface files
  - `model/`: Contains recognition models and Haar Cascade
- `media/`: Directory for storing user images
  - `dataSet/`: Directory containing user face data
- `face_recognition_project/`: Django configuration directory

## Notes
- Ensure adequate lighting when capturing images and recognizing
- Capture at least 50 images for each user to increase accuracy
- Retrain the model after adding new users 