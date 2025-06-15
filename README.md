# Hệ thống nhận diện khuôn mặt

## Giới thiệu
Hệ thống nhận diện khuôn mặt sử dụng OpenCV và Django, cho phép thêm người dùng, chụp ảnh khuôn mặt, huấn luyện mô hình và nhận diện khuôn mặt từ webcam hoặc ảnh tĩnh.

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