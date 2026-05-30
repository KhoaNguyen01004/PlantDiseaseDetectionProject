# NHIỆM VỤ

Hãy tạo file `report.md`.

Đây là báo cáo đồ án môn học:

**THỊ GIÁC MÁY TÍNH TRONG TƯƠNG TÁC NGƯỜI MÁY**

Dự án hiện tại là hệ thống nhận diện bệnh trên lá cây bằng Deep Learning và triển khai trên thiết bị Android với khả năng hoạt động ngoại tuyến.

---

# YÊU CẦU QUAN TRỌNG

## 1. Ngôn ngữ

Toàn bộ báo cáo phải được viết bằng:

**Tiếng Việt học thuật, trang trọng, chuyên nghiệp.**

Không sử dụng:

* Tôi
* Chúng tôi
* Nhóm em
* Nhóm tác giả

Ưu tiên sử dụng:

* Hệ thống
* Mô hình đề xuất
* Ứng dụng được xây dựng
* Dự án
* Giải pháp được đề xuất
* Kết quả thực nghiệm
* Hệ thống được triển khai

Báo cáo phải được viết theo văn phong ngôi thứ ba.

---

## 2. Không được bịa dữ liệu

Khi chưa có dữ liệu runtime:

KHÔNG tự tạo số liệu.

Thay vào đó ghi:

```markdown
TODO: Bổ sung độ chính xác sau khi huấn luyện

TODO: Bổ sung hình ảnh Confusion Matrix

TODO: Bổ sung kết quả kiểm thử trên Android
```

---

## 3. Phải phân tích source code

Trước khi viết báo cáo:

Đọc và phân tích:

* README.md
* SourceCode/README.md
* PROJECT_SUMMARY.md
* VALIDATION_GUIDE.md
* CHANGELOG.md
* configs/config.yaml
* train.py
* inference.py
* gradcam.py
* quality_validator.py
* metadata.py
* Android application
* TFLite export pipeline

Nội dung báo cáo phải phản ánh chính xác hệ thống thực tế.

Không được viết theo mẫu chung chung.

---

# CẤU TRÚC BÁO CÁO

# TRANG BÌA

Tạo placeholder:

* Tên trường
* Khoa
* Môn học
* Tên đề tài
* Giảng viên hướng dẫn
* Sinh viên thực hiện
* MSSV
* Lớp
* Năm học

---

# LỜI MỞ ĐẦU

Viết hoàn chỉnh.

Nội dung gồm:

* Bối cảnh nghiên cứu
* Vai trò của thị giác máy tính trong nông nghiệp
* Tính cấp thiết của việc phát hiện bệnh cây trồng
* Động lực thực hiện đề tài

---

# CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI

## 1.1 Đặt vấn đề

Viết chi tiết.

## 1.2 Mục tiêu đề tài

### Mục tiêu tổng quát

### Mục tiêu cụ thể

---

## 1.3 Phạm vi nghiên cứu

Nêu rõ:

* Những gì được thực hiện
* Những gì chưa thực hiện

---

## 1.4 Ý nghĩa khoa học và thực tiễn

Viết đầy đủ.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1 Tổng quan về Thị giác máy tính

## 2.2 Deep Learning trong nhận dạng ảnh

## 2.3 Bài toán phân loại bệnh trên lá cây

## 2.4 Mạng EfficientNet-B2

Giải thích:

* Kiến trúc
* Nguyên lý hoạt động
* Lý do lựa chọn

Chèn:

```markdown
Hình 2.x. Kiến trúc EfficientNet-B2

TODO: Chèn hình
```

---

## 2.5 TensorFlow Lite

## 2.6 Grad-CAM

Giải thích:

* Khái niệm
* Nguyên lý
* Vai trò trong giải thích kết quả

Chèn:

```markdown
Hình 2.x. Ví dụ Grad-CAM

TODO: Chèn hình
```

---

## 2.7 Android On-device AI

---

# CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1 Kiến trúc tổng thể

Chèn:

```markdown
Hình 3.1. Kiến trúc tổng thể hệ thống

TODO: Chèn sơ đồ
```

Mô tả chi tiết luồng dữ liệu.

---

## 3.2 Phân tích yêu cầu

### Yêu cầu chức năng

Tạo bảng.

### Yêu cầu phi chức năng

Tạo bảng.

---

## 3.3 Thiết kế luồng xử lý

Chèn:

```markdown
Hình 3.x. Luồng xử lý hệ thống

TODO: Chèn sơ đồ
```

---

## 3.4 Thiết kế cơ sở dữ liệu và metadata

Nếu có.

---

# CHƯƠNG 4. XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG

⚠️ Đây phải là chương dài nhất.

---

## 4.1 Cấu trúc thư mục dự án

```markdown
Hình 4.1. Cấu trúc dự án

TODO: Chèn hình
```

---

## 4.2 Bộ dữ liệu

Mô tả:

* PlantVillage
* Cấu trúc dữ liệu
* Nhãn dữ liệu

Tạo bảng thống kê.

```markdown
Bảng 4.x. Thống kê dữ liệu

TODO: Điền số liệu
```

---

## 4.3 Tiền xử lý dữ liệu

Mô tả:

* Resize
* Normalize
* Data Loader

---

## 4.4 Tăng cường dữ liệu

Mô tả từng kỹ thuật augmentation.

Tạo bảng:

| Kỹ thuật | Mục đích |
| -------- | -------- |

---

## 4.5 Xây dựng mô hình EfficientNet-B2

Mô tả chi tiết.

---

## 4.6 Quy trình huấn luyện

Mô tả:

* Loss
* Optimizer
* Scheduler
* Checkpoint

---

## 4.7 Module Quality Validation

Mô tả:

* Blur Detection
* Brightness Validation
* Resolution Validation

---

## 4.8 Module Grad-CAM

Mô tả chi tiết.

---

## 4.9 Metadata Management

---

## 4.10 Pipeline xuất mô hình

PyTorch

↓

ONNX

↓

TensorFlow Lite

Chèn:

```markdown
Hình 4.x. Pipeline chuyển đổi mô hình

TODO: Chèn sơ đồ
```

---

## 4.11 Ứng dụng Android

Mô tả:

* Giao diện
* Chức năng
* Luồng hoạt động

Chèn:

```markdown
Hình 4.x. Màn hình chính

TODO: Chèn ảnh

Hình 4.x. Màn hình nhận diện

TODO: Chèn ảnh

Hình 4.x. Màn hình kết quả

TODO: Chèn ảnh
```

---

# CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 5.1 Môi trường thực nghiệm

Tạo bảng.

---

## 5.2 Kết quả huấn luyện

```markdown
TODO: Chèn biểu đồ Loss

TODO: Chèn biểu đồ Accuracy

TODO: Chèn kết quả cuối cùng
```

---

## 5.3 Đánh giá mô hình

Tạo bảng:

* Accuracy
* Precision
* Recall
* F1-score

---

## 5.4 Confusion Matrix

```markdown
TODO: Chèn hình Confusion Matrix
```

---

## 5.5 Đánh giá Grad-CAM

```markdown
TODO: Chèn các ví dụ Grad-CAM
```

---

## 5.6 Đánh giá trên Android

Tạo bảng:

| Tiêu chí           | Giá trị |
| ------------------ | ------- |
| Kích thước mô hình | TODO    |
| Thời gian suy luận | TODO    |
| RAM sử dụng        | TODO    |

---

## 5.7 Thảo luận kết quả

Viết hoàn chỉnh.

---

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết luận

Viết hoàn chỉnh.

## 6.2 Hạn chế

Viết thực tế.

## 6.3 Hướng phát triển

Bao gồm:

* Thu thập dữ liệu thực tế
* Tối ưu Android
* Bổ sung tập bệnh
* Leaf Detection
* Nâng cao độ chính xác

---

# TÀI LIỆU THAM KHẢO

Tạo placeholder.

---

# PHỤ LỤC

## Phụ lục A

Cấu hình hệ thống.

## Phụ lục B

Các lệnh chạy chương trình.

## Phụ lục C

Cấu trúc thư mục dự án.

---

# YÊU CẦU CHẤT LƯỢNG

* Báo cáo phải đủ nội dung để chuyển thành báo cáo 20–30 trang.
* Mỗi mục phải có nhiều đoạn văn hoàn chỉnh.
* Không viết kiểu gạch đầu dòng ngắn.
* Chương 4 phải chi tiết nhất.
* Giải thích rõ các quyết định kỹ thuật dựa trên source code thực tế.
* Không được viết hời hợt.
* Kết quả cuối cùng lưu vào file `report.md`.
