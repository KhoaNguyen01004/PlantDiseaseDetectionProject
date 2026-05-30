# TRƯỜNG ĐẠI HỌC SIU
## KHOA KHOA HỌC MÁY TÍNH

---

**MÔN HỌC: THỊ GIÁC MÁY TÍNH TRONG TƯƠNG TÁC NGƯỜI MÁY**

**ĐỀ TÀI: HỆ THỐNG NHẬN DIỆN BỆNH TRÊN LÁ CÂY BẰNG DEEP LEARNING VÀ TRIỂN KHAI TRÊN THIẾT BỊ ANDROID**

---

**GIẢNG VIÊN HƯỚNG DẪN:** [Tên giảng viên]

**SINH VIÊN THỰC HIỆN:** [Tên sinh viên]

**MSSV:** [Mã số sinh viên]

**LỚP:** [Tên lớp]

**NĂM HỌC:** 2025-2026

---

# LỜI MỞ ĐẦU

## Bối cảnh nghiên cứu

Nông nghiệp đóng vai trò then chốt trong an ninh lương thực toàn cầu, đặc biệt trong bối cảnh dân số thế giới dự kiến đạt 9.7 tỷ người vào năm 2050. Theo Tổ chức Lương thực và Nông nghiệp Liên Hợp Quốc (FAO), khoảng 20-40% sản lượng cây trồng bị thất thoát hàng năm do sâu bệnh và dịch hại. Việc phát hiện sớm và chính xác các bệnh trên cây trồng là yếu tố then chốt để giảm thiểu thiệt hại và đảm bảo năng suất.

## Vai trò của thị giác máy tính trong nông nghiệp

Thị giác máy tính (Computer Vision) kết hợp với học sâu (Deep Learning) đã mở ra những khả năng mới trong nông nghiệp chính xác. Các kỹ thuật nhận dạng ảnh tự động cho phép phát hiện bệnh trên lá cây thông qua phân tích các đặc trưng hình thái, màu sắc và kết cấu bề mặt lá. Phương pháp này mang lại nhiều ưu điểm vượt trội so với quan sát thủ công truyền thống: tốc độ xử lý nhanh, khả năng mở rộng, và tính khách quan trong đánh giá.

## Tính cấp thiết của việc phát hiện bệnh cây trồng

Việc phát hiện sớm các bệnh trên cây trồng giúp nông dân:
- Áp dụng biện pháp xử lý kịp thời, giảm thiểu thiệt hại
- Tối ưu hóa việc sử dụng thuốc bảo vệ thực vật
- Giảm chi phí sản xuất và tăng năng suất
- Ngăn chặn sự lây lan của dịch bệnh trên diện rộng

Tuy nhiên, tại nhiều khu vực nông thôn, việc tiếp cận chuyên gia nông nghiệp và dịch vụ tư vấn còn hạn chế. Do đó, một hệ thống tự động có thể hoạt động ngoại tuyến trên thiết bị di động sẽ mang lại giá trị thực tiễn cao.

## Động lực thực hiện đề tài

Dự án được phát triển với mục tiêu xây dựng một hệ thống nhận diện bệnh trên lá cây có khả năng:
1. Phân loại chính xác nhiều loại bệnh trên các loại cây trồng khác nhau
2. Hoạt động ngoại tuyến trên thiết bị di động Android
3. Cung cấp giải thích trực quan về kết quả dự đoán thông qua Grad-CAM
4. Đảm bảo chất lượng ảnh đầu vào thông qua các bộ lọc kiểm tra
5. Quản lý metadata và versioning mô hình một cách hệ thống

Hệ thống sử dụng kiến trúc EfficientNet-B2 - một mô hình cân bằng tốt giữa độ chính xác và hiệu suất tính toán, phù hợp cho triển khai trên thiết bị di động.

---

# CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI

## 1.1 Đặt vấn đề

Bệnh trên lá cây là một trong những nguyên nhân chính gây giảm năng suất nông nghiệp. Các bệnh thường gặp bao gồm: đốm lá, mốc sương, gỉ sắt, thối rễ, và các bệnh do virus. Mỗi loại bệnh có các đặc trưng hình thái riêng biệt trên bề mặt lá, tạo cơ sở cho việc áp dụng các thuật toán thị giác máy tính để nhận diện tự động.

Thách thức chính trong việc phát triển hệ thống nhận diện bệnh trên lá cây bao gồm:
- **Đa dạng sinh học**: Hàng chục loại cây trồng với hàng trăm loại bệnh khác nhau
- **Điều kiện chụp ảnh thực tế**: Ánh sáng, góc chụp, độ phân giải khác nhau
- **Yêu cầu về độ chính xác**: Sai sót trong chẩn đoán có thể dẫn đến thiệt hại kinh tế
- **Hạn chế về tài nguyên**: Triển khai trên thiết bị di động đòi hỏi tối ưu về kích thước mô hình và thời gian suy luận

## 1.2 Mục tiêu đề tài

### Mục tiêu tổng quát

Xây dựng hệ thống nhận diện bệnh trên lá cây sử dụng deep learning, có khả năng triển khai trên thiết bị Android với khả năng hoạt động ngoại tuyến, cung cấp kết quả dự đoán kèm theo giải thích trực quan.

### Mục tiêu cụ thể

1. **Xây dựng mô hình phân loại ảnh**:
   - Sử dụng kiến trúc EfficientNet-B2 làm backbone
   - Phân loại 38 lớp bệnh trên các loại cây trồng khác nhau
   - Tích hợp lớp "Unknown" để xử lý các trường hợp không thuộc các lớp đã biết

2. **Phát triển hệ thống tiền xử lý và tăng cường dữ liệu**:
   - Xây dựng pipeline tiền xử lý ảnh chuẩn hóa
   - Áp dụng các kỹ thuật augmentation mô phỏng điều kiện thực tế (thời tiết, artifact máy ảnh)
   - Sử dụng segmentation mask để tập trung vào vùng lá cây

3. **Triển khai trên thiết bị di động**:
   - Chuyển đổi mô hình sang định dạng tối ưu (ONNX, TFLite, TorchScript)
   - Phát triển ứng dụng Android với giao diện thân thiện
   - Tích hợp khả năng suy luận ngoại tuyến

4. **Tích hợp các module hỗ trợ**:
   - Grad-CAM để giải thích kết quả dự đoán
   - Quality Validator để kiểm tra chất lượng ảnh đầu vào
   - Metadata Manager để quản lý thông tin mô hình

## 1.3 Phạm vi nghiên cứu

### Những gì được thực hiện

- **Dữ liệu**: Sử dụng bộ dữ liệu PlantVillage và các biến thể mở rộng
- **Mô hình**: EfficientNet-B2 với 39 lớp đầu ra (38 bệnh + 1 Unknown)
- **Huấn luyện**: Full fine-tuning với pretrained weights từ ImageNet
- **Triển khai**: Ứng dụng Android sử dụng PyTorch Mobile (TorchScript)
- **Công cụ hỗ trợ**: Grad-CAM, Quality Validator, Metadata Manager

### Những gì chưa thực hiện

- **Thu thập dữ liệu thực địa**: Dữ liệu chủ yếu từ bộ PlantVillage trong điều kiện phòng thí nghiệm
- **Tối ưu hóa nâng cao cho mobile**: Chưa áp dụng các kỹ thuật quantization nâng cao (pruning, knowledge distillation)
- **Phát hiện vùng bệnh (object detection)**: Hệ thống chỉ thực hiện phân loại ảnh toàn cục
- **Đa dạng hóa loại cây**: Chưa bao phủ tất cả các loại cây trồng phổ biến
- **Tích hợp cloud**: Chưa có cơ chế đồng bộ và cập nhật mô hình từ xa

## 1.4 Ý nghĩa khoa học và thực tiễn

### Ý nghĩa khoa học

- Đóng góp vào lĩnh vực nông nghiệp chính xác (precision agriculture)
- Nghiên cứu ứng dụng transfer learning và fine-tuning trong nhận dạng bệnh cây trồng
- Khám phá hiệu quả của các kỹ thuật augmentation đặc thù cho ảnh nông nghiệp
- Tích hợp explainable AI (Grad-CAM) trong hệ thống thực tế

### Ý nghĩa thực tiễn

- Cung cấp công cụ chẩn đoán nhanh cho nông dân, đặc biệt ở vùng sâu vùng xa
- Giảm phụ thuộc vào chuyên gia và dịch vụ tư vấn trực tiếp
- Hỗ trợ ra quyết định kịp thời trong quản lý dịch bệnh
- Có thể mở rộng thành hệ thống cảnh báo sớm dịch bệnh trên diện rộng

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1 Tổng quan về Thị giác máy tính

Thị giác máy tính (Computer Vision) là lĩnh vực nghiên cứu về cách máy tính có thể "nhìn" và hiểu được nội dung của hình ảnh kỹ thuật số. Các nhiệm vụ chính bao gồm:
- **Phân loại ảnh (Image Classification)**: Gán nhãn cho toàn bộ bức ảnh
- **Phát hiện đối tượng (Object Detection)**: Xác định vị trí và phân loại nhiều đối tượng trong ảnh
- **Phân đoạn ngữ nghĩa (Semantic Segmentation)**: Gán nhãn cho từng pixel trong ảnh
- **Phân đoạn instance (Instance Segmentation)**: Phân đoạn và phân biệt các instance riêng lẻ của cùng một lớp

Trong nông nghiệp, thị giác máy tính được ứng dụng để: đếm quả, ước tính năng suất, phát hiện cỏ dại, và chẩn đoán bệnh cây trồng.

## 2.2 Deep Learning trong nhận dạng ảnh

Học sâu (Deep Learning) đã cách mạng hóa thị giác máy tính thông qua việc sử dụng các mạng neural tích chập (Convolutional Neural Networks - CNNs). CNN học các đặc trưng hình ảnh thông qua các lớp tích chập, từ các đặc trưng cấp thấp (cạnh, góc) đến các đặc trưng cấp cao (hình dạng, kết cấu).

Các kiến trúc CNN tiêu biểu:
- **LeNet-5** (1998): Kiến trúc CNN đầu tiên cho nhận dạng chữ số
- **AlexNet** (2012): Đột phá với ReLU và dropout
- **VGG** (2014): Sử dụng các bộ lọc 3x3 nhỏ
- **ResNet** (2015): Giới thiệu kết nối tắt (skip connections)
- **EfficientNet** (2019): Tối ưu hóa đồng thời độ sâu, chiều rộng và độ phân giải

## 2.3 Bài toán phân loại bệnh trên lá cây

Phân loại bệnh trên lá cây là bài toán phân loại ảnh đa lớp (multi-class classification). Mỗi ảnh lá cây được gán vào một trong nhiều lớp bệnh hoặc lớp "khỏe mạnh".

Đặc thù của bài toán:
- **Đa dạng nội lớp**: Cùng một bệnh có thể biểu hiện khác nhau trên các loại cây khác nhau
- **Tương đồng giữa các lớp**: Một số bệnh có triệu chứng tương tự nhau
- **Mất cân bằng dữ liệu**: Số lượng ảnh giữa các lớp không đồng đều
- **Nhiễu thực tế**: Ảnh chụp trong điều kiện thực tế có thể bị mờ, thiếu sáng, hoặc có vật cản

## 2.4 Mạng EfficientNet-B2

### Kiến trúc

EfficientNet là một họ các mô hình CNN được phát triển bởi Google Research, sử dụng phương pháp compound scaling để cân bằng đồng thời ba chiều của mạng:
- **Độ sâu (depth)**: Số lượng lớp
- **Chiều rộng (width)**: Số kênh trong mỗi lớp
- **Độ phân giải (resolution)**: Kích thước ảnh đầu vào

EfficientNet-B2 là phiên bản thứ hai trong họ EfficientNet, với các thông số:
- **Độ sâu**: 23 lớp tích chập
- **Chiều rộng**: Hệ số mở rộng 1.2
- **Độ phân giải đầu vào**: 260×260 pixels
- **Số tham số**: khoảng 9.2 triệu
- **FLOPs**: khoảng 1.0 tỷ

Kiến trúc EfficientNet sử dụng MBConv (Mobile Inverted Bottleneck Convolution) blocks với squeeze-and-excitation (SE) optimization.

### Nguyên lý hoạt động

EfficientNet-B2 hoạt động theo nguyên tắc:
1. **Stem layer**: Tích chập 3×3 với stride=2 để giảm kích thước không gian
2. **MBConv blocks**: Các khối chính với tích chập sâu (depthwise convolution) và tích chập điểm (pointwise convolution)
3. **Squeeze-and-Excitation**: Tăng cường biểu diễn đặc trưng bằng cách học trọng số kênh
4. **Top layer**: Tích chập 1×1 và global average pooling
5. **Classification head**: Fully connected layer với softmax activation

### Lý do lựa chọn

EfficientNet-B2 được lựa chọn vì:
- **Cân bằng tốt giữa độ chính xác và hiệu suất**: Đạt độ chính xác cao hơn ResNet-50 với ít tham số hơn
- **Phù hợp cho mobile deployment**: Kích thước mô hình vừa phải, thời gian suy luận nhanh
- **Transfer learning hiệu quả**: Pretrained weights từ ImageNet giúp hội tụ nhanh hơn
- **Khả năng tổng quát hóa tốt**: Kiến trúc compound scaling giúp mô hình học được các đặc trưng đa dạng

```
Hình 2.1. Kiến trúc EfficientNet-B2

TODO: Chèn hình kiến trúc EfficientNet-B2
```

## 2.5 TensorFlow Lite

TensorFlow Lite (TFLite) là framework của Google cho việc triển khai machine learning trên thiết bị di động, embedded và IoT. TFLite cung cấp:

- **Kích thước nhỏ gọn**: Binary size chỉ khoảng 300KB
- **Tối ưu hóa phần cứng**: Hỗ trợ GPU, DSP, NPU delegates
- **Quantization**: Giảm kích thước mô hình và tăng tốc độ suy luận
- **Cross-platform**: Hỗ trợ Android, iOS, Linux, macOS, Windows

Quy trình chuyển đổi mô hình sang TFLite:
1. Xuất mô hình PyTorch sang ONNX
2. Chuyển đổi ONNX sang TensorFlow SavedModel
3. Chuyển đổi SavedModel sang TFLite với các tùy chọn optimization

## 2.6 Grad-CAM

### Khái niệm

Grad-CAM (Gradient-weighted Class Activation Mapping) là kỹ thuật explainable AI dùng để trực quan hóa vùng ảnh đóng góp nhiều nhất vào quyết định của mô hình CNN. Grad-CAM sử dụng gradient của lớp đích chảy vào lớp tích chập cuối cùng để tạo ra bản đồ nhiệt (heatmap) coarse localization.

### Nguyên lý

Grad-CAM hoạt động theo các bước:
1. **Forward pass**: Đưa ảnh vào mô hình để lấy logits đầu ra
2. **Backward pass**: Tính gradient của lớp đích đối với feature maps của lớp tích chập cuối cùng
3. **Global average pooling**: Tính trọng số cho mỗi kênh bằng cách lấy trung bình gradient theo không gian
4. **Weighted combination**: Kết hợp feature maps theo trọng số
5. **ReLU và normalization**: Áp dụng ReLU để chỉ giữ các vùng có đóng góp dương, sau đó chuẩn hóa về [0,1]

### Vai trò trong giải thích kết quả

Grad-CAM giúp:
- **Kiểm tra tính hợp lý**: Xác nhận mô hình tập trung vào vùng lá thay vì nền
- **Phát hiện bias**: Nhận diện các đặc trưng giả mạo (watermark, nhãn) mà mô hình có thể học
- **Tăng niềm tin**: Người dùng hiểu được lý do đằng sau dự đoán
- **Hỗ trợ chuyên gia**: Cung cấp thông tin cho nhà nông học trong việc xác nhận chẩn đoán

```
Hình 2.2. Ví dụ Grad-CAM

TODO: Chèn hình ví dụ Grad-CAM heatmap trên ảnh lá cây
```

## 2.7 Android On-device AI

Triển khai AI trên thiết bị Android mang lại nhiều lợi ích:
- **Hoạt động ngoại tuyến**: Không cần kết nối internet
- **Bảo mật**: Dữ liệu không rời khỏi thiết bị
- **Độ trễ thấp**: Không có độ trễ mạng
- **Chi phí**: Không tốn chi phí server

Các framework phổ biến:
- **TensorFlow Lite**: Phổ biến nhất, hỗ trợ rộng rãi
- **PyTorch Mobile**: Tích hợp tốt với hệ sinh thái PyTorch
- **ML Kit**: API cấp cao của Google
- **MediaPipe**: Cho các tác vụ multimodal

Trong dự án này, ứng dụng Android sử dụng PyTorch Mobile (TorchScript) cho suy luận, kết hợp với camera để chụp và phân tích ảnh lá cây.

---

# CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1 Kiến trúc tổng thể

Hệ thống được thiết kế theo kiến trúc module hóa, bao gồm các thành phần chính:

```
┌─────────────────────────────────────────────────────────────┐
│                    Ứng dụng Android (AgriLens)               │
├─────────────────────────────────────────────────────────────┤
│  Camera  │  Quality Validator  │  Inference Engine (PyTorch) │
├─────────────────────────────────────────────────────────────┤
│              Grad-CAM Visualization (Python backend)         │
├─────────────────────────────────────────────────────────────┤
│              Metadata Manager & Label System                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Backend ML (SourceCode)                   │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline  │  Training  │  Export (ONNX/TFLite/Torch)  │
└─────────────────────────────────────────────────────────────┘
```

```
Hình 3.1. Kiến trúc tổng thể hệ thống

TODO: Chèn sơ đồ kiến trúc hệ thống chi tiết
```

Luồng dữ liệu chính:
1. **Thu thập ảnh**: Camera Android hoặc file ảnh
2. **Kiểm tra chất lượng**: Quality Validator đánh giá độ mờ, độ sáng, độ phân giải
3. **Tiền xử lý**: Resize, normalize theo ImageNet statistics
4. **Suy luận**: Mô hình EfficientNet-B2 dự đoán lớp bệnh
5. **Hậu xử lý**: Softmax, top-k predictions, unknown detection
6. **Giải thích**: Grad-CAM tạo heatmap (tuỳ chọn)
7. **Hiển thị**: Kết quả và hướng dẫn điều trị

## 3.2 Phân tích yêu cầu

### Yêu cầu chức năng

| STT | Yêu cầu | Mô tả |
|-----|---------|-------|
| 1 | Phân loại bệnh | Hệ thống phải phân loại được 38 loại bệnh trên lá cây |
| 2 | Phát hiện unknown | Hệ thống phải phát hiện được các lá cây không thuộc các lớp đã biết |
| 3 | Kiểm tra chất lượng ảnh | Hệ thống phải từ chối các ảnh quá mờ, quá tối/sáng, hoặc độ phân giải không phù hợp |
| 4 | Giải thích kết quả | Hệ thống phải cung cấp heatmap Grad-CAM để giải thích vùng ảnh đóng góp vào dự đoán |
| 5 | Hoạt động ngoại tuyến | Ứng dụng Android phải hoạt động được không cần internet |
| 6 | Quản lý metadata | Hệ thống phải quản lý version, labels, và thông tin mô hình |
| 7 | Hiển thị kết quả | Ứng dụng phải hiển thị kết quả với confidence score và tên bệnh |
| 8 | Hướng dẫn điều trị | Ứng dụng phải cung cấp thông tin về cách điều trị bệnh |

### Yêu cầu phi chức năng

| STT | Yêu cầu | Mô tả |
|-----|---------|-------|
| 1 | Hiệu suất | Thời gian suy luận phải dưới 1 giây trên thiết bị mobile |
| 2 | Kích thước mô hình | Mô hình phải đủ nhỏ để nhúng vào ứng dụng Android (< 50MB) |
| 3 | Độ chính xác | Độ chính xác phải đạt mức chấp nhận được cho ứng dụng thực tế |
| 4 | Khả năng mở rộng | Hệ thống phải dễ dàng thêm các lớp bệnh mới |
| 5 | Bảo mật | Dữ liệu người dùng không được gửi lên server |
| 6 | Tương thích | Ứng dụng phải chạy trên Android API 24+ |
| 7 | Tiêu thụ tài nguyên | Ứng dụng không được tiêu thụ quá nhiều RAM và pin |

## 3.3 Thiết kế luồng xử lý

Luồng xử lý chi tiết của hệ thống:

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ảnh đầu    │───▶│  Quality Check   │───▶│  Tiền xử lý     │
│   vào        │    │  (Blur, Light,   │    │  (Resize,       │
│              │    │   Resolution)    │    │   Normalize)    │
└──────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Kết quả    │◀───│   Hậu xử lý      │◀───│   Suy luận      │
│   (disease,  │    │   (Softmax,      │    │   (EfficientNet │
│   confidence)│    │    top-k)        │    │    -B2)         │
└──────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Hiển thị   │◀───│   Grad-CAM       │◀───│   Kiểm tra      │
│   kết quả    │    │   Visualization  │    │   unknown       │
└──────────────┘    └──────────────────┘    └─────────────────┘
```

```
Hình 3.2. Luồng xử lý hệ thống

TODO: Chèn sơ đồ luồng xử lý chi tiết
```

## 3.4 Thiết kế cơ sở dữ liệu và metadata

Hệ thống metadata được thiết kế để quản lý thông tin mô hình:

```json
{
  "model": {
    "architecture": "efficientnet_b2",
    "num_classes": 39,
    "image_size": 260,
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225]
  },
  "training": {
    "dataset": "PlantVillage",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4
  },
  "version": {
    "model": "1.0.0",
    "export_date": "2026-05-30T...",
    "export_tool": "plant_disease_detection_v1"
  },
  "deployment": {
    "target_platform": "android",
    "tflite_compatible": true,
    "min_android_api": 24
  },
  "labels": {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    ...
  }
}
```

Metadata được lưu trữ dưới dạng JSON, dễ dàng đọc và cập nhật. Labels được xuất ra cả định dạng JSON (cho ứng dụng) và TXT (cho TFLite metadata).

---

# CHƯƠNG 4. XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG

## 4.1 Cấu trúc thư mục dự án

Dự án được tổ chức theo cấu trúc module hóa:

```
PlantDiseaseDetectionProject/
├── README.md                          # Tài liệu chính
├── PROJECT_SUMMARY.md                 # Tóm tắt dự án
├── CHANGELOG.md                       # Lịch sử thay đổi
├── Instruction.md                     # Hướng dẫn viết báo cáo
├── SourceCode/                        # Backend ML
│   ├── src/
│   │   ├── __init__.py
│   │   ├── train.py                   # Huấn luyện mô hình
│   │   ├── inference.py               # Suy luận TFLite
│   │   ├── evaluate_and_convert.py    # Đánh giá và xuất mô hình
│   │   ├── gradcam.py                 # Grad-CAM visualization
│   │   ├── quality_validator.py       # Kiểm tra chất lượng ảnh
│   │   ├── metadata.py                # Quản lý metadata
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── preprocess.py          # Tiền xử lý và augmentation
│   │       └── generate_background_noise.py
│   ├── configs/
│   │   ├── __init__.py
│   │   └── config.yaml                # Cấu hình tập trung
│   ├── data/                          # Dữ liệu (không commit)
│   ├── models/                        # Checkpoint (không commit)
│   ├── saved_model/                   # TFLite export output
│   ├── plant_model.onnx               # ONNX model
│   ├── plant_model_tflite_float32/    # Float32 TFLite
│   ├── plant_model_tflite_int8/       # INT8 TFLite
│   ├── plant_model.pt                 # TorchScript model
│   ├── labels.json                    # Class labels
│   ├── labels.txt                     # Class labels (text)
│   ├── requirements.txt               # Python dependencies
│   ├── pyproject.toml                 # Package config
│   └── README.md
├── agrilens/                          # Android App
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/com/example/
│   │   │   │   ├── MainActivity.java
│   │   │   │   ├── ImageClassifierHelper.java
│   │   │   │   ├── GuideActivity.kt
│   │   │   │   ├── DiseaseTreatmentRepository.java
│   │   │   │   └── OverlayView.java
│   │   │   ├── assets/
│   │   │   │   ├── plant_model.pt
│   │   │   │   ├── labels.txt
│   │   │   │   └── labels.json
│   │   │   └── res/
│   │   └── build.gradle.kts
│   ├── gradle/
│   ├── gradlew
│   ├── build.gradle.kts
│   └── README.md
└── PlantDiseaseDetectionKnowledge/    # Tài liệu kỹ thuật
```

```
Hình 4.1. Cấu trúc thư mục dự án

TODO: Chèn hình cấu trúc thư mục
```

## 4.2 Bộ dữ liệu

### Nguồn dữ liệu

Hệ thống sử dụng bộ dữ liệu PlantVillage - một bộ dữ liệu công khai phổ biến cho bài toán phân loại bệnh cây trồng. Bộ dữ liệu bao gồm ảnh lá cây của 38 loại cây trồng khác nhau, mỗi loại có thể có nhiều bệnh hoặc trạng thái khỏe mạnh.

### Cấu trúc dữ liệu

Dữ liệu được tổ chức theo cấu trúc thư mục phân cấp:

```
data/raw/
├── Apple___Apple_scab/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Apple___Black_rot/
│   ├── image_001.jpg
│   └── ...
├── Tomato___Early_blight/
│   ├── image_001.jpg
│   └── ...
└── ... (38 classes total)
```

### Nhãn dữ liệu

Hệ thống sử dụng 39 lớp đầu ra:
- 38 lớp bệnh trên các loại cây trồng (từ bộ PlantVillage)
- 1 lớp "Unknown" để xử lý các trường hợp không thuộc các lớp đã biết

Danh sách đầy đủ các lớp:

| Index | Class Name |
|-------|------------|
| 0 | Apple___Apple_scab |
| 1 | Apple___Black_rot |
| 2 | Apple___Cedar_apple_rust |
| 3 | Apple___healthy |
| 4 | Blueberry___healthy |
| 5 | Cherry_(including_sour)___Powdery_mildew |
| 6 | Cherry_(including_sour)___healthy |
| 7 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| 8 | Corn_(maize)___Common_rust_ |
| 9 | Corn_(maize)___Northern_Leaf_Blight |
| 10 | Corn_(maize)___healthy |
| 11 | Grape___Black_rot |
| 12 | Grape___Esca_(Black_Measles) |
| 13 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) |
| 14 | Grape___healthy |
| 15 | Orange___Haunglongbing_(Citrus_greening) |
| 16 | Peach___Bacterial_spot |
| 17 | Peach___healthy |
| 18 | Pepper,_bell___Bacterial_spot |
| 19 | Pepper,_bell___healthy |
| 20 | Potato___Early_blight |
| 21 | Potato___Late_blight |
| 22 | Potato___healthy |
| 23 | Raspberry___healthy |
| 24 | Soybean___healthy |
| 25 | Squash___Powdery_mildew |
| 26 | Strawberry___Leaf_scorch |
| 27 | Strawberry___healthy |
| 28 | Tomato___Bacterial_spot |
| 29 | Tomato___Early_blight |
| 30 | Tomato___Late_blight |
| 31 | Tomato___Leaf_Mold |
| 32 | Tomato___Septoria_leaf_spot |
| 33 | Tomato___Spider_mites Two-spotted_spider_mite |
| 34 | Tomato___Target_Spot |
| 35 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 36 | Tomato___Tomato_mosaic_virus |
| 37 | Tomato___healthy |
| 38 | Unknown |

```
Bảng 4.1. Thống kê dữ liệu

TODO: Điền số liệu thống kê (số ảnh mỗi lớp, tỷ lệ train/val/test)
```

### Phân chia dữ liệu

Dữ liệu được phân chia theo tỷ lệ:
- **Training set**: 80% - Dùng để huấn luyện mô hình
- **Validation set**: 10% - Dùng để điều chỉnh hyperparameter và early stopping
- **Test set**: 10% - Dùng để đánh giá cuối cùng

Việc phân chia được thực hiện stratified để đảm bảo tỷ lệ các lớp được giữ nguyên trong mỗi tập.

## 4.3 Tiền xử lý dữ liệu

### Resize

Tất cả ảnh đầu vào được resize về kích thước 260×260 pixels để phù hợp với đầu vào của EfficientNet-B2. Phương pháp resize sử dụng `cv2.INTER_AREA` để giảm thiểu aliasing artifacts.

```python
# Kích thước đầu vào
IMAGE_SIZE = 260

# Resize với interpolation phù hợp
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
```

### Normalize

Ảnh được chuẩn hóa theo thống kê ImageNet:
- **Mean**: [0.485, 0.456, 0.406]
- **Std**: [0.229, 0.224, 0.225]

```python
# Chuyển từ BGR sang RGB và normalize về [0, 1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Áp dụng ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = (img - mean) / std
```

### Data Loader

Hệ thống sử dụng PyTorch DataLoader với các tùy chọn:
- **Batch size**: 32 (có thể điều chỉnh qua config)
- **Num workers**: 4 (song song loading)
- **Pin memory**: True (tăng tốc transfer sang GPU)
- **WeightedRandomSampler**: Xử lý class imbalance

```python
# WeightedRandomSampler cho class imbalance
class_counts = Counter([label for _, _, _, label in train_samples])
sample_weights = [1.0 / class_counts[label] for _, _, _, label in train_samples]
sampler = WeightedRandomSampler(torch.tensor(sample_weights), len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, 
                          num_workers=4, pin_memory=True)
```

## 4.4 Tăng cường dữ liệu

### Mục đích

Tăng cường dữ liệu (data augmentation) được áp dụng để:
- Tăng kích thước tập huấn luyện ảo
- Giảm overfitting
- Cải thiện khả năng tổng quát hóa
- Mô phỏng điều kiện thực tế khi chụp ảnh bằng mobile

### Các kỹ thuật augmentation

Hệ thống áp dụng các kỹ thuật augmentation được chia thành 4 nhóm:

| Nhóm | Kỹ thuật | Mục đích |
|------|----------|----------|
| **Weather simulations** | RandomShadow, RandomFog, RandomRain | Mô phỏng điều kiện thời tiết thực tế |
| **Camera artifacts** | MotionBlur, GaussNoise, ImageCompression, GaussianBlur | Mô phỏng artifact từ camera mobile |
| **Geometric transforms** | Rotate, RandomScale, RandomPerspective, Affine | Mô phỏng góc chụp và khoảng cách khác nhau |
| **Color transforms** | RandomBrightnessContrast, HueSaturationValue, ToGray | Mô phỏng điều kiện ánh sáng khác nhau |

### Chi tiết cấu hình augmentation

```yaml
augmentation:
  # Weather simulations
  shadow_prob: 0.3
  fog_prob: 0.2
  rain_prob: 0.2
  
  # Camera artifacts
  motion_blur_prob: 0.3
  gauss_noise_prob: 0.4
  compression_prob: 0.3
  blur_prob: 0.2
  
  # Color transforms
  brightness_prob: 0.8
  grayscale_prob: 0.05
  
  # Geometric transforms
  random_erasing_prob: 0.25
```

### Segmentation Mask

Một kỹ thuật quan trọng là sử dụng segmentation mask để tập trung vào vùng lá cây:
- Mask được áp dụng với xác suất 70% trong quá trình huấn luyện
- Mask giúp mô hình học các đặc trưng trên lá thay vì nền
- Trong quá trình validation/test, mask được áp dụng với xác suất 20%

```python
# Áp dụng mask với xác suất mask_prob
if seg_path and os.path.exists(seg_path) and np.random.random() < mask_prob:
    seg = safe_read_image(seg_path, cv2.IMREAD_GRAYSCALE)
    mask = (seg > 127).astype(np.float32)
    mask = np.stack([mask] * 3, axis=-1)
    return rgb * mask
```

## 4.5 Xây dựng mô hình EfficientNet-B2

### Kiến trúc mô hình

Mô hình được xây dựng dựa trên EfficientNet-B2 pretrained từ torchvision:

```python
import torch
from torchvision import models

# Load pretrained EfficientNet-B2
model = models.efficientnet_b2(
    weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
)

# Full fine-tuning: tất cả các tham số đều được cập nhật
for param in model.parameters():
    param.requires_grad = True

# Thay đổi classification head cho bài toán 39 lớp
num_classes = 39
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
```

### Lý do lựa chọn EfficientNet-B2

1. **Cân bằng độ chính xác và hiệu suất**: EfficientNet-B2 có số tham số vừa phải (~9.2M) nhưng đạt độ chính xác cao
2. **Phù hợp cho mobile**: Kích thước mô hình đủ nhỏ để triển khai trên Android
3. **Transfer learning hiệu quả**: Pretrained weights từ ImageNet giúp hội tụ nhanh
4. **Compound scaling**: Kiến trúc được tối ưu đồng thời về depth, width, resolution

## 4.6 Quy trình huấn luyện

### Loss function

Sử dụng CrossEntropyLoss với label smoothing:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Label smoothing giúp:
- Giảm overfitting
- Cải thiện calibration của mô hình
- Tăng khả năng tổng quát hóa

### Optimizer

Sử dụng AdamW optimizer:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

AdamW tách biệt weight decay khỏi adaptive learning rate, giúp tối ưu hóa ổn định hơn.

### Learning rate scheduler

Sử dụng CosineAnnealingWarmRestarts:

```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,        # Initial restart period
    T_mult=2      # Multiplicative factor sau mỗi restart
)
```

Scheduler này:
- Giảm learning rate theo cosine curve
- Restart sau mỗi T_0 epochs
- Tăng period sau mỗi restart (T_mult=2)

### Gradient clipping

Áp dụng gradient clipping để tránh gradient explosion:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Mixed precision training

Sử dụng Automatic Mixed Precision (AMP) để tăng tốc và giảm memory:

```python
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Checkpointing

Mô hình tốt nhất được lưu dựa trên validation accuracy:

```python
if val_acc > best_acc:
    best_acc = val_acc
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc
    }, 'models/best_model.pth')
```

### Cấu hình huấn luyện

```yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  epochs: 20
  label_smoothing: 0.1
  gradient_clip: 1.0
```

## 4.7 Module Quality Validation

Module Quality Validator kiểm tra chất lượng ảnh đầu vào trước khi đưa vào mô hình phân loại.

### Blur Detection

Sử dụng Laplacian variance để phát hiện ảnh mờ:

```python
def _check_blur(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < self.config["blur_threshold"]:
        return False, {
            "guidance": f"Ảnh quá mờ (độ sắc nét: {laplacian_var:.1f}, ngưỡng: {self.config['blur_threshold']})"
        }
    return True, {"guidance": f"Độ sắc nét OK (score: {laplacian_var:.1f})"}
```

**Ngưỡng**: 100.0 (có thể điều chỉnh qua config)

### Brightness Validation

Kiểm tra độ sáng trung bình của ảnh:

```python
def _check_brightness(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < self.config["min_brightness"]:
        return False, {"guidance": f"Ảnh quá tối (độ sáng: {mean_brightness:.1f})"}
    if mean_brightness > self.config["max_brightness"]:
        return False, {"guidance": f"Ảnh quá sáng (độ sáng: {mean_brightness:.1f})"}
    return True, {"guidance": f"Độ sáng OK (score: {mean_brightness:.1f})"}
```

**Ngưỡng**: Min 30.0, Max 220.0 (trên thang 0-255)

### Resolution Validation

Kiểm tra độ phân giải ảnh:

```python
def _check_resolution(self, width, height):
    if width < self.config["min_width"] or height < self.config["min_height"]:
        return False, {"guidance": f"Độ phân giải quá thấp ({width}x{height})"}
    if width > self.config["max_width"] or height > self.config["max_height"]:
        return True, {"guidance": f"Độ phân giải cao ({width}x{height}), sẽ được downscale"}
    return True, {"guidance": f"Độ phân giải OK ({width}x{height})"}
```

**Ngưỡng**: Min 224×224, Max 4000×4000

### Tích hợp vào hệ thống

Quality Validator được tích hợp vào cả backend Python và Android app:
- **Backend**: Kiểm tra trước khi inference
- **Android**: Kiểm tra ngay sau khi chụp ảnh, cung cấp hướng dẫn cho người dùng

## 4.8 Module Grad-CAM

### Kiến trúc

Grad-CAM module được triển khai như một class độc lập:

```python
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Target layer mặc định cho EfficientNet-B2
        if target_layer is None:
            target_layer = 'features.8.0'  # Final conv layer
        
        self.target_layer_name = target_layer
        self.target_layer = None
        
        # Tìm target layer
        for name, layer in model.named_modules():
            if name == target_layer:
                self.target_layer = layer
                break
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
```

### Tạo heatmap

```python
def generate_heatmap(self, input_tensor, target_class=None):
    # Forward pass
    output = self.model(input_tensor)
    
    # Xác định target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    target_score = output[0, target_class]
    target_score.backward()
    
    # Lấy gradients và activations
    gradients = self.gradients.cpu().data.numpy()
    activations = self.activations.cpu().data.numpy()
    
    # Global average pooling của gradients
    weights = np.mean(gradients, axis=(2, 3))
    
    # Weighted combination của activations
    heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights[0]):
        heatmap += w * activations[0, i, :, :]
    
    # Áp dụng ReLU và normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap
```

### Overlay heatmap lên ảnh gốc

```python
def apply_heatmap_to_image(image, heatmap, alpha=0.5):
    # Resize heatmap theo kích thước ảnh gốc
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Tạo heatmap màu (jet colormap)
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend ảnh gốc và heatmap
    overlay = (image * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    
    return overlay
```

### Xuất visualization

Grad-CAM xuất ra 3 hình ảnh:
1. Ảnh gốc
2. Heatmap ( grayscale)
3. Overlay (heatmap blend với ảnh gốc)

## 4.9 Metadata Management

Module metadata quản lý thông tin mô hình và labels:

### Cấu trúc metadata

```python
class ModelMetadata:
    def __init__(self, config=None):
        self.metadata = {
            "model": {
                "architecture": "efficientnet_b2",
                "num_classes": 39,
                "image_size": 260,
                "input_mean": [0.485, 0.456, 0.406],
                "input_std": [0.229, 0.224, 0.225],
            },
            "training": {
                "dataset": "PlantVillage",
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 1e-4,
            },
            "version": {
                "model": "1.0.0",
                "export_date": datetime.now().isoformat(),
                "export_tool": "plant_disease_detection_v1",
            },
            "deployment": {
                "target_platform": "android",
                "tflite_compatible": True,
                "min_android_api": 24,
            },
            "labels": {}
        }
```

### Xuất labels

Labels được xuất ra 2 định dạng:
- **JSON**: Cho ứng dụng Android và Python inference
- **TXT**: Cho TFLite metadata writer

```python
def export_labels_json(self, output_dir="."):
    labels_dict = {str(k): v for k, v in self.metadata["labels"].items()}
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels_dict, f, indent=2)

def export_labels_txt(self, output_dir="."):
    sorted_labels = sorted(self.metadata["labels"].items(), key=lambda x: int(x[0]))
    with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
        for _, label in sorted_labels:
            f.write(label + '\n')
```

## 4.10 Pipeline xuất mô hình

### Tổng quan

Mô hình được xuất qua nhiều định dạng để phục vụ các mục đích triển khai khác nhau:

```
PyTorch (.pth)
    ↓
ONNX (.onnx)
    ↓
TensorFlow SavedModel
    ↓
TensorFlow Lite (.tflite)
    ├── Float32 version
    └── INT8 quantized version

PyTorch (.pth)
    ↓
TorchScript (.pt) → Android (PyTorch Mobile)
```

```
Hình 4.2. Pipeline chuyển đổi mô hình

TODO: Chèn sơ đồ pipeline chuyển đổi
```

### Bước 1: PyTorch → ONNX

```python
def export_onnx(model, input_tensor, output_path, opset_version=15):
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
```

**Lưu ý**: ONNX opset 15 được sử dụng để đảm bảo compatibility.

### Bước 2: ONNX → TensorFlow SavedModel

Sử dụng `onnx-tf` để chuyển đổi:

```python
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("plant_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model")
```

### Bước 3: TensorFlow SavedModel → TFLite

#### Float32 version

Sử dụng `onnx2tf` tool:

```bash
onnx2tf -i plant_model.onnx -o plant_model_tflite_float32 -osd -nuo
```

#### INT8 quantized version

Post-Training Quantization (PTQ) với representative dataset:

```python
def representative_dataset_gen():
    for images, _ in test_loader:
        # Denormalize và convert sang uint8
        images = images.numpy()
        images = (images * std + mean) * 255
        images = np.transpose(images, (0, 2, 3, 1))
        images = images.astype(np.uint8)
        yield [images]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
tflite_model = converter.convert()
```

### Bước 4: PyTorch → TorchScript

```python
def export_torchscript(model, input_tensor, output_path):
    model.eval()
    traced_script_module = torch.jit.trace(model, input_tensor)
    traced_script_module.save(output_path)
```

## 4.11 Ứng dụng Android

### Tổng quan

Ứng dụng Android "AgriLens" được phát triển bằng Kotlin/Java với các thành phần chính:

### Kiến trúc ứng dụng

```
┌─────────────────────────────────────────────┐
│              MainActivity                    │
│  (Camera, Permission, UI Management)        │
├─────────────────────────────────────────────┤
│         ImageClassifierHelper                │
│  (PyTorch Mobile Inference, Preprocessing)  │
├─────────────────────────────────────────────┤
│         DiseaseTreatmentRepository           │
│  (Treatment Data, Guide Information)         │
├─────────────────────────────────────────────┤
│              GuideActivity                   │
│  (Display Treatment Guide, Recommendations)  │
└─────────────────────────────────────────────┘
```

### ImageClassifierHelper

Lớp chịu trách nhiệm suy luận mô hình:

```java
public class ImageClassifierHelper {
    private Module module;  // PyTorch Mobile module
    private List<String> labels;
    
    public ImageClassifierHelper(Context context) {
        // Load model từ assets
        module = Module.load(context, "plant_model.pt");
        
        // Load labels
        labels = loadLabels(context);
    }
    
    public ClassificationResult classify(Bitmap bitmap) {
        // 1. Preprocess: resize, normalize
        Tensor inputTensor = preprocessBitmap(bitmap);
        
        // 2. Inference
        IValue output = module.forward(IValue.from(inputTensor));
        Tensor outputTensor = output.toTensor();
        
        // 3. Postprocess: softmax, top-k
        float[] probabilities = softmax(outputTensor.getDataAsFloatArray());
        int topClass = argmax(probabilities);
        
        // 4. Unknown detection
        if (probabilities[topClass] < UNKNOWN_THRESHOLD) {
            return new ClassificationResult("Unknown", 0);
        }
        
        return new ClassificationResult(labels.get(topClass), probabilities[topClass]);
    }
    
    private Tensor preprocessBitmap(Bitmap bitmap) {
        // Resize to 260x260
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 260, 260, true);
        
        // Convert to tensor và normalize
        float[] data = new float[260 * 260 * 3];
        for (int i = 0; i < 260; i++) {
            for (int j = 0; j < 260; j++) {
                int pixel = resized.getPixel(j, i);
                data[i * 260 * 3 + j * 3] = ((pixel >> 16) & 0xFF) / 255.0f - 0.485f / 0.229f;
                data[i * 260 * 3 + j * 3 + 1] = ((pixel >> 8) & 0xFF) / 255.0f - 0.456f / 0.224f;
                data[i * 260 * 3 + j * 3 + 2] = (pixel & 0xFF) / 255.0f - 0.406f / 0.225f;
            }
        }
        
        return Tensor.fromBlob(data, new long[]{1, 3, 260, 260});
    }
}
```

### Xử lý Unknown class

Ứng dụng tích hợp cơ chế phát hiện unknown:

```java
private static final float UNKNOWN_THRESHOLD = 0.65f;

if (maxProbability < UNKNOWN_THRESHOLD) {
    // Không đủ tự tin, trả về "Unknown"
    return new ClassificationResult("Unknown", maxProbability);
}
```

### Giao diện người dùng

Ứng dụng có các màn hình chính:

1. **Màn hình chính**: Camera preview với overlay kết quả
2. **Màn hình kết quả**: Hiển thị bệnh được phát hiện với confidence score
3. **Màn hình hướng dẫn**: Thông tin về bệnh và cách điều trị

```
Hình 4.3. Màn hình chính ứng dụng

TODO: Chèn ảnh screenshot màn hình chính
```

```
Hình 4.4. Màn hình nhận diện

TODO: Chèn ảnh screenshot màn hình nhận diện
```

```
Hình 4.5. Màn hình kết quả

TODO: Chèn ảnh screenshot màn hình kết quả
```

### Tích hợp model assets

Model và labels được đặt trong thư mục assets:

```
app/src/main/assets/
├── plant_model.pt      # TorchScript model
├── labels.txt          # Labels (một label mỗi dòng)
└── labels.json         # Labels (JSON format)
```

### Cấu hình Gradle

```kotlin
dependencies {
    implementation("org.pytorch:pytorch_android:1.13.1")
    implementation("org.pytorch:pytorch_android_torchvision:1.13.1")
    // ... other dependencies
}
```

---

# CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 5.1 Môi trường thực nghiệm

### Phần cứng

| Thành phần | Thông số |
|------------|----------|
| CPU | TODO: Bổ sung thông tin CPU |
| GPU | TODO: Bổ sung thông tin GPU |
| RAM | TODO: Bổ sung thông tin RAM |
| Thiết bị Android | TODO: Bổ sung thông tin thiết bị |

### Phần mềm

| Thành phần | Phiên bản |
|------------|-----------|
| Python | 3.10+ |
| PyTorch | 2.4+ |
| TensorFlow | 2.17+ |
| OpenCV | 4.10+ |
| Android Studio | TODO: Bổ sung phiên bản |
| Android API | 24+ |

## 5.2 Kết quả huấn luyện

```
TODO: Bổ sung biểu đồ Loss theo epochs

TODO: Bổ sung biểu đồ Accuracy theo epochs

TODO: Bổ sung kết quả huấn luyện cuối cùng
```

### Thông số huấn luyện

- **Số epochs**: 20
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)
- **Label smoothing**: 0.1
- **Gradient clipping**: max_norm=1.0

## 5.3 Đánh giá mô hình

```
Bảng 5.1. Kết quả đánh giá mô hình

TODO: Bổ sung Accuracy

TODO: Bổ sung Precision (macro, weighted)

TODO: Bổ sung Recall (macro, weighted)

TODO: Bổ sung F1-score (macro, weighted)
```

### Phân tích kết quả

```
TODO: Phân tích chi tiết kết quả theo từng lớp bệnh

TODO: Nhận diện các lớp có performance tốt/xấu

TODO: Đề xuất cải thiện
```

## 5.4 Confusion Matrix

```
Hình 5.1. Confusion Matrix

TODO: Chèn hình Confusion Matrix

TODO: Phân tích các confusion patterns (các lớp dễ nhầm lẫn)
```

## 5.5 Đánh giá Grad-CAM

```
Hình 5.2. Ví dụ Grad-CAM visualization

TODO: Chèn các ví dụ Grad-CAM trên các lớp bệnh khác nhau

TODO: Đánh giá chất lượng heatmap (có tập trung vào vùng bệnh không)

TODO: Phân tích các trường hợp Grad-CAM không hiệu quả
```

### Nhận xét

Grad-CAM được sử dụng để:
- Kiểm tra xem mô hình có tập trung vào vùng lá cây không
- Phát hiện các bias (học nền thay vì lá)
- Cung cấp giải thích trực quan cho người dùng

## 5.6 Đánh giá trên Android

```
Bảng 5.2. Kết quả đánh giá trên Android

| Tiêu chí           | Giá trị |
|--------------------|---------|
| Kích thước mô hình | TODO    |
| Thời gian suy luận | TODO    |
| RAM sử dụng        | TODO    |
| Pin tiêu thụ       | TODO    |
| Độ chính xác       | TODO    |
```

### Nhận xét

```
TODO: Đánh giá trải nghiệm người dùng

TODO: Phân tích các vấn đề hiệu năng

TODO: Đề xuất tối ưu hóa
```

## 5.7 Thảo luận kết quả

### Tóm tắt kết quả

```
TODO: Tóm tắt các kết quả chính đạt được

TODO: So sánh với các nghiên cứu liên quan

TODO: Phân tích ý nghĩa thực tiễn
```

### Hạn chế

```
TODO: Thảo luận về các hạn chế của hệ thống

TODO: Phân tích nguyên nhân của các hạn chế
```

---

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết luận

Dự án đã xây dựng thành công một hệ thống nhận diện bệnh trên lá cây sử dụng deep learning với các thành phần chính:

1. **Mô hình phân loại**: EfficientNet-B2 với 39 lớp đầu ra, được huấn luyện trên bộ dữ liệu PlantVillage với các kỹ thuật augmentation đa dạng
2. **Hệ thống tiền xử lý**: Pipeline chuẩn hóa ảnh với segmentation mask và augmentation mô phỏng điều kiện thực tế
3. **Module hỗ trợ**: Grad-CAM cho explainability, Quality Validator cho kiểm tra chất lượng ảnh, Metadata Manager cho quản lý mô hình
4. **Ứng dụng Android**: AgriLens với khả năng suy luận ngoại tuyến sử dụng PyTorch Mobile
5. **Pipeline xuất mô hình**: Chuyển đổi từ PyTorch sang ONNX, TFLite (float32/INT8), và TorchScript

Hệ thống được thiết kế theo kiến trúc module hóa, dễ bảo trì và mở rộng. Các quyết định kỹ thuật được đưa ra dựa trên sự cân bằng giữa độ chính xác, hiệu suất và khả năng triển khai thực tế.

## 6.2 Hạn chế

Hệ thống hiện tại có một số hạn chế:

1. **Dữ liệu**: Chủ yếu từ bộ PlantVillage trong điều kiện phòng thí nghiệm, chưa đa dạng hóa với ảnh thực địa
2. **Kiến trúc mô hình**: EfficientNet-B2 chưa phải là mô hình nhẹ nhất cho mobile deployment
3. **Phạm vi phân loại**: Chỉ phân loại ảnh toàn cục, chưa phát hiện vùng bệnh cụ thể
4. **Số lớp bệnh**: Chưa bao phủ tất cả các loại bệnh trên cây trồng phổ biến
5. **Tối ưu mobile**: Chưa áp dụng các kỹ thuật tối ưu nâng cao như pruning, knowledge distillation
6. **Dual deployment**: Hệ thống có hai đường triển khai (PyTorch Mobile và TFLite) gây phức tạp trong bảo trì

## 6.3 Hướng phát triển

### Thu thập dữ liệu thực tế

- Thu thập ảnh lá cây trong điều kiện thực địa
- Đa dạng hóa góc chụp, điều kiện ánh sáng, loại camera
- Hợp tác với nông dân và chuyên gia để xác nhận nhãn

### Tối ưu Android

- Áp dụng TFLite thay vì PyTorch Mobile để giảm kích thước và tăng tốc độ
- Sử dụng GPU delegate và NNAPI cho inference nhanh hơn
- Tối ưu memory footprint và battery consumption

### Bổ sung tập bệnh

- Mở rộng sang các loại cây trồng khác
- Thêm các bệnh mới xuất hiện
- Tích hợp với cơ sở dữ liệu bệnh cây trồng quốc gia/quốc tế

### Leaf Detection

- Phát triển module object detection (YOLO, SSD) để phát hiện lá cây trong ảnh
- Tự động crop và căn chỉnh lá cây trước khi phân loại
- Giảm ảnh hưởng của nền và các vật thể khác

### Nâng cao độ chính xác

- Sử dụng ensemble methods (multiple models)
- Áp dụng semi-supervised learning để tận dụng dữ liệu không nhãn
- Fine-tuning với dữ liệu thực tế thu thập được
- Thử nghiệm các kiến trúc mới hơn (EfficientNetV2, ConvNeXt)

### Tích hợp cloud

- Đồng bộ kết quả và dữ liệu lên cloud
- Cập nhật mô hình từ xa
- Phân tích dịch bệnh trên diện rộng
- Cảnh báo sớm cho nông dân và cơ quan chức năng

---

# TÀI LIỆU THAM KHẢO

```
TODO: Bổ sung tài liệu tham khảo theo chuẩn APA/IEEE

Gợi ý:
1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
3. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science.
4. TensorFlow Lite Documentation. https://www.tensorflow.org/lite
5. PyTorch Mobile Documentation. https://pytorch.org/mobile/
```

---

# PHỤ LỤC

## Phụ lục A. Cấu hình hệ thống

```yaml
# configs/config.yaml
model:
  architecture: "efficientnet_b2"
  num_classes: 39
  pretrained: true

image:
  size: 260
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  epochs: 20
  label_smoothing: 0.1
  gradient_clip: 1.0

augmentation:
  shadow_prob: 0.3
  fog_prob: 0.2
  rain_prob: 0.2
  motion_blur_prob: 0.3
  gauss_noise_prob: 0.4
  compression_prob: 0.3
  brightness_prob: 0.8
  grayscale_prob: 0.05
  random_erasing_prob: 0.25

quality_validation:
  blur_threshold: 100.0
  min_brightness: 30.0
  max_brightness: 220.0
  min_width: 224
  min_height: 224
```

## Phụ lục B. Các lệnh chạy chương trình

### Huấn luyện mô hình

```bash
cd SourceCode
python -m src.train --epochs 20 --batch-size 32 --lr 1e-4
```

### Đánh giá và xuất mô hình

```bash
python -m src.evaluate_and_convert
```

### Suy luận với TFLite

```bash
python -m src.inference --model plant_model_tflite_float32/plant_model.tflite --image test.jpg
```

### Tạo Grad-CAM visualization

```bash
python -m src.gradcam --image test.jpg --model models/best_model.pth --output gradcam_output.png
```

### Kiểm tra chất lượng ảnh

```bash
python -m src.quality_validator --image test.jpg --verbose
```

### Xuất metadata

```bash
python -m src.metadata --output-dir . --labels labels.json
```

### Xây dựng Android app

```bash
cd agrilens
./gradlew build
# Mở bằng Android Studio để run trên emulator/device
```

## Phụ lục C. Cấu trúc thư mục dự án

```
PlantDiseaseDetectionProject/
├── README.md
├── PROJECT_SUMMARY.md
├── CHANGELOG.md
├── report.md
├── SourceCode/
│   ├── src/
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── evaluate_and_convert.py
│   │   ├── gradcam.py
│   │   ├── quality_validator.py
│   │   ├── metadata.py
│   │   └── preprocessing/
│   │       ├── preprocess.py
│   │       └── generate_background_noise.py
│   ├── configs/
│   │   └── config.yaml
│   ├── data/
│   ├── models/
│   ├── saved_model/
│   ├── plant_model.onnx
│   ├── plant_model.pt
│   ├── labels.json
│   ├── labels.txt
│   ├── requirements.txt
│   └── README.md
├── agrilens/
│   ├── app/
│   │   └── src/main/
│   │       ├── java/com/example/
│   │       │   ├── MainActivity.java
│   │       │   ├── ImageClassifierHelper.java
│   │       │   ├── GuideActivity.kt
│   │       │   ├── DiseaseTreatmentRepository.java
│   │       │   └── OverlayView.java
│   │       └── assets/
│   │           ├── plant_model.pt
│   │           ├── labels.txt
│   │           └── labels.json
│   └── README.md
└── PlantDiseaseDetectionKnowledge/
    ├── 00_Index.md
    ├── 01 Inference_TFLite.md
    ├── 02 Inference_Preprocess.md
    └── ...