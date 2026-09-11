# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU (Tesla T4)

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.5.1 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}`
- Record này mô tả toàn ảnh như thế nào?
  Record này gán một nhãn phân loại duy nhất (`cab` - xe taxi) cho toàn bộ bức ảnh ở mức tổng thể (image-level), bỏ qua việc bức ảnh thực tế chứa rất nhiều đối tượng và loại phương tiện khác nhau cùng xuất hiện.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Bộ dữ liệu huấn luyện ImageNet-1K định nghĩa class list gồm 1000 danh mục cố định; mô hình không tự sáng tạo ra tên lớp mà chỉ dự đoán xác suất trên danh mục có sẵn này.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  Cần giữ cả ba trường để đảm bảo tính toàn vẹn và không mơ hồ về mặt ngữ nghĩa: `class_id` là mã định danh số trong mô hình, `class_name` giúp con người đọc hiểu, còn `taxonomy_name` xác định ngữ cảnh ngữ nghĩa (tránh nhầm lẫn khi cùng một tên lớp nhưng thuộc taxonomy khác như COCO hay OpenImages có định nghĩa phạm vi khác nhau).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ nguyên tắc chọn nhãn ưu tiên: ví dụ chọn chủ thể chiếm diện tích pixel lớn nhất, chủ thể nằm ở trọng tâm ảnh, hoặc chuyển bài toán sang multi-label classification / object detection thay vì ép gán nhãn đơn (single-label).
- Vì sao model score không phải ground truth?
  Model score chỉ là giá trị xác suất (softmax probability) do mạng nơ-ron tính toán dựa trên trọng số đã huấn luyện; nó phản ánh mức độ tự tin của mô hình chứ không phải sự thật khách quan đã được con người kiểm chứng và gắn nhãn theo guideline chuẩn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}`
- Diễn giải vị trí box bằng lời:
  Hộp bao quanh người đầu bếp đang đứng quay lưng ở khu vực trung tâm bên phải gian bếp. Tọa độ góc trên bên trái của hộp nằm tại pixel (x=385.33, y=69.24) và góc dưới bên phải tại pixel (x=498.92, y=348.92), với chiều rộng hộp là 113.58 pixel và chiều cao là 279.68 pixel.
- So sánh số prediction ở hai threshold:
  Ở ngưỡng mặc định 0.35, mô hình phát hiện 11 vật thể (gồm người, các khuôn bát, lò nướng, cốc). Khi nâng ngưỡng lên cao (ví dụ 0.50 - 0.70), số lượng prediction giảm xuống còn khoảng 4-5 vật thể tự tin nhất (người 0.91, hai bát 0.72 và 0.70), loại bỏ các vật thể có độ tự tin thấp như `cup 0.45` hay `bowl 0.46`. Ngược lại khi hạ ngưỡng xuống 0.25, số lượng box tăng lên đáng kể.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Hạ threshold làm tăng độ bao phủ (coverage / recall) nhằm tránh bỏ sót đối tượng nhỏ hoặc mờ, nhưng đồng thời sinh ra nhiều dự đoán sai (false positives), làm tăng khối lượng kiểm tra và áp lực rà soát của reviewer. Nâng threshold giúp reviewer đỡ việc nhưng tăng nguy cơ bỏ sót vật thể thực tế (false negatives).
- Đề xuất một quy tắc box chặt:
  Bounding box phải ôm khít các pixel biên ngoài cùng nhìn thấy được của vật thể (dung sai không quá 2 pixel), không bao gồm bóng đổ của vật thể và không được lấy dư khoảng trống nền xung quanh.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Cần guideline quy định tỷ lệ diện tích nhìn thấy tối thiểu để được gán nhãn (ví dụ: trường hợp bàn tay người thò vào ở mép trái ảnh `kitchen` có score 0.61 - guideline cần nêu rõ chỉ gán nhãn `person` nếu nhìn thấy tối thiểu bao nhiêu % cơ thể hay có phần đầu/thân); các trường hợp không đủ thông tin nhận dạng bắt buộc phải escalation xin ý kiến phê duyệt.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 71.0], [442.0, 72.0]]}`
- Polygon bổ sung chi tiết gì so với box?
  Polygon cung cấp ranh giới hình học chính xác bám sát từng đường viền cơ thể và trang phục của người (tóc, bờ vai, quai đeo, tạp dề, ống quần), loại bỏ hoàn toàn các pixel nền trống (background) vốn bị gộp chung bên trong bounding box hình chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` dùng để định danh và phân biệt riêng biệt từng cá thể đối tượng trong cùng một ảnh (ví dụ phân biệt `kitchen-002` và `kitchen-003` đều là `bowl`). `instance_id` không phải là `class_id` (mã số loại đối tượng) và không phải là `tracking_id` (mã theo vết đối tượng xuyên suốt qua chuỗi video).
- Đề xuất một quy tắc biên mask:
  Đường biên polygon phải khép kín và đi sát mép ngoài của vật thể (sai số tối đa 2-3 pixel), tách bạch phần tiếp xúc giữa các vật thể liền kề và phải đục lỗ (loại trừ) các khoảng trống nền nhìn xuyên qua (như khoảng hở giữa hai chân người hoặc quai treo kim loại).
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Guideline cần quy định ranh giới cho các vùng chuyển tiếp mờ (ví dụ bột mì vương trên mặt bàn làm việc, các chùm lá thảo mộc khô treo bị mô hình nhận nhầm thành `potted plant 0.63`); khi gặp đối tượng bị che khuất một phần hoặc ranh giới phân tách không rõ ràng, annotator phải escalate để nhận quy chuẩn thống nhất thay vì tự suy diễn ranh giới.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn cấp ảnh (`class_id` + `class_name` thuộc taxonomy chuẩn) | Ảnh `traffic` có nhiều chủ thể phức tạp (xe buýt, xe con, người) nhưng mô hình chỉ đoán nhãn `cab` | Tuân theo guideline ưu tiên chủ thể chính (diện tích lớn nhất hoặc ở trung tâm); nếu không có quy tắc, escalate xin hướng dẫn | Kiểm tra tính nhất quán của việc chọn nhãn theo đúng quy tắc ưu tiên; không chấp nhận việc chọn nhãn theo cảm tính |
| Phát hiện vật thể | Bounding box 2D (`[x_min, y_min, x_max, y_max]` theo pixel) kèm `class_id` | Bàn tay ở góc dưới mép trái bị đóng box gán nhãn `person 0.61`; tủ kim loại bị nhận nhầm là `oven` | Kiểm tra tỷ lệ nhìn thấy của người; không đóng box `person` cho bàn tay lẻ loi nếu chưa đủ điều kiện nhận diện theo guideline | Soát lại độ khít của box (không dư nền, không xén vật thể), kiểm tra kỹ các ca cắt mép/che khuất xem có vi phạm tiêu chuẩn không |
| Instance segmentation | Tập hợp tọa độ đa giác khép kín (`polygon_xy` pixel) + `instance_id` + `class_id` | Bó thảo mộc khô treo bị nhận nhầm thành `potted plant 0.63`; mặt bàn bị phủ đè lên các vật dụng khác | Vẽ polygon bám sát đường bao thực tế của đối tượng; bỏ qua bó lá khô nếu không có class tương ứng trong guideline | Phóng to kiểm tra độ chính xác của đường biên mask, rà soát các điểm tiếp xúc/chồng lấn giữa các instance và độ sạch của viền |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Tuyệt đối không tải lên, lưu trữ hoặc chia sẻ công khai dữ liệu chứa thông tin định danh cá nhân (PII như khuôn mặt nhạy cảm, biển số xe chưa ẩn danh), dữ liệu khách hàng hoặc tài liệu nội bộ ra ngoài các môi trường/kho lưu trữ chưa được cấp phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Giảng viên phụ trách, Lab Lead hoặc Quản lý dự án (Project Manager) của chương trình đào tạo để tiến hành cô lập và xử lý dữ liệu theo quy trình escalation an toàn.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
