# 🎯 Multi-Domain Aspect-Based Sentiment Analysis: STL vs MTL Comparison

Dự án so sánh hiệu quả của **Single-Task Learning (STL)** và **Multi-Task Learning (MTL)** trong phân tích cảm xúc đa khía cạnh (Aspect-Based Sentiment Analysis) trên nhiều domain bằng PhoBERT.

## 📊 Tổng quan

Aspect-Based Sentiment Analysis (ABSA) là nhiệm vụ phân tích cảm xúc theo từng khía cạnh cụ thể của một sản phẩm/dịch vụ. Ví dụ, với review "Đồ ăn ngon nhưng phục vụ chậm", ta cần phân tích:
- **FOOD#QUALITY**: Positive (ngon)
- **SERVICE#GENERAL**: Negative (chậm)

## 🎯 Mục tiêu nghiên cứu

So sánh 2 phương pháp:

### 🔹 Single-Task Learning (STL)
- Mỗi aspect có một model riêng biệt
- Ưu điểm: Tập trung cao cho từng aspect
- Nhược điểm: Không tận dụng được shared knowledge

### 🔹 Multi-Task Learning (MTL)  
- Một model chia sẻ cho tất cả aspects trong cùng domain
- Ưu điểm: Tận dụng shared knowledge giữa các aspects
- Nhược điểm: Có thể bị nhiễu từ các task khác

## 📁 Cấu trúc dự án

```
Lab10_1/
├── data/
│   ├── full_data.csv                 # Dữ liệu mỹ phẩm (Cosmetic)
│   ├── VLSP-Hotel/                   # Dữ liệu khách sạn VLSP 2018
│   │   ├── 1-VLSP2018-SA-Hotel-train.csv
│   │   ├── 2-VLSP2018-SA-Hotel-dev.csv
│   │   └── 3-VLSP2018-SA-Hotel-test.csv
│   └── VLSP-Restaurant/              # Dữ liệu nhà hàng VLSP 2018
│       ├── 1-VLSP2018-SA-Restaurant-train.csv
│       ├── 2-VLSP2018-SA-Restaurant-dev.csv
│       └── 3-VLSP2018-SA-Restaurant-test.csv
├── nlp-lab-10 (1).ipynb            # Notebook chính
├── README.md
├── requirements.txt
└── .gitignore
```

## 🛠️ Công nghệ sử dụng

- **Model**: PhoBERT (vinai/phobert-base) - BERT cho tiếng Việt
- **Framework**: PyTorch, Transformers
- **Metrics**: Accuracy, F1-score (weighted)
- **Hardware**: GPU/CPU với tối ưu hóa memory

## 📊 Domains và Aspects

### 1. 💄 Cosmetic Domain
- **stayingpower**: Độ bền của sản phẩm
- **texture**: Chất cảm sản phẩm  
- **smell**: Mùi hương
- **price**: Giá cả
- **colour**: Màu sắc
- **shipping**: Vận chuyển
- **packing**: Đóng gói

### 2. 🏨 Hotel Domain  
- **FACILITIES**: Tiện nghi (cleanliness, comfort, design, general)
- **ROOMS**: Phòng ở (cleanliness, comfort, design, general, quality)
- **SERVICE**: Dịch vụ chung
- **LOCATION**: Vị trí địa lý

### 3. 🍽️ Restaurant Domain
- **FOOD**: Đồ ăn (prices, quality, style&options)
- **DRINKS**: Đồ uống (prices, quality, style&options)  
- **AMBIENCE**: Không gian quán
- **SERVICE**: Phục vụ

## 🚀 Hướng dẫn chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy notebook

```bash
jupyter notebook "nlp-lab-10 (1).ipynb"
```

### 3. Thực thi comparison

```python
# Chạy so sánh đầy đủ
main()

# Hoặc phiên bản nhẹ cho memory hạn chế
lightweight_comparison()
```

## 📈 Kết quả

### Memory Optimization
- **Batch size**: 1-2 (tối ưu cho GPU memory)
- **Max length**: 96 tokens (giảm từ 256)
- **Gradient accumulation**: Effective batch size lớn hơn
- **CPU fallback**: Tự động chuyển sang CPU nếu GPU hết memory

### Performance Metrics
Model được đánh giá bằng:
- **Accuracy**: Độ chính xác tổng thể
- **F1-score**: F1 weighted cho multi-class

## 🔧 Tối ưu hóa Memory

Dự án được tối ưu hóa để chạy trên hardware hạn chế:

```python
# Cấu hình tối ưu memory
comparison = STL_vs_MTL_Comparison(
    model_name="vinai/phobert-base",
    batch_size=1,           # Giảm batch size
    max_length=96,          # Giảm sequence length  
    force_cpu=True          # Dùng CPU nếu cần
)
```

## 📚 Nguồn dữ liệu

- **VLSP 2018**: Vietnamese Language and Speech Processing Workshop
- **Cosmetic Dataset**: Dữ liệu review mỹ phẩm tiếng Việt

## 🤝 Đóng góp

1. Fork repository này
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`) 
5. Mở Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- **Tác giả**: [Tên của bạn]
- **Email**: [Email của bạn]
- **GitHub**: [GitHub username]

## 🙏 Acknowledgments

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Pre-trained BERT for Vietnamese
- [VLSP 2018](http://vlsp.org.vn/) - Vietnamese Language and Speech Processing Workshop
- [Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP library 