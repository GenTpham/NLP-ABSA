# 🎯 Multi-Domain Aspect-Based Sentiment Analysis: STL vs MTL Comparison

Dự án so sánh hiệu quả của **Single-Task Learning (STL)** và **Multi-Task Learning (MTL)** trong phân tích cảm xúc đa khía cạnh (Aspect-Based Sentiment Analysis) trên nhiều domain bằng PhoBERT.

## 📊 Tổng quan

Aspect-Based Sentiment Analysis (ABSA) là nhiệm vụ phân tích cảm xúc theo từng khía cạnh cụ thể của một sản phẩm/dịch vụ. Ví dụ, với review "Đồ ăn ngon nhưng phục vụ chậm", ta cần phân tích:
- **FOOD#QUALITY**: Positive (ngon)
- **SERVICE#GENERAL**: Negative (chậm)

## 🎯 Mục tiêu nghiên cứu

So sánh 2 phương pháp:

### 🔹 Single-Task Learning (STL)
**📍 Vị trí trong code**: Lines 82-97 (`STLModel`) và Lines 254-358 (`train_stl_models()`)

**🏗️ Kiến trúc**:
```python
class STLModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)  # MỘT classifier duy nhất
```

**🔄 Training Process**:
- **Mỗi aspect có model hoàn toàn riêng biệt**: `for aspect, data in domain_data.items()`
- **Train tuần tự**: Model cho `texture` → Model cho `smell` → Model cho `price`...
- **Độc lập hoàn toàn**: Không chia sẻ parameters giữa các aspects

**✅ Ưu điểm**: 
- Tập trung cao cho từng aspect cụ thể
- Không bị nhiễu từ aspects khác
- Dễ debug và tune cho từng task

**❌ Nhược điểm**: 
- Không tận dụng được shared knowledge
- Tốn nhiều memory và thời gian (N models cho N aspects)
- Không học được mối quan hệ giữa các aspects

### 🔹 Multi-Task Learning (MTL)  
**📍 Vị trí trong code**: Lines 100-117 (`MTLModel`) và Lines 370-525 (`train_mtl_model()`)

**🏗️ Kiến trúc**:
```python
class MTLModel(nn.Module):
    def __init__(self, model_name, aspect_classes, dropout_rate=0.3):
        # SHARED transformer encoder cho tất cả aspects
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # NHIỀU task-specific classifiers
        self.classifiers = nn.ModuleDict()
        for aspect, num_classes in aspect_classes.items():
            self.classifiers[aspect] = nn.Linear(hidden_size, num_classes)
```

**🔄 Training Process**:
- **Một model duy nhất cho tất cả aspects**: `model = MTLModel(self.model_name, aspect_classes)`
- **Train đồng thời**: Tất cả aspects được học cùng lúc trong mỗi batch
- **Shared feature extraction**: `shared_features = self.dropout(outputs.pooler_output)`
- **Multi-task loss**: `total_batch_loss += aspect_loss` cho tất cả aspects

**✅ Ưu điểm**: 
- Tận dụng shared knowledge giữa các aspects
- Hiệu quả memory (1 model thay vì N models)
- Học được correlation giữa các aspects
- Regularization effect từ multiple tasks

**❌ Nhược điểm**: 
- Có thể bị nhiễu từ các task khác
- Khó tune optimal cho từng aspect riêng lẻ
- Task imbalance có thể ảnh hưởng performance

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

## 🔄 So sánh Architecture STL vs MTL

### 📊 Sự khác biệt chính trong code:

| Aspect | STL | MTL |
|--------|-----|-----|
| **Model Class** | `STLModel` | `MTLModel` |
| **Training Function** | `train_stl_models()` | `train_mtl_model()` |
| **Number of Models** | N models (1 per aspect) | 1 model (shared) |
| **Classifier Layer** | `nn.Linear(hidden_size, num_classes)` | `nn.ModuleDict()` với nhiều classifiers |
| **Forward Output** | `logits` (single tensor) | `logits` (dictionary của tensors) |
| **Training Loop** | `for aspect in domain_data.items()` | `for aspect in aspect_classes` |
| **Loss Calculation** | Độc lập cho mỗi aspect | Cộng dồn tất cả aspect losses |

### 🏗️ Architecture Diagram:

```
STL Architecture:
[Input Text] → [PhoBERT] → [Classifier_texture] → [Output_texture]
[Input Text] → [PhoBERT] → [Classifier_smell] → [Output_smell]  
[Input Text] → [PhoBERT] → [Classifier_price] → [Output_price]
... (N separate models)

MTL Architecture:
[Input Text] → [Shared PhoBERT] → [Classifier_texture] → [Output_texture]
                                ↗ [Classifier_smell] → [Output_smell]
                                ↗ [Classifier_price] → [Output_price]
                                ↗ ... (all aspects)
```

### 🔧 Key Code Differences:

**STL Forward Pass:**
```python
def forward(self, input_ids, attention_mask):
    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
    output = self.dropout(pooled_output)
    logits = self.classifier(output)  # Single output
    return logits
```

**MTL Forward Pass:**
```python
def forward(self, input_ids, attention_mask):
    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    shared_features = self.dropout(outputs.pooler_output)  # Shared features
    
    logits = {}
    for aspect in self.classifiers:
        logits[aspect] = self.classifiers[aspect](shared_features)  # Multiple outputs
    return logits
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

## 📞 Liên hệ

- **Tác giả**: Pham Trung Truc
- **Email**: phamtruc120604@gamil.com
- **GitHub**: GenTpham

## 🙏 Acknowledgments

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Pre-trained BERT for Vietnamese
- [VLSP 2018](http://vlsp.org.vn/) - Vietnamese Language and Speech Processing Workshop
- [Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP library 