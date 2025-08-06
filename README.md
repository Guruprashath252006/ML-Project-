# ♻️ Plastic Type Classifier

A comprehensive AI-powered application for identifying plastic types and providing detailed recycling information. Built with PyTorch and Gradio.

## 🎯 Features

- **🤖 AI Classification**: Identifies 6 plastic types (PET, HDPE, PVC, LDPE, PP, PS)
- **♻️ Recycling Information**: Provides detailed recycling methods for each type
- **📦 Product Information**: Shows what each plastic becomes when recycled
- **🎨 Modern Interface**: Beautiful, user-friendly web interface
- **📊 Training Pipeline**: Complete model training with data augmentation
- **📈 Performance Metrics**: Training curves and model evaluation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Dataset
```bash
python setup_dataset.py
```

### 3. Train Model (Optional)
```bash
python train_model.py
```

### 4. Run Application
```bash
python plastic_classifier.py
```

### 5. Open Browser
- Go to: `http://localhost:7860`

## 📊 Supported Plastic Types

| Type | Full Name | Recycling Method | Common Products |
|------|-----------|------------------|-----------------|
| **PET** | Polyethylene Terephthalate | Curbside recycling | Beverage bottles, food containers |
| **HDPE** | High-Density Polyethylene | Curbside recycling | Milk jugs, detergent bottles |
| **PVC** | Polyvinyl Chloride | Specialized collection | Pipes, medical devices |
| **LDPE** | Low-Density Polyethylene | Store drop-off | Grocery bags, food wraps |
| **PP** | Polypropylene | Curbside recycling | Food containers, automotive parts |
| **PS** | Polystyrene | Special programs | Packaging, disposable cups |

## 🏗️ Project Structure

```
plastic-classifier/
├── plastic_classifier.py    # Main application
├── train_model.py          # Model training script
├── setup_dataset.py        # Dataset setup utility
├── requirements.txt        # Dependencies
├── README.md              # This file
├── dataset/               # Dataset directory
│   ├── PET/
│   ├── HDPE/
│   ├── PVC/
│   ├── LDPE/
│   ├── PP/
│   └── PS/
└── best_plastic_model.pth # Trained model (after training)
```

## 🎯 Dataset

This project uses the **"Dataset for Visual Plastic Type Recognition"** from Kaggle:
- **Author**: Harshit Kandoi
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/harshitkandoi/plastic-type-recognition)
- **Size**: 3,000+ images across 6 plastic types
- **Format**: JPG/PNG images organized by class

### Dataset Setup

1. **Download** the dataset from Kaggle
2. **Extract** the zip file
3. **Run** `python setup_dataset.py` to organize files
4. **Verify** the structure matches the expected format

## 🤖 Model Architecture

- **Backbone**: ResNet18 (pre-trained)
- **Transfer Learning**: Fine-tuned for plastic classification
- **Data Augmentation**: Random flips, rotations, color jittering
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers to prevent overfitting

## 📈 Training

### Training Parameters
- **Epochs**: 15 (configurable)
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Data Split**: 80% train, 20% validation

### Training Features
- **Progress Tracking**: Real-time loss and accuracy monitoring
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Visualization**: Training curves and performance metrics
- **Early Stopping**: Prevents overfitting

## 🎨 Interface Features

### Upload Section
- **Image Upload**: Drag & drop or click to upload
- **Multiple Formats**: Supports JPG, PNG, BMP, TIFF
- **Preview**: Real-time image preview
- **Clear Function**: Easy reset functionality

### Results Section
- **Plastic Type**: Identified type with confidence score
- **Recycling Method**: How to recycle this plastic
- **Recycled Products**: What it becomes when recycled
- **Description**: Detailed information about the plastic type

### Model Control
- **Initialize Model**: Load trained model
- **Status Display**: Model loading status
- **Error Handling**: Graceful error messages

## 🔧 Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **Gradio**: Web interface framework
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Scikit-learn**: Machine learning utilities

### Performance
- **Inference Speed**: ~100ms per image
- **Accuracy**: >90% on validation set
- **Memory Usage**: ~500MB for model + interface
- **GPU Support**: Automatic CUDA detection

## 🚀 Usage Examples

### Basic Classification
1. Upload a plastic bottle image
2. Click "Submit"
3. Get instant classification results

### Batch Processing
```python
from plastic_classifier import classify_plastic, load_model

model = load_model()
result = classify_plastic(image, model)
print(result)
```

### Custom Training
```python
from train_model import train_model

model = train_model(
    data_dir='dataset',
    epochs=20,
    batch_size=64,
    learning_rate=0.0005
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"No trained model found"**
   - Solution: Run `python train_model.py` first

2. **"Dataset not found"**
   - Solution: Run `python setup_dataset.py` and organize your dataset

3. **"Port already in use"**
   - Solution: Change port in `plastic_classifier.py` or kill existing process

4. **"CUDA out of memory"**
   - Solution: Reduce batch size in training or use CPU

### Performance Tips

- **GPU Training**: Use CUDA for faster training
- **Data Augmentation**: Improves model generalization
- **Batch Size**: Adjust based on available memory
- **Learning Rate**: Fine-tune for optimal convergence

## 📝 License

This project is developed for educational purposes by **GURUPRASHATH R (RA2311003020078)**.

## 👨‍💻 Developer

- **Name**: GURUPRASHATH R
- **Roll Number**: RA2311003020078
- **GitHub**: [Guruprashath252006](https://github.com/Guruprashath252006)
- **Kaggle**: [guruprashath2006](https://www.kaggle.com/guruprashath2006)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**♻️ Making the world greener, one plastic at a time! ♻️** 