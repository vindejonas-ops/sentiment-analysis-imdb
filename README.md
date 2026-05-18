# IMDB Sentiment Analysis with DeBERTa + MLP

## Model Architecture
- **Feature Extractor**: microsoft/deberta-v3-small
- **Pooling**: Mean + Max concatenation
- **Classifier**: MLP (1024 → 512 → 256 → 1)

## Performance
- Dataset: IMDB balanced 10k
- Target Accuracy: ≥95%

## Usage
```bash
cd src && python extract_features.py
cd src && python train_mlp.py
python src/predict.py "This movie is fantastic!"