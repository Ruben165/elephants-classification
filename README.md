# African vs Asian Elephant Classification

Binary image classifier using transfer learning on a pre-trained **EfficientNetB0** (ImageNet).
Dataset: [Kaggle – Asian vs African Elephants](https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification/data)

## Structure
```
data/{train,test}/{African,Asian}   # images
impl/Elephants_Classification.ipynb # all code
result/                             # plots (samples, history, confusion matrix, predictions)
result/model/best_model.keras       # best checkpoint (lowest val_loss)
```

## Pipeline
1. **Data**: `data/train` is split 85/15 into train/validation; `data/test` is held out for final evaluation.
2. **Model**: augmentation (flip, rotate, zoom, translate, contrast), then EfficientNetB0 (avg pooling), dropout, and a sigmoid output.
3. **Training**: binary cross-entropy with label smoothing 0.1. Stage 1 trains the head (Adam 1e-3, 10 epochs). Stage 2 fine-tunes the top 60 base layers with BatchNorm frozen (Adam 1e-5, up to 15 epochs). EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint are used in both stages.
4. **Evaluation**: predictions average the original and horizontally flipped image (test-time augmentation). The decision threshold is tuned on the validation set, then accuracy, AUC, classification report, and confusion matrix are reported on the test set.
5. **Inference**: `predict_image(path)` loads the best model and classifies a single image.

All hyperparameters live in the constants cell.

## Usage
```bash
pip install tensorflow scikit-learn matplotlib
cd impl && jupyter nbconvert --to notebook --execute --inplace Elephants_Classification.ipynb
```
