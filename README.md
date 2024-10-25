# Liveness Detection Project

This project focuses on detecting spoofing or liveness using image or video input. The detection is based on deep learning models, particularly the **ResNet** and **VGG16** architectures. The **ResNet model** performs significantly well with over **90% accuracy** on texture classification tasks, making it ideal for spoof detection, while the **VGG16 model** did not yield promising results.

## Project Structure

```bash
.
├── Training
│   ├── Resnet based training.ipynb        # Jupyter notebook for training the ResNet model
│   ├── VGG16 based training.py            # Python script for training the VGG16 model
│   └── model_deployment
│       ├── confi_model_test.py            # Script focusing on confidence scoring for live/spoof
│       └── model_test.py                  # Deployment script for liveness detection



## Model Performance

- **ResNet**: The **ResNet** model showed exceptional performance in texture-based classification for liveness detection, with **90%+ accuracy**. It performed well in detecting live and spoof instances with high confidence.

  - **Optimal threshold**: A threshold of **0.45** was found to be ideal for distinguishing between live and spoof images.

- **VGG16**: This model did not perform well in comparison to ResNet, showing lower accuracy and weaker performance in detecting spoofing attacks.

## Results

1. **ResNet** is performing well for texture classification.
2. A threshold of **0.45** is used for determining liveness.
3. The model has high confidence for spoof detection but lower confidence for live faces in comparison.

## Training Details

### ResNet Training Parameters:
```python
history = model.fit(
    training_set,
    epochs=25,  # Adjust the number of epochs as needed
    validation_data=validation_set
)
