# Model Checkpoints

This directory is where trained PyTorch model checkpoints (`.pth`) must be placed for FastAPI inference.

### Required Checkpoint Files:

1. **Image Deepfake Model**:
   - Filename: `deepfake_efficientnet.pth`
   - Architecture: EfficientNet-B0 (custom 2-class linear classifier)
   - Trained on: FaceForensics++ / Deepfake image dataset

2. **Video Deepfake Model**:
   - Filename: `deepfake_video_efficientnet_newdataset.pth`
   - Architecture: EfficientNet-B0 (custom 2-class linear classifier)
   - Evaluated on: Sampled facial frames extracted with OpenCV Haar Cascade

> **Note**: Due to GitHub file size limits and best practices for binary assets, `.pth` files are excluded from Git version control. You must train the models using `train.py` and `train_video_newdataset.py` or download pre-trained weights into this directory prior to starting the FastAPI server.
