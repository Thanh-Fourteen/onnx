# ONNX Model Conversion and Inference

This is a small experimental project focused on training a model, converting it from `.pth` to `ONNX` format, and further to `.ort` for optimized inference. The project includes scripts for training, exporting models, and performing inference on sample images.

## Project Structure

```
├── .gitignore
├── note.txt                # Project notes
├── sample                  # Sample images for testing
│   ├── sample7.png
│   ├── sample_digit1.png
│   ├── sample_digit2.png
│   ├── ...
│   └── sample_digit9.png
├── scripts                 # Utility scripts
│   ├── inferCpp.sh        # Shell script for C++ inference
│   └── wav2spec.sh        # Shell script for waveform to spectrogram conversion
└── src                     # Source code
    ├── export.py          # Script to convert .pth to ONNX
    ├── inference.cpp      # C++ inference with ONNX Runtime
    ├── inference.py       # Python inference script
    ├── makeSample.py      # Script to generate sample images
    ├── train.py           # Script to train the model
    └── wav2spec.cpp       # C++ code for waveform to spectrogram
```

## Prerequisites

- Python 3.8+
- PyTorch
- ONNX
- ONNX Runtime
- OpenCV (for image processing)
- GCC/G++ (for C++ inference)
- Librosa (optional, for audio processing in `wav2spec`)

Install the required Python dependencies:

```bash
pip install torch onnx onnxruntime opencv-python numpy
```

For C++ inference, ensure you have ONNX Runtime installed. Follow the [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/).

## Model Weights

Pre-trained model weights and converted ONNX/ORT files are available at:

- [Hugging Face Dataset](https://huggingface.co/datasets/Tony2202/onnx/tree/main)

Download the weights and place them in the appropriate directory (e.g., `weights/` if specified in your scripts).

## Usage

### 1. Training
To train the model, run:

```bash
python src/train.py
```

This will generate a `.pth` file containing the trained weights.

### 2. Exporting to ONNX
Convert the `.pth` model to ONNX format:

```bash
python src/export.py
```

This will output an ONNX model file (e.g., `model.onnx`).

### 3. Inference
#### Python Inference
Run inference on sample images using the ONNX model:

```bash
python src/inference.py --model model.onnx --image sample/sample_digit1.png
```

#### C++ Inference
Compile and run the C++ inference code:

```bash
g++ src/inference.cpp -o inference -lonnxruntime
./scripts/inferCpp.sh
```

### 4. Generating Sample Images
Create sample images for testing:

```bash
python src/makeSample.py
```

### 5. Waveform to Spectrogram
Convert audio waveforms to spectrograms (if applicable):

```bash
./scripts/wav2spec.sh
```

or compile and run the C++ version:

```bash
g++ src/wav2spec.cpp -o wav2spec
./wav2spec
```

## Notes
- The project is designed for experimentation with model conversion and inference optimization.
- Sample images in the `sample/` directory are used for testing inference.
- Check `note.txt` for additional project-specific details.
- For more details, refer to the source code on [GitHub](https://github.com/Thanh-Fourteen/onnx).

## Contributing
Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License.
