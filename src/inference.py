import os
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import onnxruntime as ort

from train import SimpleNN

def load_image(image_path):
    """ Load ảnh và chuyển sang tensor. """
    transform = transforms.Compose([
        transforms.Grayscale(),  # Đảm bảo ảnh là grayscale
        transforms.ToTensor(),  
    ])
    image = Image.open(image_path)
    return transform(image)

def inference(model, image, label, device='cpu', plot = False):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Thêm batch dimension

    # Đo thời gian inference
    start_time = time.time()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # Chuyển sang millisecond

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')  # squeeze() để bỏ dimension thừa
        plt.title(f'Predicted: {predicted.item()}, Actual: {label}')
        plt.axis('off')
        plt.show()
    
    print(f'\nInference result:')
    print(f'Predicted: {predicted.item()}, Actual: {label}')
    print(f'Inference time: {inference_time:.4f} ms')

def inference_onnx(session, image, label, plot=False):
    """ Inference với mô hình ONNX """
    image_np = image.unsqueeze(0).numpy()  # Shape: (1, 1, 28, 28)
    input_name = session.get_inputs()[0].name

    # Đo thời gian inference
    start_time = time.time()
    outputs = session.run(None, {input_name: image_np})
    end_time = time.time()

    output_tensor = torch.from_numpy(outputs[0])
    _, predicted = torch.max(output_tensor, 1)

    inference_time = (end_time - start_time) * 1000  # Chuyển sang millisecond

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().numpy(), cmap='gray')  # squeeze() để bỏ dimension thừa
        plt.title(f'Predicted: {predicted.item()}, Actual: {label}')
        plt.axis('off')
        plt.show()
    
    print(f'\nONNX Inference result:')
    print(f'Predicted: {predicted.item()}, Actual: {label}')
    print(f'Inference time: {inference_time:.4f} ms')

def main_py(root):
    model_path = os.path.join(root, "weights", "best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    best_model = SimpleNN().to(device)
    best_model.load_state_dict(torch.load(model_path))

    for i in range(1, 10):
        image_path = os.path.join(root, "sample", f"sample_digit{str(i)}.png")
        sample_label = i

        sample_image = load_image(image_path)
        inference(best_model, sample_image, sample_label, device)

def main_onnx(root):
    onnx_model_path = os.path.join(root, "weights", "best_model.onnx")
    session = ort.InferenceSession(onnx_model_path)
    
    for i in range(1, 10):
        image_path = os.path.join(root, "sample", f"sample_digit{str(i)}.png")
        sample_label = i

        sample_image = load_image(image_path)
        inference_onnx(session, sample_image, sample_label)

if __name__ == '__main__':
    root = os.getcwd()
    # main_py(root)
    main_onnx(root)
    