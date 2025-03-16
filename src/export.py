import os
import torch
import time
import onnx
from train import SimpleNN

def export_onnx(weights_path, img_size=(28, 28), batch_size=1, device='cpu'):
    # Load model từ file .pth
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Tạo input giả
    img = torch.zeros(batch_size, 1, *img_size).to(device)  # MNIST là ảnh 1 kênh (grayscale)

    # Thời gian bắt đầu
    t = time.time()

    # Export sang ONNX
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    onnx_file = weights_path.replace(".pth", ".onnx")
    
    torch.onnx.export(
        model,
        img,
        onnx_file,
        verbose=False,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}  # Hỗ trợ batch động
    )

    # Kiểm tra và lưu
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_file)
    print(f"ONNX export success, saved as {onnx_file}")
    print(f"Export complete (%.2fs). Visualize with https://github.com/lutzroeder/netron." % (time.time() - t))

    return onnx_file

if __name__ == '__main__':
    root = os.getcwd()
    model_path = os.path.join(root, "weights", "best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_file = export_onnx(model_path, img_size=(28, 28), batch_size=1, device=device)
    