import torch
print("Torch version:", torch.__version__)  # 检查 torch 版本
print("CUDA available:", torch.cuda.is_available())  # 检查 CUDA 是否可用
print("CUDA version:", torch.version.cuda)  # 检查 PyTorch 识别的 CUDA 版本
print("cuDNN version:", torch.backends.cudnn.version())  # 检查 cuDNN 版本
print("GPU device count:", torch.cuda.device_count())  # 检查可用 GPU 数量
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")