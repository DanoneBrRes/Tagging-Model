import os, sys, subprocess, torch

print("Python:", sys.executable)
print("Platform:", sys.platform)
print("Torch version:", torch.__version__)
print("Torch.cuda.version (build):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())

try:
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
except Exception:
    pass

if torch.cuda.is_available():
    print("torch.cuda.device_count():", torch.cuda.device_count())
    try:
        di = torch.cuda.current_device()
        print("torch.cuda.current_device():", di)
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        print("torch.cuda.get_device_properties(0):", torch.cuda.get_device_properties(0))
    except Exception as e:
        print("Ошибка при получении свойств устройства:", e)

# Попробуем вызвать nvidia-smi (если доступно)
try:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    print("\nnvidia-smi stdout:\n", smi.stdout)
    print("nvidia-smi stderr:\n", smi.stderr)
except Exception as e:
    print("\nНе удалось запустить nvidia-smi:", e)
