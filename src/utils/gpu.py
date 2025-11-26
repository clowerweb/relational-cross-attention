"""
GPU Detection and Setup Utilities
"""

import torch


def detect_gpu():
    """Detect and report GPU backend (ROCm/CUDA/CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)

        # Check if it's ROCm (AMD GPU)
        is_rocm = 'AMD' in device_name or 'Radeon' in device_name or hasattr(torch.version, 'hip')

        print("=" * 60)
        if is_rocm:
            print("[ROCm] GPU DETECTED!")
            if hasattr(torch.version, 'hip'):
                print(f"   HIP Version: {torch.version.hip}")
        else:
            print("[CUDA] GPU DETECTED")
            print(f"   CUDA Version: {torch.version.cuda}")

        print(f"   Device: {device_name}")
        print(f"   Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"   Total Memory: {device_props.total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        return torch.device("cuda"), is_rocm
    else:
        print("=" * 60)
        print("[WARNING] NO GPU DETECTED - Using CPU")
        print("=" * 60)
        return torch.device("cpu"), False
