#!/usr/bin/env python3
"""
Test script to check GPU/MPS support in PyTorch environment.
This script will help verify if your setup can utilize Apple Silicon GPU acceleration.
"""

import torch
import sys
import platform

def test_pytorch_installation():
    """Test basic PyTorch installation and version."""
    print("=" * 60)
    print("PyTorch Installation Test")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

def test_mps_support():
    """Test Apple Metal Performance Shaders (MPS) support."""
    print("=" * 60)
    print("Apple MPS (Metal Performance Shaders) Support Test")
    print("=" * 60)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS backend is available!")
        
        # Check if MPS is built
        if torch.backends.mps.is_built():
            print("✅ MPS backend is built into PyTorch!")
            
            try:
                # Test creating a tensor on MPS device
                device = torch.device("mps")
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                
                # Perform a computation
                z = torch.matmul(x, y)
                print("✅ Successfully created tensors and performed computation on MPS device!")
                print(f"   Result tensor shape: {z.shape}")
                print(f"   Result tensor device: {z.device}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error using MPS device: {e}")
                return False
        else:
            print("❌ MPS backend is not built into this PyTorch installation")
            return False
    else:
        print("❌ MPS backend is not available on this system")
        return False

def test_cuda_support():
    """Test CUDA support (should not be available on Apple Silicon)."""
    print("=" * 60)
    print("CUDA Support Test")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("❌ CUDA is not available (expected on Apple Silicon)")
        return False

def performance_benchmark():
    """Simple performance benchmark comparing CPU vs MPS."""
    print("=" * 60)
    print("Performance Benchmark: CPU vs MPS")
    print("=" * 60)
    
    import time
    
    # Test parameters
    size = 2000
    iterations = 5
    
    # CPU benchmark
    print("Testing CPU performance...")
    cpu_times = []
    for i in range(iterations):
        start_time = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
    
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    print(f"   Average CPU time: {avg_cpu_time:.4f} seconds")
    
    # MPS benchmark (if available)
    if torch.backends.mps.is_available():
        print("Testing MPS performance...")
        mps_times = []
        device = torch.device("mps")
        
        for i in range(iterations):
            start_time = time.time()
            x_mps = torch.randn(size, size, device=device)
            y_mps = torch.randn(size, size, device=device)
            z_mps = torch.matmul(x_mps, y_mps)
            # Synchronize to ensure computation is complete
            torch.mps.synchronize()
            mps_time = time.time() - start_time
            mps_times.append(mps_time)
        
        avg_mps_time = sum(mps_times) / len(mps_times)
        print(f"   Average MPS time: {avg_mps_time:.4f} seconds")
        
        if avg_mps_time < avg_cpu_time:
            speedup = avg_cpu_time / avg_mps_time
            print(f"🚀 MPS is {speedup:.2f}x faster than CPU!")
        else:
            slowdown = avg_mps_time / avg_cpu_time
            print(f"⚠️  MPS is {slowdown:.2f}x slower than CPU (this can happen for small operations)")
    else:
        print("   MPS not available, skipping MPS benchmark")

def main():
    """Main function to run all tests."""
    print("🔍 Testing GPU/Acceleration Support for PyTorch")
    print()
    
    test_pytorch_installation()
    print()
    
    mps_available = test_mps_support()
    print()
    
    cuda_available = test_cuda_support()
    print()
    
    if mps_available:
        performance_benchmark()
        print()
    
    print("=" * 60)
    print("Summary and Recommendations")
    print("=" * 60)
    
    if mps_available:
        print("✅ Your system supports Apple MPS for GPU acceleration!")
        print("💡 To use MPS in your PyTorch code, use:")
        print("   device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')")
        print("   tensor = tensor.to(device)")
    elif cuda_available:
        print("✅ Your system supports CUDA for GPU acceleration!")
        print("💡 To use CUDA in your PyTorch code, use:")
        print("   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')")
        print("   tensor = tensor.to(device)")
    else:
        print("ℹ️  No GPU acceleration available. Using CPU only.")
        print("💡 Your code will run on CPU, which is still functional but slower for large models.")

if __name__ == "__main__":
    main()
