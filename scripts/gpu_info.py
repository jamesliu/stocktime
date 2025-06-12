#!/usr/bin/env python3
# scripts/gpu_info.py

"""
GPU Information and Capability Check for StockTime
"""

import torch
import subprocess
import sys

def get_gpu_info():
    """Get detailed GPU information"""
    print("🔍 GPU Information & Capability Check")
    print("=" * 50)
    
    # PyTorch CUDA info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {memory_gb:.1f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
            # Current memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_free = memory_gb - memory_reserved
                
                print(f"  Memory Allocated: {memory_allocated:.2f} GB")
                print(f"  Memory Reserved: {memory_reserved:.2f} GB") 
                print(f"  Memory Free: {memory_free:.2f} GB")
    else:
        print("❌ No CUDA-capable GPU detected")
        print("   System will run on CPU (slower but functional)")
    
    print("\n🤖 Model Memory Requirements:")
    print("-" * 30)
    print("microsoft/DialoGPT-small:     ~0.5 GB")
    print("microsoft/DialoGPT-medium:    ~1.5 GB") 
    print("microsoft/DialoGPT-large:     ~3.0 GB")
    print("meta-llama/Llama-2-7b-hf:     ~13-14 GB")
    print("meta-llama/Llama-2-13b-hf:    ~26 GB")
    
    # Recommendations
    if torch.cuda.is_available():
        max_memory = max(torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                        for i in range(torch.cuda.device_count()))
        
        print(f"\n💡 Recommendations for {max_memory:.1f}GB GPU:")
        print("-" * 40)
        
        if max_memory >= 16:
            print("✅ Can run Llama-2-7B comfortably")
            print("✅ Can run DialoGPT (all sizes)")
            print("🎯 Recommended: Try Llama-2-7B for best performance")
        elif max_memory >= 12:
            print("⚠️  Can run Llama-2-7B (tight fit)")
            print("✅ Can run DialoGPT (all sizes)")
            print("🎯 Recommended: DialoGPT-large or Llama-2-7B with caution")
        elif max_memory >= 8:
            print("❌ Cannot run Llama-2-7B reliably")
            print("✅ Can run DialoGPT-large")
            print("🎯 Recommended: DialoGPT-large for best balance")
        elif max_memory >= 4:
            print("❌ Cannot run large models")
            print("✅ Can run DialoGPT-medium")
            print("🎯 Recommended: DialoGPT-medium")
        else:
            print("❌ Limited GPU memory")
            print("✅ Can run DialoGPT-small")
            print("🎯 Recommended: DialoGPT-small (current choice)")
    
    # Try nvidia-smi for additional info
    try:
        print("\n🔧 nvidia-smi Output:")
        print("-" * 20)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract relevant lines
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'Quadro' in line:
                    print(line.strip())
        else:
            print("nvidia-smi not available")
    except:
        print("nvidia-smi not found")

def check_model_compatibility():
    """Check which models can run on current system"""
    print("\n🧪 Model Compatibility Test")
    print("=" * 30)
    
    models = [
        ("microsoft/DialoGPT-small", 0.5),
        ("microsoft/DialoGPT-medium", 1.5),
        ("microsoft/DialoGPT-large", 3.0),
        ("meta-llama/Llama-2-7b-hf", 13.5),
    ]
    
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_memory = available_memory * 0.8  # Leave 20% buffer
        
        print(f"Available GPU Memory: {available_memory:.1f} GB")
        print(f"Usable Memory (80%): {free_memory:.1f} GB\n")
        
        compatible_models = []
        for model_name, memory_req in models:
            if memory_req <= free_memory:
                status = "✅ Compatible"
                compatible_models.append(model_name)
            else:
                status = "❌ Too large"
            
            print(f"{model_name:<30} {memory_req:>6.1f} GB  {status}")
        
        print(f"\n🎯 Recommended Models for Your System:")
        for model in compatible_models[-2:]:  # Show top 2 compatible
            print(f"   • {model}")
    else:
        print("CPU mode - all models compatible but slower")

if __name__ == "__main__":
    get_gpu_info()
    check_model_compatibility()
