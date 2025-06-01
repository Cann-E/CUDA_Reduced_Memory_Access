# 🚀 CUDA_Reduced_Memory_Access

A collection of CUDA matrix multiplication kernels optimized to reduce global memory access and maximize GPU throughput. This project demonstrates step-by-step improvements starting from a naive implementation to a fully optimized kernel using shared memory, tiling, and thread coarsening.

---

## 🧠 Description

The project evaluates multiple CUDA matrix multiplication approaches (`A * B = C`) for large square matrices in row-major format. Starting with a naive row-wise version, it introduces memory coalescing, 2x2 thread coarsening, shared memory tiling, and finally, a fully optimized 4x4 tiled coarsened kernel. Each version is benchmarked for GFLOPS performance on increasing matrix sizes, with emphasis on reducing memory access and maximizing parallel throughput.

---

## 🔧 Tech Stack

- CUDA
- C++
- cuBLAS (optional comparison)
- Makefile-based build system

---

## ✨ Key Features

- ✅ `matmul1_naive`: Basic row-major matrix multiplication kernel  
- ✅ `matmul2_coalesced`: Improved indexing for global memory coalescing  
- ✅ `coarsened_matmul2x2`: Thread coarsening to process 2×2 blocks  
- ✅ `MatMulTiled`: Shared memory tiling with 16×16 thread blocks  
- ✅ `MatmulBest`: Fully optimized 4×4 coarsened, tiled kernel with shared memory  
- ✅ Kernel launcher that selects implementation based on input ID  
- ✅ Configurable with `make` and includes performance output

---

## 📂 Folder Structure

```
.
├── Makefile              # Build file
├── matmul                # Executable
├── template.cu           # Source code with all kernels
```

---

## 🛠️ Setup & Usage

### ✅ Requirements

- CUDA Toolkit 12.x  
- NVIDIA GPU (Compute Capability 3.5+)  
- Linux/macOS with `make` and `nvcc`

### ⚙️ Build

```bash
make
```

### ▶️ Run

```bash
./matmul
```

Kernels are launched based on an internal ID selector (`kernelId = 1` to `5`). You can modify the selected kernel in the launcher or main file if needed.

---

## 📈 Performance Tips

- Use `kernelId = 5` for the best performance (~10,000 GFLOPS on large matrices)
- Tune `BS` (block size) and `COARSE` (coarsening factor) for further gains
- Avoid kernelId 1 & 2 in production (for benchmarking only)

---

## 👥 Contributors

- Can Ercan (@cann-e)

---
