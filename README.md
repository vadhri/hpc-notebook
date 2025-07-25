# CUDA Programming Notebook

This repository contains a collection of notebooks and source code files for learning and experimenting with CUDA programming, parallel computing, and high-performance computing (HPC) concepts. The materials are organized by topic and include both Python and C/CUDA examples, as well as Jupyter notebooks for interactive exploration.


## Contents Overview

| File / Notebook                                      | Description                                                                 |
|------------------------------------------------------|-----------------------------------------------------------------------------|
| `hello.cu`                                           | A minimal example that prints a message from both the CPU and GPU, demonstrating the basics of CUDA kernel launch and device management. |
| `cuda_hello.ipynb`                                   | Interactive walkthrough of a CUDA hello world example, including code, explanation, and output, to introduce GPU programming concepts. |
| `Math/Cuda_parallelism.ipynb`                        | Step-by-step exploration of CUDA parallelism, showing how to implement and optimize vector addition using different block and thread configurations. |
| `Math/Matrix/MatrixTranspose.ipynb`                  | Detailed explanation and code for performing matrix transpose operations on the GPU, including performance considerations. |
| `Math/Matrix/matrix_transpose.cu`                    | Source code for transposing a matrix using CUDA, illustrating memory access patterns and parallelization. |
| `Math/Matrix/multiprocess_py_matrix_mul.ipynb`       | Demonstrates matrix multiplication using Python's multiprocessing module, comparing CPU and parallel approaches. |
| `Math/Monte-Carlo/Metropolis_algo.ipynb`             | Implements the Metropolis algorithm for Monte Carlo simulations, with explanations of statistical sampling and convergence. |
| `Math/PDE/2d_heat_diffusion.ipynb`                   | Solves the 2D heat diffusion partial differential equation using numerical methods and visualizes the results. |
| `Math/PDE/finite_diff_sin.ipynb`                     | Uses the finite difference method to approximate derivatives of the sine function and analyze accuracy. |
| `pde_fft_1d_heat_equation.ipynb`                     | Solves the 1D heat equation using Fast Fourier Transform (FFT) techniques for efficient computation. |
| `FourierTransforms_1D_2D_3D.ipynb`                   | Provides examples and visualizations of Fourier transforms in one, two, and three dimensions. |
| `mpi.ipynb`                                          | Introduces the basics of MPI (Message Passing Interface) for distributed memory parallel programming, with runnable code samples. |
| `finite_difference_mpi.ipynb`                        | Shows how to implement parallel finite difference methods using MPI, including communication patterns. |
| `openacc.c`                                          | Example C code using OpenACC directives to parallelize computations for accelerators like GPUs. |
| `Multiple GPUs/01-page-faults.cu`                    | Demonstrates how page faults can occur in multi-GPU environments and how to handle them in CUDA. |
| `Multiple GPUs/01-vector-add-sm-optimized.cu`         | An optimized version of vector addition for multi-GPU systems, focusing on shared memory and performance. |
| `Multiple GPUs/01-vector-add.cu`                     | Implements vector addition across multiple GPUs, illustrating device management and data transfer. |
| `Multiple GPUs/nbody.cu`                             | Simulates the N-body problem using CUDA, distributing computation across multiple GPUs for scalability. |
| `Multiple GPUs/saxpy.cu`                             | Performs the SAXPY operation (Single-Precision A·X Plus Y) in a multi-GPU setup, showing parallel execution. |
| `Multiple GPUs/vector_add.cu`                        | Another example of vector addition, highlighting differences in implementation for multi-GPU systems. |
| `Multiple GPUs/README.md`                            | Documentation and instructions specific to the multi-GPU code examples in this directory. |
| `Multiple GPUs/Profiling/pagefault_profiling.ipynb`  | Analyzes and profiles page faults in CUDA applications, with step-by-step profiling workflow. |
| `Multiple GPUs/Profiling/profiling-optimization.ipynb`| Explores techniques for profiling and optimizing CUDA code, including tool usage and performance tips. |
| `Multiple GPUs/Profiling/profiling.ipynb`            | Introduces CUDA profiling tools and demonstrates how to interpret profiling results. |
| `Multiple GPUs/Profiling/Streaming and Visual Profiling.ipynb` | Explains and demonstrates streaming and visual profiling techniques for CUDA programs. |
| `python_multi_processing.ipynb`                      | Explores parallel computation in Python using the multiprocessing module, with practical examples and benchmarks. |
| `start-fs.sh`                                       | Bash script to initialize or mount a file system or set up the environment for experiments. |

## How to Use

- Explore the Jupyter notebooks (`.ipynb`) for interactive explanations and code.
- CUDA source files (`.cu`) can be compiled with `nvcc` and run on a CUDA-capable GPU.
- Some notebooks and scripts require Python and additional packages (see notebook cells for details).

---


## Compiling and Running CUDA Code

To compile a CUDA source file (e.g., `vector_add.cu`), use the NVIDIA CUDA Compiler (`nvcc`):

```bash
nvcc vector_add.cu -o vector_add
```

This will produce an executable named `vector_add`. To run the executable:

```bash
./vector_add
```

Make sure you have a CUDA-capable GPU and the CUDA toolkit installed on your system. For more advanced examples, see the relevant notebook or source file for specific compilation flags or input requirements.

---

For more details on each topic, refer to the corresponding notebook or source file. Contributions and suggestions are welcome!


