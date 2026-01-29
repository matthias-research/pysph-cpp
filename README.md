# PySPH C++ Port

A C++ port of the Python [pysph](https://github.com/...) screen-space fluid rendering demo.

## Features

- **SPH Fluid Simulation** - Smoothed Particle Hydrodynamics with CUDA
  - Grid-based neighbor search using spatial hashing
  - Thrust-accelerated sorting
  - M4 (cubic spline) kernel for density estimation
  - Pressure and viscosity forces

- **Screen-Space Fluid Rendering** (Simon Green's technique)
  - Three render modes: Points, Balls, Advanced
  - Depth map with curvature flow smoothing
  - Thickness map with Gaussian blur
  - Final compositing with Beer's law absorption and Fresnel specular

## Requirements

- Windows 10/11
- Visual Studio 2019 or 2022
- CUDA 12.x (with CUDA build customizations for Visual Studio)
- vcpkg with the following packages:
  - `freeglut:x64-windows-static`
  - `glew:x64-windows-static`
  - `imgui[freeglut-binding,opengl3-binding]:x64-windows-static`

## Building

1. Install vcpkg dependencies:
   ```powershell
   cd ../vcpkg
   ./vcpkg install freeglut:x64-windows-static glew:x64-windows-static imgui[freeglut-binding,opengl3-binding]:x64-windows-static
   ```

2. Open `pysph-cpp.sln` in Visual Studio

3. Build (Release x64 recommended for performance)

## Usage

```
pysph-cpp.exe [num_particles]
```

Default is 8000 particles. Start with a lower number and increase based on your GPU.

## Controls

- **Left mouse drag**: Rotate view
- **Right mouse drag**: Translate view
- **Mouse wheel**: Zoom
- **H**: Toggle UI
- **Space**: Pause/Resume simulation
- **M**: Cycle render mode (Points → Balls → Advanced)
- **R**: Reset simulation
- **ESC**: Exit

## Render Modes

1. **Points**: Simple GL_POINTS rendering
2. **Balls**: Point sprites rendered as lit spheres
3. **Advanced**: Full screen-space fluid rendering with:
   - Depth smoothing (curvature flow)
   - Thickness-based transparency
   - Beer's law color absorption (water-like blue)
   - Fresnel reflections

## Parameters (in Advanced mode)

- **Smooth Depth**: Enable/disable curvature flow smoothing
- **Blur Thickness**: Enable/disable Gaussian blur on thickness map
- **Smoothing Iterations**: Number of curvature flow iterations (0-100)
- **Z Contribution**: Adaptive smoothing factor based on depth gradient

## Credits

- Original Python implementation: [pysph project]
- Screen-space rendering technique: Simon Green (NVIDIA)
- SPH theory: Matthias Müller et al.
