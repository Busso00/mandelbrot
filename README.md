﻿# Mandelbrot rendering
- main OpenCL uses OpenCL on Linux/MinGW relying respectively on f128::f128/std::f128
- sol_double_linux is a bug fix to compare f128::f128 variables
- implementation on Rayon is similar in performances with GPU version only when using f64
- easiest version to run is main_rayon (openCL require to manually install libraries .a)
- quadmath is not supported in MSVC and can be used only with Linux/MinGW
- mingw must run with nightly toolchain
