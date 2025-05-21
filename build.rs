fn main() {
    // Compile the C source
    cc::Build::new()
        .file("./src/f128cmp.c")
        .flag("-std=c17")
        .flag("-O2")
        .flag("-fno-inline")
        .compile("native");
    println!("building f128cmp");

    // Add the directory containing libOpenCL.a to the linker search path
    println!("cargo:rustc-link-search=native=C:/Users/fedeb/Downloads/mandelbrot/lib");

    // Link statically to libOpenCL.a (note: no 'lib' prefix or '.a' suffix)
    println!("cargo:rustc-link-lib=static=OpenCL");
    println!("cargo:rustc-link-lib=static=quadmath");
     
}
