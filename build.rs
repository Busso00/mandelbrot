fn main() {
    cc::Build::new()
            .file("./src/f128cmp.c")
            .flag("-std=c11")
            .flag("-O2")
            .flag("-fno-inline")
            .compile("native");
    println!("building f128cmp");
}

