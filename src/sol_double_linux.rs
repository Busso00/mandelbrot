
use f128::f128;
use std::mem;


// Define a struct to hold 128-bit float bits with endianness awareness
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct F128Bits {
    lo: u64, // least significant bits
    hi: u64, // most significant bits
}

// Extract bits from __float128 (assuming __float128 is a C type)

fn float128_to_bits(f: f128) -> F128Bits {
    unsafe { mem::transmute_copy(&f) }
}

fn float128_to_ordered(bits: &mut F128Bits) {
    let sign = bits.hi >> 63;
    if sign != 0 {
        bits.hi = !bits.hi;
        bits.lo = !bits.lo;
    } else {
        bits.hi |= 0x8000000000000000;
    }
}

fn f128_eq(a: f128, b: f128) -> bool {
    let ba = float128_to_bits(a);
    let bb = float128_to_bits(b);
    ba == bb
}

fn f128_lt(a: f128, b: f128) -> bool {
    let mut ba = float128_to_bits(a);
    let mut bb = float128_to_bits(b);
    float128_to_ordered(&mut ba);
    float128_to_ordered(&mut bb);
    (ba.hi < bb.hi) || (ba.hi == bb.hi && ba.lo < bb.lo)
}

fn f128_cmp(a: f128, b: f128) -> i32 {
    if f128_lt(a, b) {
        -1
    } else if f128_eq(a, b) {
        0
    } else {
        1
    }
}

fn print_f128_bytes(num: f128) -> (){
    let bytes: [u8; 16] = unsafe { mem::transmute(num) };
    // Print bytes in hex (most common)
    print!("Bits of z_re + z_im: ");
    for b in bytes.iter().rev() { // reverse if you want MSB first
        print!("{:02x}", b);
    }
    println!();
}


fn main() -> () {
    // Just use the default options since the exact window size field names have changed
    // between eframe versions
    let a = f128::from(0.0);
    let b = f128::from(4.0);
    let c = f128::from(2.0);
    let d = f128::from(0.5);
    print_f128_bytes(a);
    print_f128_bytes(b);
    print_f128_bytes(c);
    print_f128_bytes(d);

    let values = [("a", a), ("b", b), ("c", c), ("d", d)];

    // Compare all pairs
    for (_, &(name_i, val_i)) in values.iter().enumerate() {
        for (_, &(name_j, val_j)) in values.iter().enumerate() {
            let cmp_res = f128_cmp(val_i, val_j);
            let cmp_str = match cmp_res {
                -1 => "<",
                0 => "==",
                1 => ">",
                _ => "?",
            };
            println!("{} {} {}", name_i, cmp_str, name_j);
        }
    }

}
