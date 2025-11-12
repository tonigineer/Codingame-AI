use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};

type City = &'static str;
type Temperature = f32;

const CITIES: [City; 16] = [
    "Braunschweig",
    "Bremen",
    "Cuxhaven",
    "Emden",
    "Flensburg",
    "Hamburg",
    "Hannover",
    "Kiel",
    "Lübeck",
    "Oldenburg",
    "Osnabrück",
    "Rostock",
    "Schwerin",
    "Stralsund",
    "Wilhelmshaven",
    "Wolfsburg",
];

const TEMPERATURES: [Temperature; 37] = [
    19.9, 7.6, 25.4, -2.4, 13.6, 22.4, 0.1, 10.1, 29.9, 3.3, 18.5, 11.5, 24.3, 14.3, 9.3, 32.4,
    6.7, 21.5, 1.9, 26.7, 10.8, 5.7, 16.4, 23.3, 12.3, 37.1, 8.5, 4.6, -7.1, 28.1, 17.0, 13.0,
    20.7, 15.0, 17.7, 19.2, 15.7,
];

fn main() {
    let num_entries: usize = env::args()
        .nth(1)
        .and_then(|s| s.replace('_', "").parse().ok())
        .unwrap_or(1_000);

    let file = File::create("input.in").unwrap();
    let mut write_file = BufWriter::new(file);

    println!("Creating input with {} lines.", num_entries);

    // Writing line by line really simple and slow.
    for i in 0..num_entries {
        if (i + 1) % 1_000 == 0 {
            let percentage = (i + 1) as f32 / num_entries as f32 * 100.0;
            print!("\r\x1b[2KProgress: {:>6.2}%", percentage);
            io::stdout().flush().unwrap();
        }

        writeln!(
            write_file,
            "{};{:.1}",
            CITIES[i % CITIES.len()],
            TEMPERATURES[i % TEMPERATURES.len()]
        )
        .unwrap()
    }

    println!();
}
