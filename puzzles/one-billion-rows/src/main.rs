use std::collections::HashMap;

use std::env;
use std::fs::File;
use std::io::Read;

use std::time::Instant;

#[derive(Debug)]
struct Record {
    min: f32,
    max: f32,
    cum: f32,
    num: usize,
}

fn main() {
    let now = Instant::now();

    let filename = env::args().nth(1).unwrap_or("input.in".to_string());

    let mut file = File::open(filename).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();

    let mut city_records: HashMap<String, Record> = HashMap::new();

    for line in buffer.lines() {
        let (city, temperature) = line.split_once(";").unwrap();
        let city = city.to_string();
        let temperature = temperature.parse::<f32>().unwrap();

        if let Some(r) = city_records.get_mut(&city) {
            r.min = r.min.min(temperature);
            r.max = r.max.max(temperature);
            r.cum += temperature;
            r.num += 1;
        } else {
            city_records.insert(
                city.clone(),
                Record {
                    min: 0.0,
                    max: 0.0,
                    cum: 0.0,
                    num: 0,
                },
            );
        }
    }

    let mut results = city_records.into_iter().collect::<Vec<_>>();
    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for (city_name, record) in results {
        println!(
            "{}: {:.1}/{:.1}/{:.1}",
            city_name,
            record.min,
            record.cum / record.num as f32,
            record.max
        );
    }

    println!("Took: {}", now.elapsed().as_secs());
}
