use hashbrown::HashMap;

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

impl Record {
    fn default() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            cum: 0.0,
            num: 0,
        }
    }

    fn add(&mut self, temperature: f32) {
        self.min = self.min.min(temperature);
        self.max = self.max.max(temperature);
        self.cum += temperature;
        self.num += 1;
    }

    fn average(&self) -> f32 {
        self.cum / self.num as f32
    }
}

/// Stable helper: split a slice at the first element matching `pred`.
/// Returns (left, right_without_sep). `None` if no separator found.
fn split_once<T, F>(s: &[T], mut pred: F) -> Option<(&[T], &[T])>
where
    F: FnMut(&T) -> bool,
{
    let i = s.iter().position(|x| pred(x))?;
    let (left, right_with_sep) = s.split_at(i);
    Some((left, &right_with_sep[1..]))
}

fn main() {
    let now = Instant::now();

    let filename = env::args().nth(1).unwrap_or("input.in".to_string());

    let mut file = File::open(filename).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    assert_eq!(buffer.pop(), Some(b'\n'));

    let mut city_records: HashMap<&[u8], Record> = HashMap::new();

    for line in buffer.split(|&c| c == b'\n') {
        let (city, temperature) =
            split_once(line, |&c| c == b';').expect("missing separator in line");

        let temperature = std::str::from_utf8(temperature)
            .expect("invalid utf-8 in temperature")
            .parse::<f32>()
            .expect("invalid floating point number");

        city_records
            .entry(city)
            .or_insert(Record::default())
            .add(temperature);
    }

    let mut results = city_records.into_iter().collect::<Vec<_>>();
    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for (city_name, record) in results {
        println!(
            "{:>16}: {:.1}/{:.1}/{:.1}",
            std::str::from_utf8(city_name).unwrap(),
            record.min,
            record.average(),
            record.max
        );
    }

    println!("Elapsed time: {}", now.elapsed().as_secs());
}
