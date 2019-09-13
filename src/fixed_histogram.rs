
pub const HISTOGRAM_BINS: usize = 512;
pub const REDUCED_HISTOGRAM_BINS: usize = 100;


pub struct Histogram {
    pub bins: [u32; HISTOGRAM_BINS],
    pub reduced_bins: [u32; REDUCED_HISTOGRAM_BINS],
    pub min: f32,
    pub max: f32,
    pub initialized: bool,
}

impl Histogram {
    pub fn new() -> Histogram {
        Histogram {
            bins: [0; HISTOGRAM_BINS],
            reduced_bins: [0; REDUCED_HISTOGRAM_BINS],
            min: 0.0,
            max: 0.0,
            initialized: false,
        }
    }
    pub fn calibrate(&mut self, value: f32) {
        if self.initialized {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        } else {
            self.min = value;
            self.max = value;
            self.initialized = true;
        }
    }
    pub fn add(&mut self, value: f32, weight: u32) {
        if value>=self.min && value<=self.max{
            let index =
                (((HISTOGRAM_BINS - 1) as f32) * (value - self.min) / (self.max - self.min)) as usize;
            self.bins[index] += weight;
        }
    }
    pub fn max_bin_value(&self) -> u32 {
        *self.bins.iter().max().unwrap()
    }

    pub fn resize_to(&mut self, size: u32) {
        let max = self.max_bin_value();
        if max > 0 {
            for bin in self.bins.iter_mut() {
                *bin = (*bin * size) / max;
            }
        }
    }

    pub fn reduced_max(&self) -> u32 {
        *self.reduced_bins.iter().max().unwrap()
    }

    pub fn reduced_resize_to(&mut self, size: u32) {
        let max = self.reduced_max();
        if max > 0 {
            for bin in self.reduced_bins.iter_mut() {
                *bin = (*bin * size) / max;
            }
        }
    }

    pub fn distance_histogram(&mut self) {
        for i in 0..(REDUCED_HISTOGRAM_BINS-1) {
            for j in 0..(HISTOGRAM_BINS-1) {
                let index = i + j;
                if index >= HISTOGRAM_BINS {
                    break;
                }
                self.reduced_bins[i] += self.bins[index] * self.bins[j];
            }
        }
    }
}

