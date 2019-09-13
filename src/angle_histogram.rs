
pub struct AngleHistogram {
    pub bins: [usize; 256],
}

impl AngleHistogram {
    pub fn new() -> AngleHistogram {
        AngleHistogram { bins: [0; 256] }
    }
    pub fn add(&mut self, angle: u8, weight: i32) {
        self.bins[angle as usize] += weight as usize;
    }
    pub fn max(&self) -> usize {
        *self.bins.iter().max().unwrap()
    }
    pub fn resize_to(&mut self, size: usize) {
        let max = self.max();
        if max > 0 {
            for bin in self.bins.iter_mut() {
                *bin = (*bin * size) / max;
            }
        }
    }
    pub fn sum_quadrants(&mut self) {
        for i in 0..63 {
            self.bins[i] =
                self.bins[i] + self.bins[i + 64] + self.bins[i + 128] + self.bins[i + 192];
        }
    }
    pub fn main_angle_full(&self) -> u8 {
        let mut angle = 0;
        let mut max_value = 0;
        for i in 0..255 {
            let value = self.bins[i];
            if value > max_value {
                angle = i;
                max_value = value;
            }
        }
        angle as u8
    }
    pub fn main_angle(&self) -> u8 {
        let mut angle = 0;
        let mut max_value = 0;
        for i in 0..63 {
            let value = self.bins[i] + self.bins[i + 64] + self.bins[i + 128] + self.bins[i + 192];
            if value > max_value {
                angle = i;
                max_value = value;
            }
        }
        angle as u8
    }
    pub fn precise_main_angle_in_degrees(&self) -> f64 {
        let mid_angle = self.main_angle();
        let mut sum = 0f64;
        let mut weight = 0f64;
        for i in -2..2 {
            let w = self.bins[((mid_angle as i32 + i + 512) % 256) as usize] as f64;
            sum += (mid_angle as f64 + i as f64) * 90.0 / 64.0 * w;
            weight += w;
        }
        sum / weight
    }
}

