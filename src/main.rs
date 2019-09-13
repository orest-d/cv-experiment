extern crate opencv;

use opencv::core;
use opencv::highgui;
use opencv::imgproc;
use opencv::videoio;

pub mod angle_histogram;
use angle_histogram::*;

const CHARACTERISTICS_BUFFER_SIZE: usize = 640 * 480;
const REGION_SIZE: usize = 8;
const REGIONLINE_BUFFER_SIZE: usize = (640 / REGION_SIZE) * (480 / REGION_SIZE);
const HISTOGRAM_BINS: usize = 512;
const REDUCED_HISTOGRAM_BINS: usize = 100;

/*
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
*/

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

#[derive(Debug, Clone, Copy)]
pub struct Characteristics {
    pub angle: u8,
    pub intensity: u8,
}

pub struct CharacteristicsGrid {
    pub cols: usize,
    pub rows: usize,
    pub data: [Characteristics; CHARACTERISTICS_BUFFER_SIZE],
}

#[derive(Debug, Clone, Copy)]
pub enum RegionLine {
    Empty,
    LineFX { x: f32, y: f32, k: f32, weight: f32, x1:f32, x2:f32 },
    LineFY { x: f32, y: f32, k: f32, weight: f32, y1:f32, y2:f32 },
}


pub struct RegionLineGrid {
    pub cols: usize,
    pub rows: usize,
    pub data: [RegionLine; REGIONLINE_BUFFER_SIZE],
}

impl RegionLineGrid {
    fn new(cols: usize, rows: usize) -> RegionLineGrid {
        let len = rows * cols;
        assert!(len < REGIONLINE_BUFFER_SIZE);
        RegionLineGrid {
            rows: rows,
            cols: cols,
            data: [RegionLine::Empty; REGIONLINE_BUFFER_SIZE],
        }
    }
    fn get(&self, x: usize, y: usize) -> RegionLine {
        self.data[x + self.cols * y]
    }

    fn set(&mut self, x: usize, y: usize, value: RegionLine) {
        self.data[x * self.cols + y] = value
    }

    fn average_k(&self) -> (f32, f32) {
        let mut sumw = 0.0f32;
        let mut sumk = 0.0f32;
        let mut sumkk = 0.0f32;
        for rline in self.data.iter() {
            if let RegionLine::LineFX {
                x: x,
                y: y,
                k: k,
                weight: w,..
            } = rline
            {
                sumk += (*w * *k);
                sumkk += (*w * *k * *k);
                sumw += *w;
            }
            if let RegionLine::LineFY {
                x: x,
                y: y,
                k: k,
                weight: w,..
            } = rline
            {
                sumk += (*w * *k);
                sumkk += (*w * *k * *k);
                sumw += *w;
            }
        }
        let avgk = sumk / sumw;
        let avgkk = sumkk / sumw;
        let sigma = (avgkk - avgk * avgk).sqrt();
        (avgk, sigma)
    }

    fn neighbor_indices(x:usize,y:usize)->impl Iterator<Item=(usize,usize)>{
        (0..2).into_iter().flat_map(
            move |i| (0..2).into_iter().map(move |j| (i+x,j+y))
        )
    }
    /*
    fn neighbor_line_fx(&self, x:usize,y:usize)->Box<dyn Iterator<Item=(f32,f32,f32,f32,f32,f32)>>{
        let get = self.get;

        Box::new(RegionLineGrid::neighbor_indices(x,y).filter_map(move |(i,j)|{
            if let RegionLine::LineFX{x:x,y:y,k:k,weight:w,x1:t1,x2:t2} = get(i,j){
                Some((x,y,k,w,t1,t2))
            }
            else{
                None
            }    
        }
        ))
    }
    */
    fn extrapolate_to_neighbors(&self) -> RegionLineGrid{
        let (k, sigma) = self.average_k();
        let mut grid = RegionLineGrid::new(self.cols, self.rows);
        let sigma = sigma.max(0.01);
        let pi = std::f32::consts::PI;

        for i in 0..self.cols-2{
            for j in 0..self.rows-2{
                if let RegionLine::LineFX {
                    x: xa,
                    y: ya,
                    k: ka,
                    weight: wa,
                    x1:xa1,
                    x2:xa2
                } = self.get(i+1, j+1){
                    let ya1=ka*(xa1-xa)+ya;
                    let dxa1 = xa1-xa;
                    let dya1 = ya1-ya;
                    let angle_a = 128.0 + 128.0*dya1.atan2(dxa1)/pi;

                    let mut sumw = 0.0f32;
                    let mut sumx = 0.0f32;
                    let mut sumy = 0.0f32;
                    let mut cxx = 0.0f32;
                    let mut cyx = 0.0f32;
                    let mut minx = xa1.min(xa2);
                    let mut maxx = xa1.max(xa2);

                    for ii in 0..2{
                        for jj in 0..2{
                            if let RegionLine::LineFX {
                                x: xb,
                                y: yb,
                                k: kb,
                                weight: wb,
                                x1:xb1,
                                x2:xb2
                            } = self.get(i+ii, j+jj){
                                let yb1=kb*(xb1-xb)+yb;
                                let dxb1 = xb1-xb;
                                let dyb1 = yb1-yb;
                                let angle_b = 128.0 + 128.0*dyb1.atan2(dxb1)/pi;
                                let diff_angle_ab = angle_b-angle_a;
                                let mut w = (-diff_angle_ab*diff_angle_ab/8.0).exp();

                                let dx = xb-xa;
                                let dy = yb-ya;
                                let dr = (dx*dx+dy*dy).sqrt();
                                if dr>1.0{
                                    // y-ya = k(x-xa) v=(1,k) n=(-k,1)/sqrt(k*k+1)
                                    // dab = n*(xb-xa,yb-ya) = ((yb-ya)-k(xb-xa))/sqrt(k*k+1)
                                    let dab = ((yb-ya)-ka*(xb-xa))/(ka*ka+1.0).sqrt();
                                    if dab.abs()<5.0{
                                        let angle_ab = 128.0 + 128.0*dy.atan2(dx)/pi;
                                        let diff_angle_aba = angle_ab-angle_a;
                                        w *= (-diff_angle_aba*diff_angle_aba/8.0).exp();
                                        minx = minx.min(xb1).min(xb2);
                                        maxx = maxx.max(xb1).max(xb2);
                                        sumx += w*xb;
                                        sumy += w*yb;
                                    } 
                                }
                            }                                             
                        }
                    }
                }
            }
        }
        grid
    }
}

impl CharacteristicsGrid {
    fn new(cols: usize, rows: usize) -> CharacteristicsGrid {
        let len = rows * cols;
        assert!(len < CHARACTERISTICS_BUFFER_SIZE);
        CharacteristicsGrid {
            rows: rows,
            cols: cols,
            data: [Characteristics {
                angle: 0,
                intensity: 0,
            }; CHARACTERISTICS_BUFFER_SIZE],
        }
    }

    fn get(&self, x: usize, y: usize) -> Characteristics {
        self.data[x + self.cols * y]
    }

    fn set(&mut self, x: usize, y: usize, value: Characteristics) {
        self.data[x * self.cols + y] = value
    }

    fn evaluate_region(
        &self,
        x1: usize,
        y1: usize,
        dx: usize,
        dy: usize,
        mean_angle: u8,
        delta_angle: u8,
        weight_threshold: f32,
    ) -> RegionLine {
        let mut cx: f32 = 0.0;
        let mut cy: f32 = 0.0;
        let mut cxx: f32 = 0.0;
        let mut cyx: f32 = 0.0;
        let mut cyy: f32 = 0.0;
        let mut weight: f32 = 0.0;
        for i in 0..dx {
            let x = x1 + i;
            for j in 0..dy {
                let y = y1 + j;
                let c = self.get(x, y);
                if angle_in_range(c.angle, mean_angle, delta_angle) {
                    weight += c.intensity as f32;
                    cx += (x as f32) * (c.intensity as f32);
                    cy += (y as f32) * (c.intensity as f32);
                }
            }
        }
        if weight > weight_threshold {
            cx /= weight;
            cy /= weight;
            if mean_angle >= 224 || mean_angle < 32 || (mean_angle >= 96 && mean_angle < 160) {
                // horizontal
                for i in 0..dx {
                    let x = x1 + i;
                    for j in 0..dy {
                        let y = y1 + j;
                        let c = self.get(x, y);
                        if angle_in_range(c.angle, mean_angle, delta_angle) {
                            let xbar = (x as f32) - cx;
                            let ybar = (y as f32) - cy;
                            cxx += xbar * xbar * (c.intensity as f32);
                            cyx += ybar * xbar * (c.intensity as f32);
                        }
                    }
                }
                RegionLine::LineFX {
                    x: cx,
                    y: cy,
                    k: cyx / cxx,
                    weight: weight,
                    x1: x1 as f32,
                    x2: (x1+dx) as f32
                }
            } else {
                for i in 0..dx {
                    let x = x1 + i;
                    for j in 0..dy {
                        let y = y1 + j;
                        let c = self.get(x, y);
                        if angle_in_range(c.angle, mean_angle, delta_angle) {
                            let xbar = (x as f32) - cx;
                            let ybar = (y as f32) - cy;
                            cyy += ybar * ybar * (c.intensity as f32);
                            cyx += ybar * xbar * (c.intensity as f32);
                        }
                    }
                }
                RegionLine::LineFY {
                    x: cx,
                    y: cy,
                    k: cyx / cyy,
                    weight: weight,
                    y1: y1 as f32,
                    y2: (y1+dy) as f32
                }
            }
        } else {
            RegionLine::Empty
        }
    }
    fn make_regionline_grid(
        &self,
        mean_angle: u8,
        delta_angle: u8,
        weight_threshold: f32,
    ) -> RegionLineGrid {
        let mut grid = RegionLineGrid::new(self.rows / REGION_SIZE, self.cols / REGION_SIZE);
        for j in 0..self.rows / REGION_SIZE {
            for i in 0..self.cols / REGION_SIZE {
                let regionline = self.evaluate_region(
                    i * REGION_SIZE,
                    j * REGION_SIZE,
                    REGION_SIZE,
                    REGION_SIZE,
                    mean_angle,
                    delta_angle,
                    weight_threshold,
                );
                grid.set(i, j, regionline);
            }
        }
        grid
    }
    fn make_regionline_grid_symmetric(
        &self,
        mean_angle: u8,
        delta_angle: u8,
        weight_threshold: f32,
    ) -> RegionLineGrid {
        let mut grid = RegionLineGrid::new(self.rows / REGION_SIZE, self.cols / REGION_SIZE);
        for j in 0..self.rows / REGION_SIZE {
            for i in 0..self.cols / REGION_SIZE {
                let regionline = self.evaluate_region(
                    i * REGION_SIZE,
                    j * REGION_SIZE,
                    REGION_SIZE,
                    REGION_SIZE,
                    mean_angle,
                    delta_angle,
                    weight_threshold,
                );
                let rls = match regionline {
                    RegionLine::Empty => {
                        self.evaluate_region(
                            i * REGION_SIZE,
                            j * REGION_SIZE,
                            REGION_SIZE,
                            REGION_SIZE,
                            mean_angle.wrapping_add(128),
                            delta_angle,
                            weight_threshold,
                        )
                    },
                    _ => regionline,
                };
                grid.set(i, j, rls);
            }
        }
        grid
    }
}

fn angle_in_range(a: u8, mean: u8, delta: u8) -> bool {
    let a = a as i16;
    let mean = mean as i16;
    let delta = delta as i16;

    let mut x = a - mean;
    if x < -128 {
        x += 256;
    }
    if x > 128 {
        x -= 256;
    }
    if x < 0 {
        x = -x;
    }
    x < delta
}

fn convolution(cols: usize, rows: usize, source: &[u8], destination: &mut [u8]) -> RegionLineGrid {
    let n = 5_usize;

    let mut characteristics = CharacteristicsGrid::new(cols - n, rows - n);
    let mut maxintensity: i32 = 0;

    let a11 = 0usize;
    let a12 = 1usize;
    let a13 = 2usize;
    let a14 = 3usize;
    let a15 = 4usize;
    let a21 = cols + 0usize;
    let a22 = cols + 1usize;
    let a23 = cols + 2usize;
    let a24 = cols + 3usize;
    let a25 = cols + 4usize;
    let a31 = 2 * cols + 0usize;
    let a32 = 2 * cols + 1usize;
    let a33 = 2 * cols + 2usize;
    let a34 = 2 * cols + 3usize;
    let a35 = 2 * cols + 4usize;
    let a41 = 3 * cols + 0usize;
    let a42 = 3 * cols + 1usize;
    let a43 = 3 * cols + 2usize;
    let a44 = 3 * cols + 3usize;
    let a45 = 3 * cols + 4usize;
    let a51 = 4 * cols + 0usize;
    let a52 = 4 * cols + 1usize;
    let a53 = 4 * cols + 2usize;
    let a54 = 4 * cols + 3usize;
    let a55 = 4 * cols + 4usize;
    let pi = std::f64::consts::PI;

    let mut histogram = AngleHistogram::new();

    for i in 0usize..(rows - n) as usize {
        for j in 0usize..(cols - n) as usize {
            let top = i * cols + j;
            let x11: i32 = source[top + a11] as i32;
            let x12: i32 = source[top + a12] as i32;
            let x13: i32 = source[top + a13] as i32;
            let x14: i32 = source[top + a14] as i32;
            let x15: i32 = source[top + a15] as i32;
            let x21: i32 = source[top + a21] as i32;
            let x22: i32 = source[top + a22] as i32;
            let x23: i32 = source[top + a23] as i32;
            let x24: i32 = source[top + a24] as i32;
            let x25: i32 = source[top + a25] as i32;
            let x31: i32 = source[top + a31] as i32;
            let x32: i32 = source[top + a32] as i32;
            let x33: i32 = source[top + a33] as i32;
            let x34: i32 = source[top + a34] as i32;
            let x35: i32 = source[top + a35] as i32;
            let x41: i32 = source[top + a41] as i32;
            let x42: i32 = source[top + a42] as i32;
            let x43: i32 = source[top + a43] as i32;
            let x44: i32 = source[top + a44] as i32;
            let x45: i32 = source[top + a45] as i32;
            let x51: i32 = source[top + a51] as i32;
            let x52: i32 = source[top + a52] as i32;
            let x53: i32 = source[top + a53] as i32;
            let x54: i32 = source[top + a54] as i32;
            let x55: i32 = source[top + a55] as i32;
            let s = (114 * (x12 - x14 + x52 - x54)
                + 181 * (x22 - x24 + x42 - x44)
                + 228 * (x21 - x25 + x41 - x45)
                + 256 * (x31 + x32 - x34 - x35))
                / 25;
            let c = (114 * (x21 - x41 + x25 - x45)
                + 181 * (x22 - x42 + x24 - x44)
                + 228 * (x12 - x52 + x14 - x54)
                + 256 * (x13 + x23 - x43 - x53))
                / 25;

            let mut sum: i32 = s * s + c * c;
            sum /= 0xE0000;
            if sum > maxintensity {
                maxintensity = sum;
            }
            let a = (128f64 + 128f64 * (s as f64).atan2(c as f64) / pi) as u8;

            characteristics.set(
                i,
                j,
                Characteristics {
                    angle: a,
                    intensity: sum as u8,
                },
            );
            histogram.add(a, sum);
            /*
            for (p, kr) in kernel.iter().enumerate(){
                let start = start_top + (p*cols) as usize;
                let sr:&[u8] = &source[start .. start + n];
                sum+=kr.iter().zip(sr.iter()).map(|(k,x)| k*(*x as i32)).sum::<i32>();
            }
            */
            //sum/=655360;
            //sum/=32768;
            //sum+=(a as i32)/8;
            sum = if sum > 255 { 255 } else { sum };
            destination[i * cols + j] = sum as u8;
            //destination[i*cols+j]=a;
        }
    }
    histogram.resize_to(200);
    for (i, binsize) in histogram.bins.iter().enumerate() {
        for j in 0..*binsize {
            destination[i * cols + j] = 128 + destination[i * cols + j] / 2;
        }
        destination[i * cols + *binsize] = 255;
        destination[i * cols + *binsize + 1] = 255;
        destination[i * cols + *binsize + 2] = 255;
        destination[i * cols + *binsize + 3] = 255;
        destination[i * cols + *binsize + 4] = 0;
        destination[i * cols + *binsize + 5] = 0;
        destination[i * cols + *binsize + 6] = 0;
    }
    //println!("MAIN ANGLE:      {}", histogram.precise_main_angle_in_degrees());
    println!("MAIN ANGLE:      {}", histogram.main_angle());
    println!("MAIN ANGLE FULL: {}", histogram.main_angle_full());
    histogram.sum_quadrants();
    histogram.resize_to(200);
    for (i, binsize) in histogram.bins.iter().take(64).enumerate() {
        for j in 0..*binsize {
            destination[i * cols + cols - j] = 128 + destination[i * cols + cols - j] / 2;
        }
    }
    println!("MAXINTENSITY:    {}", maxintensity);
    characteristics.make_regionline_grid_symmetric(histogram.main_angle(), 5, 1000f32)
}

fn run() -> opencv::Result<()> {
    let window = "video capture";
    highgui::named_window(window, 1)?;
    println!("Window created");
    //   #[cfg(feature = "opencv-32")]
    let mut cam = videoio::VideoCapture::new(0)?; // 0 is the default camera
    println!("Camera created");
    //   #[cfg(not(feature = "opencv-32"))]
    //   let mut cam = videoio::VideoCapture::new_with_backend(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    let mut counter = 0;
    loop {
        counter = (counter + 1) % 50;
        println!("Loop");
        let mut frame = core::Mat::default()?;
        //        let mut frame = unsafe{core::Mat::new_rows_cols(300,200,core::CV_8UC3)}?;
        cam.read(&mut frame)?;

        /*
        //        let mut frame1 = unsafe{core::Mat::new_rows_cols(frame.rows()?,frame.cols()?,core::CV_8UC1)}?;
                let mut frame1 = core::Mat::new_rows_cols_with_default(frame.rows()?,frame.cols()?,core::CV_8UC1,core::Scalar::all(0.0))?;
                for i in 0..frame.rows()?{
                    for j in 0..frame.cols()?{
                        let mut ptr = unsafe{frame1.ptr_2d_mut(i,j)}?;
        //                let val:u8 = (frame.at_2d(i,j)?%256_i32) as u8;
                         let val = *unsafe{frame.ptr_2d(i,j)}?;
                        *ptr = 100u8+val%100u8;
                    }
                }
        */
        let mut gray = core::Mat::default()?;
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let rows = gray.rows()?;
        let cols = gray.cols()?;
        let len = (rows * cols) as usize;
        let mut slice = unsafe {
            let ptr = gray.data_mut()? as *mut u8;
            std::slice::from_raw_parts_mut(ptr, len)
        };
        let mut clone_gray = vec![0u8; len];
        clone_gray[..len].copy_from_slice(slice);
        let grid = convolution(cols as usize, rows as usize, &clone_gray[..len], slice);
        println!(
            "*************************************** K {:?}",
            grid.average_k()
        );
        let mut colored = core::Mat::default()?;
        imgproc::cvt_color(&gray, &mut colored, imgproc::COLOR_GRAY2BGR, 0)?;
        //        let color = core::Scalar::new(0.0,0.0,255.0,0.0);
        //        imgproc::line(&mut colored,core::Point::new(100,200),core::Point::new(200,300),color,3,8,0);
        //        imgproc::rectangle(&mut colored,core::Rect::new(100,200,30,30),color,3,0,0);
        for rline in grid.data.iter() {
            if let RegionLine::LineFX {
                x: x,
                y: y,
                k: k,
                weight: w,..
            } = rline
            {
                let color = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
                let color1 = core::Scalar::new(255.0, 0.0, 255.0, 0.0);
                let x1 = *x + 10.0;
                let y1 = *y - (x - x1) * (*k);
                imgproc::line(
                    &mut colored,
                    core::Point::new(*x as i32, *y as i32),
                    core::Point::new(x1 as i32, y1 as i32),
                    color1,
                    1,
                    8,
                    0,
                );
                imgproc::rectangle(
                    &mut colored,
                    core::Rect::new(*x as i32 - 1, *y as i32 - 1, 3, 3),
                    color,
                    1,
                    1,
                    0,
                );
            }
            if let RegionLine::LineFY {
                x: x,
                y: y,
                k: k,
                weight: w,..
            } = rline
            {
                let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0);
                let color1 = core::Scalar::new(255.0, 255.0, 0.0, 0.0);
                let y1 = *y + 10.0;
                let x1 = *x - (y - y1) * (*k);
                imgproc::line(
                    &mut colored,
                    core::Point::new(*x as i32, *y as i32),
                    core::Point::new(x1 as i32, y1 as i32),
                    color1,
                    1,
                    8,
                    0,
                );
                imgproc::rectangle(
                    &mut colored,
                    core::Rect::new(*x as i32 - 1, *y as i32 - 1, 3, 3),
                    color,
                    1,
                    1,
                    0,
                );
            }
        }

        let (k, sigma) = grid.average_k();
        let mut d0o: Option<f32> = None;
        let color = core::Scalar::new(255.0, 0.0, 0.0, 0.0);

        for rline in grid.data.iter() {
            if let RegionLine::LineFX { x, y, .. } = rline {
                let d = *y - k * (*x);
                if let Some(d0) = d0o {
//                    println!("delta {}", d - d0);
                    if (d - d0).abs() < 5.0 {
                        imgproc::rectangle(
                            &mut colored,
                            core::Rect::new(*x as i32 - 3, *y as i32 - 3, 5, 5),
                            color,
                            1,
                            1,
                            0,
                        );
                    }
                    if (d - d0).abs() > 20.0 && (d - d0).abs()<30.0{
                        let color = core::Scalar::new(255.0, 0.0, 128.0, 0.0);
                        imgproc::rectangle(
                            &mut colored,
                            core::Rect::new(*x as i32 - 3, *y as i32 - 3, 5, 5),
                            color,
                            1,
                            1,
                            0,
                        );
                    }
                } else {
                    println!(
                        "====================================>     D0 = {} K = {}",
                        d, k
                    );
                    d0o = Some(d);
                }
            }
        }
        let mut histogram = Histogram::new();
        /*
        for rline in grid.data.iter() {
            if let RegionLine::LineFX { x, y, .. } = rline {
                let d = *y - k * (*x);
                histogram.calibrate(d);
            }
        }
        */
        histogram.min=0.0;
        histogram.max=HISTOGRAM_BINS as f32;
        println!("D Min: {}   Max: {}   Delta: {}", histogram.min, histogram.max, histogram.max - histogram.min);
        if histogram.max - histogram.min > 5.0 {
            for rline in grid.data.iter() {
                if let RegionLine::LineFX {
                    x: x,
                    y: y,
                    weight: w,
                    ..
                } = rline
                {
                    let d = (*y - k * (*x))/(k*k+1.0).sqrt();
                    histogram.add(d, *w as u32);
                }
            }
            histogram.resize_to(256);
            histogram.distance_histogram();
            histogram.reduced_resize_to(256);
            for (i, b) in histogram.reduced_bins.iter().enumerate() {
                let color = core::Scalar::new(255.0, 128.0, 0.0, 0.0);
                imgproc::line(
                    &mut colored,
                    core::Point::new(3*i as i32,480),
                    core::Point::new(3*i as i32, 480-*b as i32),
                    color,
                    2,
                    8,
                    0,
                );
            }
        }

        let color = core::Scalar::new(128.0, 255.0, 128.0, 0.0);
        for rline in grid.data.iter() {
            if let RegionLine::LineFX {
                x: x,
                y: y,
                k: k,
                weight: w,
                x1: x1,
                x2: x2
            } = *rline
            {
                // y = kx + q => q = y - kx
                // y' - y = k(x' - x)
                // y'     = k(x' - x) + y 
                let y1 = k*(x1-x)+y;
                let y2 = k*(x2-x)+y;
                imgproc::line(
                    &mut colored,
                    core::Point::new(x1 as i32, y1 as i32),
                    core::Point::new(x2 as i32, y2 as i32),
                    color,
                    1,
                    8,
                    0,
                );
               
            }
        }

        /*
        for rline1 in grid.data.iter() {
            if let RegionLine::LineFX {x: x1, y: y1, k: k1, weight: w1,..} = rline1 {
                for rline2 in grid.data.iter() {
                    if let RegionLine::LineFX {x: x2, y: y2, k: k2, weight: w2,.. } = rline2 {
                        let dx = *x2 - *x1;
                        let dy = *y2 - *y1;
                        let d = (dy - *k1 * dx).abs()/((*k1 * *k1 + 1.0).sqrt());
                        let dr = (dx*dx+dy*dy).sqrt();
                        if d<3.0 && dr<20.0{
                            let k = dy/dx;
                            for rline3 in grid.data.iter() {
                                if let RegionLine::LineFX {x: x3, y: y3, k: k3, weight: w3,.. } = rline3 {
                                    let dx = *x3 - *x1;
                                    let dy = *y3 - *y1;
                                    let k3 = dy/dx;
                                    if (k3-k1).abs()<0.1{
                                        let d = (dy - k * dx).abs()/((k * k + 1.0).sqrt());
                                        let dr = (dx*dx+dy*dy).sqrt();
                                        if d<3.0 && dr<60.0{

                                            imgproc::line(
                                                &mut colored,
                                                core::Point::new(*x3 as i32, *y3 as i32),
                                                core::Point::new(*x2 as i32, *y2 as i32),
                                                color,
                                                1,
                                                8,
                                                0,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
*/
        //        highgui::imshow(window, &gray)?;
        //        println!("Frame size {:?}, {:?}", frame.size(), frame);
        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut colored)?;
        }
        if highgui::wait_key(10)? == 32 {
            println!("Break");
            break;
        }
    }
    Ok(())
}

fn main() {
    println!("Start");
    run().expect("something wrong happened");
    println!("End");
}