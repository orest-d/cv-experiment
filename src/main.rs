extern crate opencv;

use opencv::core;
use opencv::highgui;
use opencv::imgproc;
use opencv::videoio;

const CHARACTERISTICS_BUFFER_SIZE: usize = 640 * 480;
const REGION_SIZE: usize = 10;
const REGIONLINE_BUFFER_SIZE: usize = (640 / REGION_SIZE) * (480 / REGION_SIZE);

pub struct Histogram {
    pub bins: [usize; 256],
}

impl Histogram {
    pub fn new() -> Histogram {
        Histogram { bins: [0; 256] }
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
    LineFX { x: f32, y: f32, k: f32, weight: f32 },
    LineFY { x: f32, y: f32, k: f32, weight: f32 },
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

    fn average_k(&self) -> (f32, f32){
        let mut sumw = 0.0f32;
        let mut sumk = 0.0f32;
        let mut sumkk = 0.0f32;
        for rline in self.data.iter(){
            if let RegionLine::LineFX{x:x,y:y,k:k,weight:w}=rline{
                sumk+= (*w * *k);
                sumkk+= (*w * *k * *k);
                sumw+= *w;
            }
            if let RegionLine::LineFY{x:x,y:y,k:k,weight:w}=rline{
                sumk+= (*w * *k);
                sumkk+= (*w * *k * *k);
                sumw+= *w;
            }
        }
        let avgk = sumk/sumw;
        let avgkk = sumkk/sumw;
        let sigma = (avgkk-avgk*avgk).sqrt();
        (avgk, sigma)
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
                let regionline = self.evaluate_region(i*REGION_SIZE, j*REGION_SIZE, REGION_SIZE, REGION_SIZE, mean_angle, delta_angle, weight_threshold);
                grid.set(i,j,regionline);
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

    let mut histogram = Histogram::new();

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
    characteristics.make_regionline_grid(histogram.main_angle(), 5, 1000f32)
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
        println!("*************************************** K {:?}", grid.average_k());
        let mut colored = core::Mat::default()?;
        imgproc::cvt_color(&gray, &mut colored, imgproc::COLOR_GRAY2BGR, 0)?;
//        let color = core::Scalar::new(0.0,0.0,255.0,0.0);
//        imgproc::line(&mut colored,core::Point::new(100,200),core::Point::new(200,300),color,3,8,0);
//        imgproc::rectangle(&mut colored,core::Rect::new(100,200,30,30),color,3,0,0);
        for rline in grid.data.iter(){
            if let RegionLine::LineFX{x:x,y:y,k:k,weight:w}=rline{
                let color = core::Scalar::new(0.0,0.0,255.0,0.0);
                let color1 = core::Scalar::new(255.0,0.0,255.0,0.0);
                let x1 = *x+10.0;
                let y1 = *y - (x-x1)*(*k);
                imgproc::line(&mut colored,core::Point::new(*x as i32, *y as i32),core::Point::new(x1 as i32,y1 as i32),color1,1,8,0);
                imgproc::rectangle(&mut colored,core::Rect::new(*x as i32 -1,*y as i32 -1,3,3),color,1,1,0);
            }
            if let RegionLine::LineFY{x:x,y:y,k:k,weight:w}=rline{
                let color = core::Scalar::new(0.0,255.0,0.0,0.0);
                let color1 = core::Scalar::new(255.0,255.0,0.0,0.0);
                let y1 = *y+10.0;
                let x1 = *x - (y-y1)*(*k);
                imgproc::line(&mut colored,core::Point::new(*x as i32, *y as i32),core::Point::new(x1 as i32,y1 as i32),color1,1,8,0);
                imgproc::rectangle(&mut colored,core::Rect::new(*x as i32 -1,*y as i32 -1,3,3),color,1,1,0);
            }
        }

        let (k, sigma) = grid.average_k();
        let mut d0o:Option<f32> = None;
        let color = core::Scalar::new(255.0,0.0,0.0,0.0);

        for rline in grid.data.iter(){
            if let RegionLine::LineFX{x,y,..}=rline {
                let d = *y-k*(*x);
                if let Some(d0) = d0o{
                    println!("delta {}",d-d0);
                    if (d-d0).abs()<5.0{
                        imgproc::rectangle(&mut colored,core::Rect::new(*x as i32 -3,*y as i32 -3,5,5),color,1,1,0); 
                    }
                }
                else{
                    println!("====================================>     D0 = {} K = {}",d,k);
                    d0o = Some(d);
                }
            }
        }
        

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
