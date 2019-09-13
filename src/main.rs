extern crate opencv;

use opencv::core;
use opencv::highgui;
use opencv::imgproc;
use opencv::videoio;

pub mod angle_histogram;
pub mod fixed_histogram;
pub mod characteristics_grid;
pub mod region_line;

use angle_histogram::*;
use fixed_histogram::*;
use characteristics_grid::*;
use region_line::*;


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
