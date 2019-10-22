extern crate opencv;

use opencv::core;
use opencv::highgui;
use opencv::imgproc;
use opencv::videoio;

use std::fs::File;
use std::io::prelude::*;

pub mod angle_histogram;
pub mod characteristics_grid;
pub mod fixed_histogram;
pub mod grids;
pub mod line;
pub mod quadratic;
pub mod line_grid;
pub mod utils;

use angle_histogram::*;
use characteristics_grid::*;
use fixed_histogram::*;
use grids::*;
use line::*;
use quadratic::*;
use line_grid::*;

fn convolution(
    cols: usize,
    rows: usize,
    source: &[u8],
    destination: &mut [u8],
    characteristics: &mut CharacteristicsGrid,
) {
    let n = 5_usize;

    //    let mut characteristics = CharacteristicsGrid::new(cols - n, rows - n);
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

    for i in 0..(rows - n) {
        for j in 0..(cols - n) {
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
            sum /= 0xC0000;
            if sum > maxintensity {
                maxintensity = sum;
            }
            if sum < 20 {
                sum = 0;
            }

            let a = (128f64 + 128f64 * (s as f64).atan2(c as f64) / pi) as u8;
            characteristics.set(
                j,
                i,
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
            sum *= 2;
            sum = if sum > 255 { 255 } else { sum };
            destination[i * cols + j] = sum as u8;
            if sum > 20 && (a == 0 || a == 255) {
                destination[i * cols + j] = 255;
            }
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
    //    characteristics.make_regionline_grid_symmetric(histogram.main_angle(), 5, 1000f32)
}

fn run() -> opencv::Result<()> {
    let window = "video capture";
    highgui::named_window(window, 1)?;
    println!("Window created");
    //   #[cfg(feature = "opencv-32")]
    let mut cam = videoio::VideoCapture::new(2)?; // 0 is the default camera
    println!("Camera created");
    //   #[cfg(not(feature = "opencv-32"))]
    //   let mut cam = videoio::VideoCapture::new_with_backend(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    let mut counter = 0;
    let mut grids = Grids::new();
    loop {
        counter = (counter + 1) % 50;
        println!("Loop");
        let mut frame = core::Mat::default()?;
        //        let mut frame = unsafe{core::Mat::new_rows_cols(300,200,core::CV_8UC3)}?;
        cam.read(&mut frame)?;

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

        convolution(
            cols as usize,
            rows as usize,
            &clone_gray[..len],
            slice,
            &mut grids.characteristics_grid,
        );
        let a = grids.calculate_main_angle();
        println!("MAIN ANGLE:      {} <-----", a);

        let mut colored = core::Mat::default()?;
        imgproc::cvt_color(&gray, &mut colored, imgproc::COLOR_GRAY2BGR, 0)?;

        grids.line_grid_mut().reset();
        grids.scan_lines(320, 240, false);
        //        grids.scan_lines_grid(false);
        //        grids.reduce_all();
        //        grids.reduce_all();
        //        grids.scan_lines_grid(true);

        let color_mid = core::Scalar::new(0.0, 128.0, 0.0, 0.0);
        let color_line = core::Scalar::new(100.0, 255.0, 100.0, 0.0);
        for line in grids.line_grid().data.iter() {
            if let Some((x1, y1, x2, y2, x, y)) = line.to_edges(600.0, 400.0).points_i32() {
                imgproc::line(
                    &mut colored,
                    core::Point::new(x1, y1),
                    core::Point::new(x2, y2),
                    color_line,
                    1,
                    8,
                    0,
                );
                imgproc::rectangle(
                    &mut colored,
                    core::Rect::new(x - 2, y - 2, 5, 5),
                    color_mid,
                    1,
                    1,
                    0,
                );
            }
        }
        let color_mid = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
        let d= grids.avg_distance();
        for p in grids.intersections() {
            imgproc::rectangle(
                &mut colored,
                core::Rect::new(p.x as i32 - 2, p.y as i32 - 2, 5, 5),
                color_mid,
                1,
                1,
                0,
            );
            imgproc::ellipse(
                &mut colored,
                core::Point::new(p.x as i32, p.y as i32),
                core::Size::new(d as i32,d as i32),
                0.0,
                0.0,
                360.0,
                color_mid,
                1,8,0
            );
        }
        let color_mid = core::Scalar::new(255.0, 0.0, 0.0, 0.0);
        if let Some(parallels) = grids.parallels(){
            for i in 0..20{
                let line = parallels.line(i as f32, 200.0);//.to_edges(640.0, 480.0);
                if let Some((x1, y1, x2, y2, x, y)) = line.points_i32() {
                    imgproc::line(
                        &mut colored,
                        core::Point::new(x1, y1),
                        core::Point::new(x2, y2),
                        color_mid,
                        1,
                        8,
                        0,
                    );
                    imgproc::rectangle(
                        &mut colored,
                        core::Rect::new(x - 3, y - 3, 7, 7),
                        color_mid,
                        2,
                        1,
                        0,
                    );
                }
            }
        }
        let mut f = File::create("test.txt").unwrap();
        let (fx, fy) = grids.fit_index();
        f.write(b"x,y,weight,ix,iy\n");
        for p in grids.intersections() {
            let ix = fx.map(|q| q.f(p.x)).unwrap_or(-1000.0);
            let iy = fy.map(|q| q.f(p.y)).unwrap_or(-1000.0);
            write!(f, "{},{},{},{},{}\n", p.x, p.y, p.weight, ix, iy);
        }

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
