use super::characteristics_grid::*;
use super::utils::*;

const REGION_SIZE: usize = 8;
const REGIONLINE_BUFFER_SIZE: usize = (640 / REGION_SIZE) * (480 / REGION_SIZE);

pub trait RegionLineEvaluator{
    fn evaluate_region(
            &self,
            x1: usize,
            y1: usize,
            dx: usize,
            dy: usize,
            mean_angle: u8,
            delta_angle: u8,
            weight_threshold: f32,
        ) -> RegionLine;
    fn make_regionline_grid(
        &self,
        mean_angle: u8,
        delta_angle: u8,
        weight_threshold: f32,
    ) -> RegionLineGrid;
    fn make_regionline_grid_symmetric(
        &self,
        mean_angle: u8,
        delta_angle: u8,
        weight_threshold: f32,
    ) -> RegionLineGrid;    
}

impl RegionLineEvaluator for CharacteristicsGrid{
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
    pub fn new(cols: usize, rows: usize) -> RegionLineGrid {
        let len = rows * cols;
        assert!(len < REGIONLINE_BUFFER_SIZE);
        RegionLineGrid {
            rows: rows,
            cols: cols,
            data: [RegionLine::Empty; REGIONLINE_BUFFER_SIZE],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> RegionLine {
        self.data[x + self.cols * y]
    }

    pub fn set(&mut self, x: usize, y: usize, value: RegionLine) {
        self.data[x * self.cols + y] = value
    }

    pub fn average_k(&self) -> (f32, f32) {
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

    pub fn neighbor_indices(x:usize,y:usize)->impl Iterator<Item=(usize,usize)>{
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
    pub fn extrapolate_to_neighbors(&self) -> RegionLineGrid{
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

