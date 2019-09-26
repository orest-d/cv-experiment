use super::characteristics_grid::*;
use super::utils::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LineType {
    FX,
    FY,
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedPoint {
    pub x: f32,
    pub y: f32,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearFit {
    x_sum: f32,
    y_sum: f32,
    xx_sum: f32,
    yy_sum: f32,
    xy_sum: f32,
    w_sum: f32,
    xmin: Option<f32>,
    xmax: Option<f32>,
    ymin: Option<f32>,
    ymax: Option<f32>,
    count: usize,
}

impl LinearFit {
    pub fn new() -> LinearFit {
        LinearFit {
            x_sum: 0.0,
            y_sum: 0.0,
            xx_sum: 0.0,
            yy_sum: 0.0,
            xy_sum: 0.0,
            w_sum: 0.0,
            xmin: None,
            xmax: None,
            ymin: None,
            ymax: None,
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.x_sum = 0.0;
        self.y_sum = 0.0;
        self.xx_sum = 0.0;
        self.yy_sum = 0.0;
        self.xy_sum = 0.0;
        self.w_sum = 0.0;
        self.xmin = None;
        self.xmax = None;
        self.ymin = None;
        self.ymax = None;
        self.count = 0;
    }

    pub fn add(&mut self, x: f32, y: f32, w: f32) {
        self.x_sum += w * x;
        self.y_sum += w * y;
        self.xx_sum += w * x * x;
        self.yy_sum += w * y * y;
        self.xy_sum += w * x * y;
        self.w_sum += w;
        self.count += 1;
        if w > 0.0 {
            self.xmin = Some(self.xmin.map_or(x, |old| old.min(x)));
            self.xmax = Some(self.xmax.map_or(x, |old| old.max(x)));
            self.ymin = Some(self.ymin.map_or(y, |old| old.min(y)));
            self.ymax = Some(self.ymax.map_or(y, |old| old.max(y)));
        }
    }

    pub fn add_wp(&mut self, wp: WeightedPoint) {
        self.add(wp.x, wp.y, wp.weight)
    }

    pub fn add_line(&mut self, line: Line, weight:f32) {
        if let Some((x1,y1,x2,y2,x,y)) = line.points(){
            self.add(x,y,weight);
            self.add(x1,y1,weight/2.0);
            self.add(x2,y2,weight/2.0);
        }
    }

    pub fn line(&self) -> Line {
        let mut line = Line::new();
        if self.w_sum > 0.0 {
            let x = self.x_sum / self.w_sum;
            let y = self.y_sum / self.w_sum;
            let xx = self.xx_sum / self.w_sum - x * x;
            let yy = self.yy_sum / self.w_sum - y * y;
            let xy = self.xy_sum / self.w_sum - x * y;
            if xx.abs() > 0.0 || yy.abs() > 0.0 {
                line.point.weight = self.w_sum / self.count as f32;
                if xx.abs() >= yy.abs() {
                    line.line_type = LineType::FX;
                    line.point.x = x;
                    line.point.y = y;
                    line.k = xy / xx;
                    line.x1 = self.xmin.unwrap_or(x);
                    line.x2 = self.xmax.unwrap_or(x);
                } else {
                    line.line_type = LineType::FY;
                    line.point.x = y;
                    line.point.y = x;
                    line.k = xy / yy;
                    line.x1 = self.ymin.unwrap_or(y);
                    line.x2 = self.ymax.unwrap_or(y);
                }
            }
        }
        line
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub line_type: LineType,
    pub point: WeightedPoint,
    pub k: f32,
    pub x1: f32,
    pub x2: f32,
}

impl Line {
    pub fn new() -> Line {
        Line {
            line_type: LineType::Empty,
            point: WeightedPoint {
                x: 0.0,
                y: 0.0,
                weight: 0.0,
            },
            k: 0.0,
            x1: 0.0,
            x2: 0.0,
        }
    }

    pub fn reset(&mut self){
        self.line_type = LineType::Empty;
        self.point = WeightedPoint{x:0.0, y:0.0, weight:0.0};
        self.k = 0.0;
        self.x1 = 0.0;
        self.x2 = 0.0;
    }

    pub fn fit_weighted_points<'a, I>(points: I) -> Self
    where
        I: IntoIterator<Item = &'a WeightedPoint>,
    {
        let mut fit = LinearFit::new();
        for &wp in points {
            fit.add_wp(wp);
        }
        fit.line()
    }

    pub fn fit_region(
        grid: &CharacteristicsGrid,
        x: usize,
        y: usize,
        dx: usize,
        dy: usize,
        angle: u8,
        delta: u8,
    ) -> Line {
        let mut fit = LinearFit::new();

        for (x, y) in region_indices(x, y, dx, dy) {
            let c = grid.get(x, y);
            if angle_difference(c.angle, angle) < delta {
                fit.add(x as f32, y as f32, c.intensity as f32);
            }
        }
        fit.line()
    }

    pub fn fit_region_c2(
        grid: &CharacteristicsGrid,
        x: usize,
        y: usize,
        dx: usize,
        dy: usize,
        angle: u8,
        delta: u8,
    ) -> Line {
        let mut fit = LinearFit::new();

        for (x, y) in region_indices(x, y, dx, dy) {
            let c = grid.get(x, y);
            if angle_difference_c2(c.angle, angle) < delta {
                fit.add(x as f32, y as f32, c.intensity as f32);
            }
        }
        fit.line()
    }

    pub fn y1(&self) -> f32 {
        self.point.y + self.k * (self.x1 - self.point.x)
    }

    pub fn y2(&self) -> f32 {
        self.point.y + self.k * (self.x2 - self.point.x)
    }

    pub fn midpoint(&self) -> Option<(f32, f32)> {
        match self.line_type {
            LineType::Empty => None,
            LineType::FX => Some((self.point.x, self.point.y)),
            LineType::FY => Some((self.point.y, self.point.x)),
        }
    }

    pub fn points_i32(&self) -> Option<(i32, i32, i32, i32, i32, i32)>{
        match self.line_type {
            LineType::Empty => None,
            LineType::FX => Some((
                self.x1 as i32, self.y1() as i32,
                self.x2 as i32, self.y2() as i32,
                self.point.x as i32, self.point.y as i32)),
            LineType::FY => Some((
                self.y1() as i32, self.x1 as i32, 
                self.y2() as i32, self.x2 as i32, 
                self.point.y as i32, self.point.x as i32))
        }        
    }

    pub fn points(&self) -> Option<(f32, f32, f32, f32, f32, f32)>{
        match self.line_type {
            LineType::Empty => None,
            LineType::FX => Some((
                self.x1, self.y1(),
                self.x2, self.y2(),
                self.point.x, self.point.y)),
            LineType::FY => Some((
                self.y1(), self.x1, 
                self.y2(), self.x2, 
                self.point.y, self.point.x))
        }        
    }

    pub fn direction(&self) -> (f32, f32) {
        let normal = (1.0 + self.k * self.k).sqrt();
        match self.line_type {
            LineType::Empty => (0.0, 0.0),
            LineType::FX => (1.0 / normal, self.k / normal),
            LineType::FY => (self.k / normal, 1.0 / normal),
        }
    }

    pub fn raw_normal(&self) -> (f32, f32) {
        let normal = (1.0 + self.k * self.k).sqrt();
        (-self.k / normal, 1.0 / normal)
    }

    pub fn normal(&self) -> (f32, f32) {
        let (x, y) = self.direction();
        (-y, x)
    }

    pub fn overlap(&self, line: Line) -> f32 {
        let (x1, y1) = self.direction();
        let (x2, y2) = line.direction();
        (x1 * x2 + y1 * y2).abs()
    }

    fn raw_distance(&self, x: f32, y: f32) -> f32 {
        let (nx, ny) = self.raw_normal();
        let (rx, ry) = (x - self.point.x, y - self.point.y);
        (rx * nx + ry * ny).abs()
    }

    pub fn distance(&self, x: f32, y: f32) -> f32 {
        match self.line_type {
            LineType::Empty => 1.0e5,
            LineType::FX => self.raw_distance(x, y),
            LineType::FY => self.raw_distance(y, x),
        }
    }

    pub fn max_endpoint_distance(&self, line:Line)->Option<f32>{
        if let Some((x1,y1,x2,y2,_,_))=line.points(){
            Some(self.distance(x1,y1).max(self.distance(x2,y2)))
        }
        else{
            None
        }
    }

    pub fn as_fx(&self) -> Line {
        match self.line_type {
            LineType::FY => {
                let mut line = Line::new();
                if self.k.abs() < 0.25 {
                    line
                } else {
                    line.line_type = LineType::FX;
                    line.point.x = self.point.y;
                    line.point.y = self.point.x;
                    line.point.weight = self.point.weight;
                    line.k = 1.0 / self.k;
                    line.x1 = self.y1();
                    line.x2 = self.y2();
                    line
                }
            }
            _ => *self,
        }
    }

    pub fn as_fy(&self) -> Line {
        match self.line_type {
            LineType::FX => {
                let mut line = Line::new();
                if self.k.abs() < 0.25 {
                    line
                } else {
                    line.line_type = LineType::FY;
                    line.point.x = self.point.y;
                    line.point.y = self.point.x;
                    line.point.weight = self.point.weight;
                    line.k = 1.0 / self.k;
                    line.x1 = self.y1();
                    line.x2 = self.y2();
                    line
                }
            }
            _ => *self,
        }
    }

    pub fn length(&self) ->f32{
        let dx = self.x2 - self.x1;
        let dy = self.y2() - self.y1();
        (dx*dx+dy*dy).sqrt()
    }

    pub fn similarity(&self, line:Line,minlength:f32) -> f32{
        fn clamp(x:f32)->f32{
            if x<0.0{
                0.0
            }
            else{
                x
            }
        }
        if self.line_type == LineType::Empty{
            0.0
        }
        else{
            if let Some(d) = self.max_endpoint_distance(line){
                let w = (self.point.weight*line.point.weight).sqrt();
                let l1 = clamp(self.length()-minlength);
                let l2 = clamp(line.length()-minlength);
                let l = (l1*l2).sqrt();
                let dw = (-d*d/20.0).exp();
                let dwc = if d<=4.0 {dw} else {0.0};
                let oc = clamp(0.1-(1.0-self.overlap(line))*(1.0+d)*(1.0+d));
                w*l*oc*dwc
            }
            else{
                0.0
            }
        }
    }

    pub fn reduce(&self, line:Line,minlength:f32) -> Line{
        if self.similarity(line,minlength)>0.0{
            let mut fit = LinearFit::new();
            fit.add_line(*self, self.point.weight);
            fit.add_line(line, line.point.weight);
            fit.line()
        }
        else{
            Line::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let line = Line::new();
        assert_eq!(line.line_type, LineType::Empty);
    }

    #[test]
    fn test_line_fit_simple() {
        let line = Line::fit_weighted_points(&[
            WeightedPoint {
                x: 0.0,
                y: 0.0,
                weight: 1.0,
            },
            WeightedPoint {
                x: 1.0,
                y: 1.0,
                weight: 1.0,
            },
        ]);
        assert_eq!(line.line_type, LineType::FX);
        assert_eq!(line.point.x, 0.5);
        assert_eq!(line.point.y, 0.5);
        assert_eq!(line.point.weight, 1.0);
        assert_eq!(line.k, 1.0);
        assert_eq!(line.x1, 0.0);
        assert_eq!(line.x2, 1.0);
    }

    #[test]
    fn test_fit_simple() {
        let mut fit = LinearFit::new();
        fit.add(0.0, 0.0, 1.0);
        fit.add(1.0, 1.0, 1.0);

        let line = fit.line();
        assert_eq!(line.line_type, LineType::FX);
        assert_eq!(line.point.x, 0.5);
        assert_eq!(line.point.y, 0.5);
        assert_eq!(line.point.weight, 1.0);
        assert_eq!(line.k, 1.0);
        assert_eq!(line.x1, 0.0);
        assert_eq!(line.x2, 1.0);
    }

    #[test]
    fn test_fit_add_line1() {
        let mut fit = LinearFit::new();
        fit.add(0.0, 0.0, 1.0);
        fit.add(1.0, 1.0, 1.0);

        let line1 = fit.line();
        let mut fit2 = LinearFit::new();
        fit2.add_line(line1, 1.0);
        let line2 = fit.line();

        assert_eq!(line2.line_type, LineType::FX);
        assert_eq!(line2.point.x, 0.5);
        assert_eq!(line2.point.y, 0.5);
        assert_eq!(line2.point.weight, 1.0);
        assert_eq!(line2.k, 1.0);
        assert_eq!(line2.x1, 0.0);
        assert_eq!(line2.x2, 1.0);
    }

    #[test]
    fn test_distance1() {
        let mut fit = LinearFit::new();
        fit.add(0.0, 0.0, 1.0);
        fit.add(1.0, 0.0, 1.0);
        let line = fit.line();
        assert_eq!(line.raw_distance(2.0, 2.0), 2.0);
    }

    #[test]
    fn test_distance2() {
        let mut fit = LinearFit::new();
        fit.add(1.0, 0.0, 1.0);
        fit.add(0.0, 1.0, 1.0);
        let line = fit.line();
        assert_eq!(line.raw_distance(0.0, 0.0), (2.0f32).sqrt() / 2.0);
    }

    #[test]
    fn test_overlap_identical() {
        let mut fit = LinearFit::new();
        fit.add(1.0, 0.0, 1.0);
        fit.add(0.0, 1.0, 1.0);
        let line = fit.line();
        assert!((line.overlap(line) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_overlap_perpendicular() {
        let mut fit = LinearFit::new();
        fit.add(0.0, 0.0, 1.0);
        fit.add(1.0, 0.0, 1.0);
        let line1 = fit.line();
        fit.reset();
        fit.add(0.0, 0.0, 1.0);
        fit.add(0.0, 1.0, 1.0);
        let line2 = fit.line();
        assert!((line1.overlap(line2)).abs() < 1e-5);
    }

    #[test]
    fn test_fit_fy() {
        let line = Line::fit_weighted_points(&[
            WeightedPoint {
                x: 0.0,
                y: 0.0,
                weight: 1.0,
            },
            WeightedPoint {
                x: 1.0,
                y: 2.0,
                weight: 1.0,
            },
        ]);
        assert_eq!(line.line_type, LineType::FY);
        assert_eq!(line.point.x, 1.0);
        assert_eq!(line.point.y, 0.5);
        assert_eq!(line.point.weight, 1.0);
        assert_eq!(line.k, 0.5);
        assert_eq!(line.x1, 0.0);
        assert_eq!(line.x2, 2.0);
    }

    #[test]
    fn test_fit_fy_as_fx() {
        let line = Line::fit_weighted_points(&[
            WeightedPoint {
                x: 0.0,
                y: 0.0,
                weight: 1.0,
            },
            WeightedPoint {
                x: 1.0,
                y: 2.0,
                weight: 1.0,
            },
        ]);
        assert_eq!(line.line_type, LineType::FY);
        assert_eq!(line.k, 0.5);
        assert_eq!(line.x1, 0.0);
        assert_eq!(line.x2, 2.0);

        let line = line.as_fx();
        assert_eq!(line.line_type, LineType::FX);
        assert_eq!(line.point.x, 0.5);
        assert_eq!(line.point.y, 1.0);
        assert_eq!(line.point.weight, 1.0);
        assert_eq!(line.k, 2.0);
        assert_eq!(line.x1, 0.0);
        assert_eq!(line.x2, 1.0);
    }

    #[test]
    fn test_fit_fx_weighted() {
        let line = Line::fit_weighted_points(&[
            WeightedPoint {
                x: 1.0,
                y: 2.0,
                weight: 1.0,
            },
            WeightedPoint {
                x: 4.0,
                y: 5.0,
                weight: 2.0,
            },
        ]);
        assert_eq!(line.line_type, LineType::FX);
        assert_eq!(line.point.x, 3.0);
        assert_eq!(line.point.y, 4.0);
        assert_eq!(line.point.weight, 1.5);
        assert_eq!(line.k, 1.0);
        assert_eq!(line.x1, 1.0);
        assert_eq!(line.x2, 4.0);
    }

}
