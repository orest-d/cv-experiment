use super::angle_histogram::*;
use super::characteristics_grid::*;
use super::line::*;
use super::line_grid::*;
use super::utils::*;

const ANGLE_DELTA: u8 = 10;

pub struct Grids {
    pub characteristics_grid: CharacteristicsGrid,
    pub angle_histogram: AngleHistogram,
    pub line_grid1: LineGrid,
    pub line_grid2: LineGrid,
    pub line_grid1_is_active: bool,
    pub main_angle: u8,
}

impl Grids {
    pub fn new() -> Grids {
        Grids {
            characteristics_grid: CharacteristicsGrid::new(640 - 5, 480 - 5),
            angle_histogram: AngleHistogram::new(),
            line_grid1: LineGrid::new(640 / 8 - 1, 480 / 8 - 1),
            line_grid2: LineGrid::new(640 / 8 - 1, 480 / 8 - 1),
            line_grid1_is_active: true,
            main_angle: 0,
        }
    }

    pub fn reset(&mut self) {
        self.characteristics_grid.reset();
        self.angle_histogram.reset();
        self.line_grid1.reset();
        self.line_grid2.reset();
        self.line_grid1_is_active = true;
        self.main_angle = 0;
    }

    pub fn line_grid(&self) -> &LineGrid {
        if (self.line_grid1_is_active) {
            &self.line_grid1
        } else {
            &self.line_grid2
        }
    }

    pub fn line_grid_mut(&mut self) -> &mut LineGrid {
        if (self.line_grid1_is_active) {
            &mut self.line_grid1
        } else {
            &mut self.line_grid2
        }
    }

    pub fn switch_linear_grid(&mut self) {
        self.line_grid1_is_active = !self.line_grid1_is_active;
    }

    pub fn calculate_main_angle(&mut self) -> u8 {
        self.angle_histogram.reset();
        for row in 0..self.characteristics_grid.rows {
            for p in self.characteristics_grid.data[(row * self.characteristics_grid.cols)
                ..((row + 1) * self.characteristics_grid.cols)]
                .iter()
            {
                self.angle_histogram.add(p.angle, p.intensity as i32);
            }
        }
        self.main_angle = self.angle_histogram.main_angle();
        self.main_angle
    }

    pub fn fit(&mut self, angle: u8, delta: u8) {
        let lg = if (self.line_grid1_is_active) {
            &mut self.line_grid1
        } else {
            &mut self.line_grid2
        };
        lg.from_characteristics(&self.characteristics_grid, angle, delta);
    }

    pub fn fit_c2(&mut self, angle: u8, delta: u8) {
        let lg = if (self.line_grid1_is_active) {
            &mut self.line_grid1
        } else {
            &mut self.line_grid2
        };
        lg.from_characteristics_c2(&self.characteristics_grid, angle, delta);
    }

    pub fn fit_vertical(&mut self) {
        self.fit_c2(self.main_angle, ANGLE_DELTA);
    }

    pub fn fit_horizontal(&mut self) {
        self.fit_c2(self.main_angle.wrapping_add(64), ANGLE_DELTA);
    }

    pub fn lines_from_neighbors(&mut self) {
        self.switch_linear_grid();
        let (target, source) = if self.line_grid1_is_active {
            (&mut self.line_grid1, &self.line_grid2)
        } else {
            (&mut self.line_grid2, &self.line_grid1)
        };
        target.from_neighbors(source);
    }

    pub fn reduce_area(&mut self, step: usize) {
        self.switch_linear_grid();
        let (target, source) = if self.line_grid1_is_active {
            (&mut self.line_grid1, &self.line_grid2)
        } else {
            (&mut self.line_grid2, &self.line_grid1)
        };
        target.reduce_area(source, step);
    }

    pub fn reduce_all(&mut self) {
        self.switch_linear_grid();
        let (target, source) = if self.line_grid1_is_active {
            (&mut self.line_grid1, &self.line_grid2)
        } else {
            (&mut self.line_grid2, &self.line_grid1)
        };
        target.reduce_all(source);
    }

    pub fn sample_line(&self, line:Line, angle:u8, delta:u8)->f32{
        let g = &self.characteristics_grid;
        let mut w = 0.0f32;
        for (x,y) in line.grid_coordinates(g.cols, g.rows){
            let c = g.get(x,y);
            if angle_difference(angle,c.angle)<=delta{
                w+=c.intensity as f32;
            }
        }
        w
    }
    pub fn sample_line_c2(&self, line:Line, angle:u8, delta:u8)->f32{
        let g = &self.characteristics_grid;
        let mut w = 0.0f32;
        for (x,y) in line.grid_coordinates(g.cols, g.rows){
            let c = g.get(x,y);
            if angle_difference_c2(angle,c.angle)<=delta{
                w+=c.intensity as f32;
            }
        }
        w
    }

    pub fn find_line(&self, x:usize, y:usize, angle:u8, delta:u8, distance:f32, length:f32, c2:bool) -> Line{
        let line = Line::new_from_angle(angle+64, x as f32, y as f32, length);
        let mut largest:Largest<Line> = Largest::new();

        for sl in line.sample_parallel_lines((distance*2.0) as usize, distance){
            let w = if c2 {self.sample_line_c2(sl, angle, delta)} else {self.sample_line(sl, angle, delta)};
            let mut ll = sl;
            ll.point.weight = w;
            largest.add(w,ll);
        }
        largest.max_item.unwrap_or(Line::new())
    }
}
