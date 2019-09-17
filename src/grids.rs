use super::angle_histogram::*;
use super::utils::*;
use super::characteristics_grid::*;
use super::line::*;
use super::line_grid::*;

const ANGLE_DELTA: u8 = 10;

pub struct Grids{
    pub characteristics_grid:CharacteristicsGrid,
    pub angle_histogram:AngleHistogram,
    pub line_grid1:LineGrid,
    pub line_grid2:LineGrid,
    pub line_grid1_is_active:bool,
    pub main_angle:u8
}

impl Grids{
    pub fn new()->Grids{
        Grids{
            characteristics_grid: CharacteristicsGrid::new(640 - 5, 480 - 5),
            angle_histogram: AngleHistogram::new(),
            line_grid1: LineGrid::new(640 / 8 - 1, 480 / 8 - 1),
            line_grid2: LineGrid::new(640 / 8 - 1, 480 / 8 - 1),
            line_grid1_is_active:true,
            main_angle: 0
        }
    }

    pub fn reset(&mut self){
        self.characteristics_grid.reset();
        self.angle_histogram.reset();
        self.line_grid1.reset();
        self.line_grid2.reset();
        self.line_grid1_is_active=true;
        self.main_angle = 0;
    }

    pub fn line_grid(&mut self) -> &mut LineGrid{
        if (self.line_grid1_is_active){&mut self.line_grid1} else {&mut self.line_grid2}
    }

    pub fn switch_linear_grid(&mut self){
        self.line_grid1_is_active = !self.line_grid1_is_active;
    }

    pub fn calculate_main_angle(&mut self) -> u8{
        self.angle_histogram.reset();
        for row in 0..self.characteristics_grid.rows{
            for p in self.characteristics_grid.data[(row*self.characteristics_grid.cols)..((row+1)*self.characteristics_grid.cols)].iter(){
                self.angle_histogram.add(p.angle, p.intensity as i32);
            }
        }
        self.main_angle = self.angle_histogram.main_angle();
        self.main_angle
    }

    pub fn fit(&mut self, angle:u8, delta:u8){
        let lg = if (self.line_grid1_is_active){&mut self.line_grid1} else {&mut self.line_grid2};
        lg.from_characteristics(&self.characteristics_grid, angle, delta);
    }

    pub fn fit_c2(&mut self, angle:u8, delta:u8){
        let lg = if (self.line_grid1_is_active){&mut self.line_grid1} else {&mut self.line_grid2};
        lg.from_characteristics_c2(&self.characteristics_grid, angle, delta);
    }

    pub fn fit_vertical(&mut self){
        self.fit_c2(self.main_angle, ANGLE_DELTA);
    }

    pub fn fit_horizontal(&mut self){
        self.fit_c2(self.main_angle.wrapping_add(64), ANGLE_DELTA);
    }
}