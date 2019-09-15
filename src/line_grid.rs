use super::characteristics_grid::*;
use super::utils::*;
use super::line::*;

const REGION_SIZE: usize = 8;
const LINE_BUFFER_SIZE: usize = (640 / REGION_SIZE) * (480 / REGION_SIZE);

pub struct LineGrid {
    pub cols: usize,
    pub rows: usize,
    pub data: [Line; LINE_BUFFER_SIZE],
}

impl LineGrid {
    pub fn new(cols: usize, rows: usize) -> LineGrid {
        debug_assert!((rows * cols) <= LINE_BUFFER_SIZE);
        LineGrid {
            rows: rows,
            cols: cols,
            data: [Line::new(); LINE_BUFFER_SIZE],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> Line {
        debug_assert!(x<self.cols);
        debug_assert!(y<self.rows);
        self.data[x + self.cols * y]
    }

    pub fn set(&mut self, x: usize, y: usize, value: Line) {
        debug_assert!(x<self.cols);
        debug_assert!(y<self.rows);
        self.data[x + self.cols * y] = value
    }

    pub fn from_characteristics(&mut self, grid:&CharacteristicsGrid, angle:u8, delta:u8){
        for i in 0..self.cols{
            for j in 0..self.rows{
                self.set(i,j,
                Line::fit_region(grid, i*REGION_SIZE, j*REGION_SIZE, REGION_SIZE, REGION_SIZE, angle, delta)
                )
            }
        }
    } 
}

