use super::characteristics_grid::*;
use super::line::*;
use super::utils::*;

const REGION_SIZE: usize = 8;
const LINE_BUFFER_SIZE: usize = (640 / REGION_SIZE) * (480 / REGION_SIZE);


pub struct LineGrid {
    pub cols: usize,
    pub rows: usize,
    pub data: [Line; LINE_BUFFER_SIZE],
    pub count: usize
}

impl LineGrid {
    pub fn new(cols: usize, rows: usize) -> LineGrid {
        debug_assert!((rows * cols) <= LINE_BUFFER_SIZE);
        LineGrid {
            rows: rows,
            cols: cols,
            data: [Line::new(); LINE_BUFFER_SIZE],
            count: 0
        }
    }

    pub fn reset(&mut self) {
        for p in self.data.iter_mut() {
            p.reset();
        }
        self.count=0;
    }

    pub fn get(&self, x: usize, y: usize) -> Line {
        debug_assert!(x < self.cols);
        debug_assert!(y < self.rows);
        self.data[x + self.cols * y]
    }

    pub fn set(&mut self, x: usize, y: usize, value: Line) {
        debug_assert!(x < self.cols);
        debug_assert!(y < self.rows);
        self.data[x + self.cols * y] = value;
        //self.count= x + self.cols * y + 1;
    }

    pub fn push(&mut self, value:Line) {
        self.data[self.count]=value;
        self.count+=1;
    }

    pub fn lines(&self) -> impl Iterator<Item=Line>{
        let data = self.data;
        (0..self.count).map(move |i| data[i])
    }

    pub fn nonempty_lines(&self) -> impl Iterator<Item=Line>{
        self.lines().filter(|line| line.line_type!=LineType::Empty)
    }

    pub fn remove_empty(&mut self){
        let lines = self.nonempty_lines();
        self.count = 0;
        for line in lines{
            self.push(line);
        }
    }


    pub fn from_characteristics(&mut self, grid: &CharacteristicsGrid, angle: u8, delta: u8) {
        for i in 0..self.cols {
            for j in 0..self.rows {
                self.set(
                    i,
                    j,
                    Line::fit_region(
                        grid,
                        i * REGION_SIZE,
                        j * REGION_SIZE,
                        REGION_SIZE,
                        REGION_SIZE,
                        angle,
                        delta,
                    ),
                )
            }
        }
    }

    pub fn from_characteristics_c2(&mut self, grid: &CharacteristicsGrid, angle: u8, delta: u8) {
        for i in 0..self.cols {
            for j in 0..self.rows {
                self.set(
                    i,
                    j,
                    Line::fit_region_c2(
                        grid,
                        i * REGION_SIZE,
                        j * REGION_SIZE,
                        REGION_SIZE,
                        REGION_SIZE,
                        angle,
                        delta,
                    ),
                )
            }
        }
    }

    pub fn from_neighbors(&mut self, grid: &LineGrid) {
        let neighbors = 3;
        let mut fit = LinearFit::new();
        self.reset();
        for y in 0..(self.rows - neighbors) {
            for x in 0..(self.cols - neighbors) {
                let central = grid.get(x + 1, y + 1);
                if central.line_type == LineType::Empty {
                    continue;
                }
                fit.reset();
                fit.add_line(central, central.similarity(central,7.0));
                let mut tl: TwoLargest<Line> = TwoLargest::new();

                for j in 0..neighbors {
                    for i in 0..neighbors {
                        if i == 1 && j == 1 {
                            continue;
                        }
                        let line = self.get(x + i, y + j);
                        if line.point.weight > 0.0 {
                            tl.add(central.similarity(line,7.0), line);
                        }
                    }
                }

                for &line in tl.max_item.iter().chain(tl.second_max_item.iter()) {
                    fit.add_line(line, central.similarity(line,7.0));
                }
                self.set(x + 1, y + 1, fit.line())
            }
        }
    }
/*
    pub fn reduce_area_(&mut self, grid: &LineGrid, step:usize) {
        self.reset();
        for y in (0..(self.rows - step)).step_by(step) {
            for x in (0..(self.cols - step)).step_by(step) {
                let reference = grid.get(x, y);
                if reference.line_type == LineType::Empty {
                    continue;
                }
                for j in 0..step {
                    for i in 0..step {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        let line = grid.get(x + i, y + j);
                        let reduced = reference.reduce(line);

                        if reduced.line_type == LineType::Empty{
                            self.set(x + i, y + j, line);
                        }
                        else{
                            self.set(x, y, reduced);
                        }
                    }
                }
            }
        }
    }
*/
    pub fn reduce_area(&mut self, grid: &LineGrid, step:usize) {
        self.reset();
        for y in (0..(self.rows - step)).step_by(step) {
            for x in (0..(self.cols - step)).step_by(step) {
                for j1 in 0..step {
                    for i1 in 0..step {
                        let index1 = i1+j1*step;
                        let mut line1 = grid.get(x + i1, y + j1);
                        if line1.line_type == LineType::Empty{
                            continue;
                        }
                        if line1.length()<8.0{
                            continue;
                        }
                        for j2 in 0..step {
                            for i2 in 0..step {
                                let index2 = i2+j2*step;
                                if index2<index1{
                                    continue;
                                }
                                let line2 = grid.get(x + i2, y + j2);
                                let reduced = line1.reduce(line2,7.0);

                                if reduced.line_type != LineType::Empty{
                                    line1=reduced;
                                }
                            }
                        }
                        self.push(line1);
                    }
                }
            }
        }
    }

    pub fn reduce_all(&mut self, grid: &LineGrid) {
        self.reset();
        for i in (0..grid.count){
            let mut line1 = grid.data[i];
            if line1.line_type == LineType::Empty{
                continue;
            }
            if line1.length()<10.0{
                continue;
            }
            for j in (0..i+1){
                let line2 = grid.data[j];
                let reduced = line1.reduce(line2,12.0);

                if reduced.line_type != LineType::Empty{
                    line1=reduced;
                }
            }
            self.push(line1);
        }
    }

}
