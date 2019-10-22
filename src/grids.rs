use super::angle_histogram::*;
use super::characteristics_grid::*;
use super::line::*;
use super::line_grid::*;
use super::quadratic::*;
use super::utils::*;

const ANGLE_DELTA: u8 = 10;
const LINECOUNT: usize = 128;
const LINECOUNT_HALF: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct Parallels {
    pub parallel_axis: Line,
    pub orthogonal_axis: Line,
    pub x: QuadraticFunction,
    pub y: QuadraticFunction,
    pub k: QuadraticFunction,
}

impl Parallels {
    pub fn new() -> Parallels {
        Parallels {
            parallel_axis: Line::new(),
            orthogonal_axis: Line::new(),
            x: QuadraticFunction::new(0.0, 1.0, 0.0),
            y: QuadraticFunction::new(0.0, 1.0, 0.0),
            k: QuadraticFunction::new(0.0, 1.0, 0.0),
        }
    }
    pub fn line(&self, i: f32, length: f32) -> Line {
        let x = self.x.f(i);
        let y = self.y.f(i);
        let k = self.k.f(i);
        let mut line = self.orthogonal_axis.orthogonal_line_through(x, y, length);
        line.k = k;
        line
    }

    pub fn point(&self, i: f32) -> (f32, f32) {
        let x = self.x.f(i);
        let y = self.y.f(i);
        (x, y)
    }
}

pub struct Grids {
    pub characteristics_grid: CharacteristicsGrid,
    pub angle_histogram: AngleHistogram,
    pub line_grid1: LineGrid,
    pub line_grid2: LineGrid,
    pub line_grid1_is_active: bool,
    pub main_angle: u8,
    pub angle_delta: u8,
    pub c2: bool,
    pub fit_distance: usize,
    pub fit_iterations: u32,
    pub fit_extension_factor: f32,
    pub scan_line_length: f32,
    pub grid_line_scan_step: f32,
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
            angle_delta: ANGLE_DELTA,
            c2: true,
            fit_distance: 3,
            fit_iterations: 4,
            fit_extension_factor: 3.0,
            scan_line_length: 128.0,
            grid_line_scan_step: 8.0,
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

    pub fn sample_line(&self, line: Line, angle: u8, delta: u8) -> f32 {
        let g = &self.characteristics_grid;
        let mut w = 0.0f32;
        for (x, y) in line.grid_coordinates(g.cols, g.rows) {
            let c = g.get(x, y);
            if angle_difference(angle, c.angle) <= delta {
                w += c.intensity as f32;
            }
        }
        w
    }
    pub fn sample_line_c2(&self, line: Line, angle: u8, delta: u8) -> f32 {
        let g = &self.characteristics_grid;
        let mut w = 0.0f32;
        for (x, y) in line.grid_coordinates(g.cols, g.rows) {
            let c = g.get(x, y);
            if angle_difference_c2(angle, c.angle) <= delta {
                w += c.intensity as f32;
            }
        }
        w
    }

    pub fn find_line(
        &self,
        x: usize,
        y: usize,
        angle: u8,
        delta: u8,
        distance: f32,
        length: f32,
        c2: bool,
    ) -> Line {
        let line = Line::new_from_angle(angle, x as f32, y as f32, length);
        let mut largest: Largest<Line> = Largest::new();

        for sl in line.sample_parallel_lines((distance * 2.0) as usize, distance, 2) {
            let w = if c2 {
                self.sample_line_c2(sl, angle, delta)
            } else {
                self.sample_line(sl, angle, delta)
            };
            let mut ll = sl;
            ll.point.weight = w;
            largest.add(w, ll);
        }
        largest.max_item.unwrap_or(Line::new())
    }

    pub fn fit_along(&self, line: Line, angle: u8, delta: u8, distance: usize) -> Line {
        let g = &self.characteristics_grid;
        let mut fit = LinearFit::new();
        for (x, y) in line.points_around(g.cols, g.rows, distance) {
            let c = g.get(x, y);
            if angle_difference(c.angle, angle) < delta {
                fit.add(x as f32, y as f32, c.intensity as f32);
            }
        }
        fit.line()
    }

    pub fn fit_along_c2(&self, line: Line, angle: u8, delta: u8, distance: usize) -> Line {
        let g = &self.characteristics_grid;
        let mut fit = LinearFit::new();
        for (x, y) in line.points_around(g.cols, g.rows, distance) {
            let c = g.get(x, y);
            if angle_difference_c2(c.angle, angle) < delta {
                fit.add(x as f32, y as f32, c.intensity as f32);
            }
        }
        fit.line()
    }

    pub fn find_line_iterative(
        &self,
        x: usize,
        y: usize,
        angle: u8,
        delta: u8,
        distance: f32,
        length: f32,
        c2: bool,
        fit_distance: usize,
        iterations: u32,
        factor: f32,
    ) -> Line {
        let mut line = self.find_line(x, y, angle, delta, distance, length, c2);
        line = if c2 {
            self.fit_along_c2(line, angle, delta, fit_distance)
        } else {
            self.fit_along(line, angle, delta, fit_distance)
        };
        let xmax = self.characteristics_grid.cols as f32;
        let ymax = self.characteristics_grid.rows as f32;
        for i in 0..iterations {
            line = line.expand(factor).clamp_ends(xmax, ymax);
            line = if c2 {
                self.fit_along_c2(line, angle, delta, fit_distance)
            } else {
                self.fit_along(line, angle, delta, fit_distance)
            };
        }
        line.clamp_ends(xmax, ymax)
    }

    pub fn refine_line_iterative(
        &self,
        line: Line,
        angle: u8,
        delta: u8,
        c2: bool,
        fit_distance: usize,
        iterations: u32,
        factor: f32,
    ) -> Line {
        let mut line = line;
        line = if c2 {
            self.fit_along_c2(line, angle, delta, fit_distance)
        } else {
            self.fit_along(line, angle, delta, fit_distance)
        };
        let xmax = self.characteristics_grid.cols as f32;
        let ymax = self.characteristics_grid.rows as f32;
        for i in 0..iterations {
            line = line.expand(factor).clamp_ends(xmax, ymax);
            line = if c2 {
                self.fit_along_c2(line, angle, delta, fit_distance)
            } else {
                self.fit_along(line, angle, delta, fit_distance)
            };
        }
        line.clamp_ends(xmax, ymax)
    }

    pub fn scan_lines(&mut self, x: usize, y: usize, orthogonal: bool) {
        let angle = if orthogonal {
            self.main_angle + 64
        } else {
            self.main_angle
        };
        let delta = self.angle_delta;
        let scan_line_length = self.scan_line_length;
        let grid_line_scan_step = self.grid_line_scan_step;
        let xmax = self.characteristics_grid.cols;
        let ymax = self.characteristics_grid.rows;
        let diagonal = ((xmax * xmax + ymax * ymax) as f32).sqrt();
        let c2 = self.c2;
        let fit_distance = self.fit_distance;
        let fit_iterations = self.fit_iterations;
        let fit_extension_factor = self.fit_extension_factor;
        let next_line_offset = (grid_line_scan_step + fit_distance as f32).abs();

        let xx = x as f32;
        let yy = y as f32;

        let axis = self.find_line_iterative(
            x,
            y,
            angle,
            delta,
            grid_line_scan_step * 2.0,
            scan_line_length,
            c2,
            fit_distance,
            fit_iterations,
            fit_extension_factor,
        );
        let orthogonal = axis.orthogonal_line_through(xx, yy, 100.0);
        self.line_grid_mut().push(axis);
        let mut line = axis;
        /*
                line = line.parallel_line(next_line_offset).with_length(scan_line_length);
                self.line_grid_mut().push(line);
                line = self.refine_line_iterative(line, angle, delta, c2, fit_distance, fit_iterations, fit_extension_factor);
                self.line_grid_mut().push(line);
        */

        let mut i = -(diagonal / next_line_offset).round();
        let (adx, ady) = axis.direction();
        loop {
            line = axis
                .parallel_line(next_line_offset * i)
                .with_length(scan_line_length);
            line = self.refine_line_iterative(
                line,
                angle,
                delta,
                c2,
                fit_distance,
                fit_iterations,
                fit_extension_factor,
            );
            let (dx, dy) = line.direction();
            let overlap = (adx * dx + ady * dy).abs();
            if overlap > 0.99 {
                self.line_grid_mut().push(line);
            }
            i += 1.0;
            if i >= 0.0 {
                break;
            }
        }
        i = 0.0;
        loop {
            line = axis
                .parallel_line(next_line_offset * i)
                .with_length(scan_line_length);
            if let Some((x, y)) = line.midpoint() {
                if x < 0.0 || y < 0.0 || x >= xmax as f32 || y >= ymax as f32 {
                    break;
                }
            } else {
                break;
            }
            line = self.refine_line_iterative(
                line,
                angle,
                delta,
                c2,
                fit_distance,
                fit_iterations,
                fit_extension_factor,
            );
            let (dx, dy) = line.direction();
            let overlap = (adx * dx + ady * dy).abs();
            if overlap > 0.99 {
                self.line_grid_mut().push(line);
            }
            i += 1.0;
        }
    }

    pub fn scan_lines_grid(&mut self, orthogonal: bool) {
        let angle = if orthogonal {
            self.main_angle + 64
        } else {
            self.main_angle
        };
        let delta = self.angle_delta;
        let xmax = self.characteristics_grid.cols;
        let ymax = self.characteristics_grid.rows;
        let diagonal = ((xmax * xmax + ymax * ymax) as f32).sqrt();

        let axis = Line::new_from_angle(angle, xmax as f32 / 2.0, ymax as f32 / 2.0, diagonal);

        for (x, y) in axis.line2().sample_coordinates(2) {
            self.scan_lines(x as usize, y as usize, orthogonal);
        }
        for (x, y) in axis.line1().sample_coordinates(2) {
            self.scan_lines(x as usize, y as usize, orthogonal);
        }
    }

    pub fn intersections(&mut self) -> impl Iterator<Item = WeightedPoint> {
        let xmax = self.characteristics_grid.cols;
        let ymax = self.characteristics_grid.rows;
        self.line_grid_mut().remove_empty();
        let axis = self.line_grid().data[0];
        let orthogonal_axis =
            axis.orthogonal_line_through(xmax as f32 / 2.0, ymax as f32 / 2.0, 100.0);
        self.line_grid().lines().filter_map(move |line| {
            orthogonal_axis
                .intersection(line)
                .map(move |(x, y)| WeightedPoint {
                    x: x,
                    y: y,
                    weight: line.point.weight,
                })
        })
    }

    pub fn avg_distance(&mut self) -> f32 {
        let mut x0 = 0.0;
        let mut y0 = 0.0;

        let mut stat = Statistics::new();

        for (i, p) in self.intersections().skip(1).enumerate() {
            if i > 0 {
                let dx = p.x - x0;
                let dy = p.y - y0;
                let d = (dx * dx + dy * dy).sqrt();
                if d > self.grid_line_scan_step {
                    stat.add(d, p.weight);
                }
            }
            x0 = p.x;
            y0 = p.y;
        }

        let d_min = stat.xmin.unwrap_or(0.0);
        let d_mean = stat.mean();
        stat.reset();

        for (i, p) in self.intersections().skip(1).enumerate() {
            if i > 0 {
                let dx = p.x - x0;
                let dy = p.y - y0;
                let d = (dx * dx + dy * dy).sqrt();
                if d > d_min && d < 1.2 * d_mean {
                    stat.add(d, p.weight);
                }
            }
            x0 = p.x;
            y0 = p.y;
        }
        stat.mean()
    }

    pub fn fit_index(
        &mut self,
    ) -> (
        Option<QuadraticFunction>,
        Option<QuadraticFunction>,
    ) {
        let xmax = self.characteristics_grid.cols;
        let ymax = self.characteristics_grid.rows;
        self.line_grid_mut().remove_empty();
        let axis = self.line_grid().data[0];
        let orthogonal_axis =
            axis.orthogonal_line_through(xmax as f32 / 2.0, ymax as f32 / 2.0, 100.0);
        let d = self.avg_distance();
        let mut qix = QuadraticFit::new();
        let mut qiy = QuadraticFit::new();
        let mut x0 = 0.0f32;
        let mut y0 = 0.0f32;
        let mut w0 = 0.0f32;
        let mut index = 0.0f32;

        for (i,p) in self.intersections().skip(1).enumerate() {
            if i>0{
                let dx = p.x-x0;
                let dy = p.y-y0;
                let di = (dx*dx+dy*dy).sqrt()/d;
                index+=di.round();
                if (di.round()/di-1.0).abs()<0.2{
                    qix.add(p.x, index, (p.weight*w0).sqrt());
                    qiy.add(p.y, index, (p.weight*w0).sqrt());
                }
            }
            x0 = p.x;
            y0 = p.y;
            w0 = p.weight;
        }
        (qix.quadratic_function(),qiy.quadratic_function())
    }

    /*
    pub fn parallels(&mut self, x: usize, y: usize, orthogonal: bool) -> Parallels {
        let angle = if orthogonal {
            self.main_angle + 64
        } else {
            self.main_angle
        };
        let delta = self.angle_delta;
        let scan_line_length = self.scan_line_length;
        let grid_line_scan_step = self.grid_line_scan_step;
        let xmax = self.characteristics_grid.cols;
        let ymax = self.characteristics_grid.rows;
        let diagonal = ((xmax * xmax + ymax * ymax) as f32).sqrt();
        let c2 = self.c2;
        let fit_distance = self.fit_distance;
        let fit_iterations = self.fit_iterations;
        let fit_extension_factor = self.fit_extension_factor;
        let next_line_offset = grid_line_scan_step + fit_distance as f32;

        let xx = x as f32;
        let yy = y as f32;

        //        let axis = Line::new_from_angle(angle, xmax as f32/2.0, ymax as f32/2.0, diagonal);

        let axis = self.find_line_iterative(
            x,
            y,
            angle,
            delta,
            grid_line_scan_step * 2.0,
            scan_line_length,
            c2,
            fit_distance,
            fit_iterations,
            fit_extension_factor,
        );
        let orthogonal = axis.orthogonal_line_through(xx, yy, 100.0);
        self.line_grid_mut().push(axis);
        let mut line = axis;
        /*
                line = line.parallel_line(next_line_offset).with_length(scan_line_length);
                self.line_grid_mut().push(line);
                line = self.refine_line_iterative(line, angle, delta, c2, fit_distance, fit_iterations, fit_extension_factor);
                self.line_grid_mut().push(line);
        */
    let mut lines = [Line::new(); LINECOUNT];
    let mut distances = [0.0f32; LINECOUNT];
    let mut line_count = 0usize;
    let mut add_line = |line: Line| {
    if line_count < lines.len() {
    if let Some((x, y)) = line.midpoint() {
    lines[line_count] = line;
    distances[line_count] = axis.distance(x, y);
    line_count += 1;
    }
    }
    };

    for i in (0..LINECOUNT_HALF).rev() {
    line = axis
    .parallel_line(-next_line_offset * i as f32)
    .with_length(scan_line_length);
    line = self.refine_line_iterative(
    line,
    angle,
    delta,
    c2,
    fit_distance,
    fit_iterations,
    fit_extension_factor,
    );
    add_line(line);
    self.line_grid_mut().push(line);
    }
    add_line(axis);
    for i in (0..LINECOUNT_HALF) {
    let mut line = axis;
    line = axis
    .parallel_line(-next_line_offset * i as f32)
    .with_length(scan_line_length);
    line = self.refine_line_iterative(
    line,
    angle,
    delta,
    c2,
    fit_distance,
    fit_iterations,
    fit_extension_factor,
    );
    add_line(line);
    self.line_grid_mut().push(line);
    }

    let mut parallels = Parallels::new();
    parallels.parallel_axis = axis;
    parallels.orthogonal_axis = orthogonal;

    if line_count > 2 {
    let mut stat1 = Statistics::new();
    let mut stat2 = Statistics::new();
    for i in 1..line_count {
    let dist = (distances[i] - distances[i - 1]).abs();
    stat1.add(dist, 1.0);
    }
    for i in 1..line_count {
    let dist = (distances[i] - distances[i - 1]).abs();
    stat2.add_average(dist, 1.0, &stat1, 1.0);
    }
    let mean_distance = stat2.mean();
    let mut x_fit = LinearFit::new();
    let mut y_fit = LinearFit::new();
    let mut k_fit = LinearFit::new();

    let line = lines[0];
    let (x0, y0) = if let Some((x, y)) = orthogonal.intersection(line) {
    x_fit.add(0.0, x, line.length());
    y_fit.add(0.0, y, line.length());
    k_fit.add(0.0, line.k, line.length());
    (x, y)
    } else {
    (0.0, 0.0)
    };

    for i in 1..line_count {
    let line = lines[i];
    if let Some((x, y)) = orthogonal.intersection(line) {
    let dx = x - x0;
    let dy = y - y0;
    let j = ((dx * dx + dy * dy).sqrt() / mean_distance).round();
    println!("j={} x={} y={}", j, x, y);
    x_fit.add(j, x, line.length());
    y_fit.add(j, y, line.length());
    k_fit.add(j, line.k, line.length());
    }
    }
    parallels.x = x_fit.line();
    parallels.y = y_fit.line();
    parallels.k = k_fit.line();
    for i in 0..3 {
    let (x, y) = parallels.point(i as f32);
    println!("i={} x={} y={}", i, x, y);
    }
    }
    parallels
    }
     */
}
