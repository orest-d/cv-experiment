pub fn angle_in_range(a: u8, mean: u8, delta: u8) -> bool {
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

pub fn angle_difference(a: u8, mean: u8) -> u8 {
    let a = a as i16;
    let mean = mean as i16;

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
    x as u8
}

pub fn angle_difference_c2(a: u8, mean: u8) -> u8 {
    angle_difference(a, mean).min(angle_difference(a, mean.wrapping_add(128)))
}

pub fn neighbor_indices(x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..2)
        .into_iter()
        .flat_map(move |j| (0..2).into_iter().map(move |i| (i + x, j + y)))
}

pub fn region_indices(
    x: usize,
    y: usize,
    dx: usize,
    dy: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (0..dy)
        .into_iter()
        .flat_map(move |j| (0..dx).into_iter().map(move |i| (i + x, j + y)))
}



#[derive(Debug, Copy, Clone)]
pub struct Largest<T: Copy+Clone> {
    pub max_value:f32,
    pub max_item:Option<T>,
} 

impl<T> Largest<T> where T:Copy+Clone{
    pub fn new()->Largest<T>{
        Largest{
            max_value:0.0,
            max_item:None,
        }
    }

    pub fn add(&mut self, value:f32, item:T){
        match self.max_item{
            Some(_) => {
                if value>self.max_value{
                    self.max_value = value;
                    self.max_item = Some(item);
                }
            },
            None => {
                self.max_value = value;
                self.max_item = Some(item);
            }
        }
    }

}

#[derive(Debug, Copy, Clone)]
pub enum TwoLargestCount{
    Empty,
    One,
    More
}

#[derive(Debug, Copy, Clone)]
pub struct TwoLargest<T: Copy+Clone> {
    pub count:TwoLargestCount,
    pub max_value:f32,
    pub max_item:Option<T>,
    pub second_max_value:f32,
    pub second_max_item:Option<T>,
} 

impl<T> TwoLargest<T> where T:Copy+Clone{
    pub fn new()->TwoLargest<T>{
        TwoLargest{
            count:TwoLargestCount::Empty,
            max_value:0.0,
            max_item:None,
            second_max_value:0.0,
            second_max_item:None
        }
    }

    fn add2(&mut self, value:f32, item:T){
        if value<=self.max_value{
            self.second_max_value = value;
            self.second_max_item = Some(item);
        }
        else{
            self.second_max_value = self.max_value;
            self.second_max_item = self.max_item;
            self.max_value = value;
            self.max_item = Some(item);
        }
    }

    pub fn add(&mut self, value:f32, item:T){
        match self.count {
            TwoLargestCount::Empty => {
                self.count = TwoLargestCount::One;
                self.max_value = value;
                self.max_item = Some(item);
            },
            TwoLargestCount::One => {
                self.count = TwoLargestCount::More;
                self.add2(value, item);
            },
            TwoLargestCount::More => {
                if value>self.second_max_value{
                    self.add2(value, item);
                }            
            }

        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    x_sum: f32,
    xx_sum: f32,
    w_sum: f32,
    xmin: Option<f32>,
    xmax: Option<f32>,
    count: usize,
}

impl Statistics {
    pub fn new() -> Statistics {
        Statistics {
            x_sum: 0.0,
            xx_sum: 0.0,
            w_sum: 0.0,
            xmin: None,
            xmax: None,
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.x_sum = 0.0;
        self.xx_sum = 0.0;
        self.w_sum = 0.0;
        self.xmin = None;
        self.xmax = None;
        self.count = 0;
    }

    pub fn add(&mut self, x: f32, w: f32) {
        self.x_sum += w * x;
        self.xx_sum += w * x * x;
        self.w_sum += w;
        self.count += 1;
        if w > 0.0 {
            self.xmin = Some(self.xmin.map_or(x, |old| old.min(x)));
            self.xmax = Some(self.xmax.map_or(x, |old| old.max(x)));
        }
    }

    pub fn add_average(&mut self, x: f32, w: f32, stat:&Statistics, sigma_multiple:f32) {
        if stat.xmin.map_or(true, |xm| x>xm){
            if stat.xmax.map_or(true, |xm| x<xm){
                let delta = stat.sigma()*sigma_multiple;
                let mean = stat.mean();
                if x>=mean-delta && x<=mean+delta{
                    self.add(x,w);
                }
            }
        }
    }

    pub fn mean(&self) ->f32 {
        self.x_sum/self.w_sum
    }

    pub fn variance(&self) ->f32 {
        let m = self.mean();
        self.xx_sum/self.w_sum - m*m
    }

    pub fn sigma(&self) ->f32 {
        self.variance().sqrt()
    }
}

pub fn determinant(axx:f32, axy:f32, ayx:f32, ayy:f32) -> f32{
    axx*ayy-axy*ayx
}
pub fn solve2x2(axx:f32, axy:f32, ayx:f32, ayy:f32, bx:f32, by:f32) -> Option<(f32,f32)>{
    let d = determinant(axx, axy, ayx, ayy);
    if d==0.0{
        None
    }
    else{
        Some((determinant(bx, axy, by, ayy)/d, determinant(axx,bx, ayx, by)/d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        let mut iter = (0..2).into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_region_indices() {
        let mut iter = region_indices(2, 3, 1, 1);
        assert_eq!(iter.next(), Some((2, 3)));
        assert_eq!(iter.next(), None);
        let mut iter = region_indices(2, 3, 2, 2);
        assert_eq!(iter.next(), Some((2, 3)));
        assert_eq!(iter.next(), Some((3, 3)));
        assert_eq!(iter.next(), Some((2, 4)));
        assert_eq!(iter.next(), Some((3, 4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_angle_in_range() {
        assert!(!angle_in_range(0, 0, 0));
        assert!(angle_in_range(0, 0, 1));
        assert!(!angle_in_range(1, 0, 1));
        assert!(!angle_in_range(255, 0, 1));
        assert!(angle_in_range(1, 0, 2));
        assert!(angle_in_range(255, 0, 2));

        assert!(!angle_in_range(254, 1, 3)); // middle - 3 => OUT
        assert!(angle_in_range(255, 1, 3)); // middle - 2 => IN
        assert!(angle_in_range(0, 1, 3)); // middle - 1 => IN
        assert!(angle_in_range(1, 1, 3)); // middle     => IN
        assert!(angle_in_range(2, 1, 3)); // middle + 1 => IN
        assert!(angle_in_range(3, 1, 3)); // middle + 2 => IN
        assert!(!angle_in_range(4, 1, 3)); // middle + 3 => OUT

        assert!(!angle_in_range(251, 254, 3)); // middle - 3 => OUT
        assert!(angle_in_range(252, 254, 3)); // middle - 2 => IN
        assert!(angle_in_range(253, 254, 3)); // middle - 1 => IN
        assert!(angle_in_range(254, 254, 3)); // middle     => IN
        assert!(angle_in_range(255, 254, 3)); // middle + 1 => IN
        assert!(angle_in_range(0, 254, 3)); // middle + 2 => IN
        assert!(!angle_in_range(1, 254, 3)); // middle + 3 => OUT
    }

    #[test]
    fn test_angle_difference() {
        assert_eq!(angle_difference(0, 0), 0);
        assert_eq!(angle_difference(0, 1), 1);
        assert_eq!(angle_difference(1, 0), 1);
        assert_eq!(angle_difference(0, 255), 1);
        assert_eq!(angle_difference(255, 0), 1);

        assert_eq!(angle_difference(0, 128), 128);
        assert_eq!(angle_difference(0, 129), 127);
        assert_eq!(angle_difference(128, 0), 128);
        assert_eq!(angle_difference(129, 0), 127);
        assert_eq!(angle_difference(1, 128), 127);
        assert_eq!(angle_difference(255, 128), 127);
        assert_eq!(angle_difference(128, 1), 127);
        assert_eq!(angle_difference(128, 255), 127);
    }

    #[test]
    fn test_angle_difference_c2() {
        assert_eq!(angle_difference_c2(0, 0), 0);
        assert_eq!(angle_difference_c2(0, 1), 1);
        assert_eq!(angle_difference_c2(1, 0), 1);
        assert_eq!(angle_difference_c2(0, 255), 1);
        assert_eq!(angle_difference_c2(255, 0), 1);

        assert_eq!(angle_difference_c2(0, 128), 0);
        assert_eq!(angle_difference_c2(0, 129), 1);
        assert_eq!(angle_difference_c2(128, 0), 0);
        assert_eq!(angle_difference_c2(129, 0), 1);
        assert_eq!(angle_difference_c2(1, 128), 1);
        assert_eq!(angle_difference_c2(255, 128), 1);
        assert_eq!(angle_difference_c2(128, 1), 1);
        assert_eq!(angle_difference_c2(128, 255), 1);
    }

    #[test]
    fn test_largest(){
        let mut l:Largest<i32> = Largest::new();
        assert_eq!(l.max_item, None);
        l.add(1.0,123);
        assert_eq!(l.max_item, Some(123));
        l.add(2.0,23);
        assert_eq!(l.max_item, Some(23));
        l.add(1.0,234);
        assert_eq!(l.max_item, Some(23));
    }

    #[test]
    fn test_two_largest(){
        let mut tl:TwoLargest<i32> = TwoLargest::new();
        assert_eq!(tl.max_item, None);
        tl.add(1.0,123);
        assert_eq!(tl.max_item, Some(123));
        assert_eq!(tl.second_max_item, None);
        tl.add(2.0,23);
        assert_eq!(tl.max_item, Some(23));
        assert_eq!(tl.second_max_item, Some(123));
    }
}
