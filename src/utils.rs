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
