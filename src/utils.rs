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

pub fn neighbor_indices(x:usize,y:usize)->impl Iterator<Item=(usize,usize)>{
    (0..2).into_iter().flat_map(
        move |j| (0..2).into_iter().map(move |i| (i+x,j+y))
    )
}

pub fn region_indices(x:usize,y:usize,dx:usize,dy:usize)->impl Iterator<Item=(usize,usize)>{
    (0..dy).into_iter().flat_map(
        move |j| (0..dx).into_iter().map(move |i| (i+x,j+y))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        let mut iter = (0..2).into_iter();        
        assert_eq!(iter.next(),Some(0));
        assert_eq!(iter.next(),Some(1));
        assert_eq!(iter.next(),None);
    }

    #[test]
    fn test_region_indices() {
        let mut iter = region_indices(2, 3, 1, 1);        
        assert_eq!(iter.next(),Some((2,3)));
        assert_eq!(iter.next(),None);
        let mut iter = region_indices(2, 3, 2, 2);
        assert_eq!(iter.next(),Some((2,3)));
        assert_eq!(iter.next(),Some((3,3)));
        assert_eq!(iter.next(),Some((2,4)));
        assert_eq!(iter.next(),Some((3,4)));
        assert_eq!(iter.next(),None);
    }

    #[test]
    fn test_angle_in_range() {
        assert!(!angle_in_range(0, 0, 0));
        assert!(angle_in_range(0, 0, 1));
        assert!(!angle_in_range(1, 0, 1));
        assert!(!angle_in_range(255, 0, 1));
        assert!(angle_in_range(1, 0, 2));
        assert!(angle_in_range(255, 0, 2));

        assert!(!angle_in_range(254, 1, 3));// middle - 3 => OUT
        assert!(angle_in_range(255, 1, 3)); // middle - 2 => IN
        assert!(angle_in_range(0, 1, 3));   // middle - 1 => IN
        assert!(angle_in_range(1, 1, 3));   // middle     => IN
        assert!(angle_in_range(2, 1, 3));   // middle + 1 => IN
        assert!(angle_in_range(3, 1, 3));   // middle + 2 => IN
        assert!(!angle_in_range(4, 1, 3));  // middle + 3 => OUT

        assert!(!angle_in_range(251, 254, 3));// middle - 3 => OUT
        assert!(angle_in_range(252, 254, 3)); // middle - 2 => IN
        assert!(angle_in_range(253, 254, 3));   // middle - 1 => IN
        assert!(angle_in_range(254, 254, 3));   // middle     => IN
        assert!(angle_in_range(255, 254, 3));   // middle + 1 => IN
        assert!(angle_in_range(0, 254, 3));   // middle + 2 => IN
        assert!(!angle_in_range(1, 254, 3));  // middle + 3 => OUT
    }

}