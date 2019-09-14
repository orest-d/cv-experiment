use super::utils::*;
use super::characteristics_grid::*;

const POINT_BUFFER_SIZE: usize = 10*10;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum LineType{
    FX,
    FY,
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedPoint{
    x: f32,
    y: f32,
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line{
    line_type: LineType,
    point:WeightedPoint,
    k: f32,
    x1:f32,
    x2:f32
}

impl Line{
    pub fn new() -> Line{
        Line{line_type:LineType::Empty, point:WeightedPoint{x:0.0,y:0.0,weight:0.0}, k:0.0, x1:0.0, x2:0.0}
    }
    pub fn fit<'a,I>(points:I) -> Self
    where I:IntoIterator<Item=&'a WeightedPoint>{
        let mut x_sum=0.0;
        let mut y_sum=0.0;
        let mut xx_sum=0.0;
        let mut yy_sum=0.0;
        let mut xy_sum=0.0;
        let mut w_sum=0.0;
        let mut count=0.0;
        let mut xmin:Option<f32> = None;
        let mut xmax:Option<f32> = None;
        let mut ymin:Option<f32> = None;
        let mut ymax:Option<f32> = None;

        for &WeightedPoint{x:x,y:y,weight:w} in points{
            x_sum += w*x;
            y_sum += w*y;
            xx_sum += w*x*x;
            yy_sum += w*y*y;
            xy_sum += w*x*y;
            w_sum += w;
            count+=1.0;
            if w>0.0{
                xmin = Some(xmin.map_or(x,|old| old.min(x)));
                xmax = Some(xmax.map_or(x,|old| old.max(x)));
                ymin = Some(ymin.map_or(y,|old| old.min(y)));
                ymax = Some(ymax.map_or(y,|old| old.max(y)));
            }
        }
        let mut line = Line::new();
        if w_sum>0.0{
            let x = x_sum/w_sum;
            let y = y_sum/w_sum;
            let xx = xx_sum/w_sum - x*x;
            let yy = yy_sum/w_sum - y*y;
            let xy = xy_sum/w_sum - x*y;
            if xx.abs()>0.0 || yy.abs()>0.0{                
                line.point.weight=w_sum/count;
                if xx.abs()>=yy.abs(){
                    line.line_type = LineType::FX;
                    line.point.x = x;
                    line.point.y = y;
                    line.k = xy/xx;
                    line.x1 = xmin.unwrap_or(x);
                    line.x2 = xmax.unwrap_or(x);
                }
                else{
                    line.line_type = LineType::FY;
                    line.point.x = y;
                    line.point.y = x;
                    line.k = xy/yy;
                    line.x1 = ymin.unwrap_or(y);
                    line.x2 = ymax.unwrap_or(y);
                }
            }
        }
        line
    }
    
    pub fn fit_region(grid:&CharacteristicsGrid, x:usize, y:usize, dx:usize, dy:usize, angle:u8, delta:u8)->Line{
        let mut buffer = [WeightedPoint{x:0.0,y:0.0,weight:0.0};POINT_BUFFER_SIZE];
        let mut count = 0;

        let points = region_indices(x,y,dx,dy).filter_map(
            |(x,y)| {
                let c = grid.get(x,y);
                if angle_in_range(c.angle,angle,delta){
                    Some(WeightedPoint{x:x as f32, y:y as f32, weight:c.intensity as f32})
                }
                else{
                    None
                }
            }
        );

        for (slot,point) in buffer.iter_mut().zip(points){
            *slot = point;
            count+=1;
        }

        Line::fit(&buffer[0..count])
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
    fn test_fit_simple() {
        let line = Line::fit(&[ 
                WeightedPoint{x:0.0,y:0.0,weight:1.0},
                WeightedPoint{x:1.0,y:1.0,weight:1.0},
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
    fn test_fit_fy() {
        let line = Line::fit(&[ 
                WeightedPoint{x:0.0,y:0.0,weight:1.0},
                WeightedPoint{x:1.0,y:2.0,weight:1.0},
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
    fn test_fit_fx_weighted() {
        let line = Line::fit(&[ 
                WeightedPoint{x:1.0,y:2.0,weight:1.0},
                WeightedPoint{x:4.0,y:5.0,weight:2.0},
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