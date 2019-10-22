use super::utils::*;

#[derive(Debug, Clone, Copy)]
pub struct QuadraticFunction {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl QuadraticFunction {
    pub fn new(a: f32, b: f32, c: f32) -> QuadraticFunction {
        QuadraticFunction { a: a, b: b, c: c }
    }
    pub fn f(&self, x: f32) -> f32 {
        self.a * x * x + self.b * x + self.c
    }
    pub fn closure(&self) -> impl Fn(f32) -> f32 {
        let q = *self;
        move |x| q.f(x)
    }
    pub fn linear(&self) -> QuadraticFunction {
        QuadraticFunction::new(0.0, self.b, self.c)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuadraticFit {
    x1_sum: f32,
    x2_sum: f32,
    x3_sum: f32,
    x4_sum: f32,
    y_sum: f32,
    x1y_sum: f32,
    x2y_sum: f32,
    w_sum: f32,
    xmin: Option<f32>,
    xmax: Option<f32>,
    ymin: Option<f32>,
    ymax: Option<f32>,
    count: usize,
}

impl QuadraticFit {
    pub fn new() -> QuadraticFit {
        QuadraticFit {
            x1_sum: 0.0,
            x2_sum: 0.0,
            x3_sum: 0.0,
            x4_sum: 0.0,
            y_sum: 0.0,
            x1y_sum: 0.0,
            x2y_sum: 0.0,
            w_sum: 0.0,
            xmin: None,
            xmax: None,
            ymin: None,
            ymax: None,
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.x1_sum = 0.0;
        self.x2_sum = 0.0;
        self.x3_sum = 0.0;
        self.x4_sum = 0.0;
        self.y_sum = 0.0;
        self.x1y_sum = 0.0;
        self.x2y_sum = 0.0;
        self.w_sum = 0.0;
        self.xmin = None;
        self.xmax = None;
        self.ymin = None;
        self.ymax = None;
        self.count = 0;
    }

    pub fn add(&mut self, x: f32, y: f32, w: f32) {
        self.x1_sum += w * x;
        self.x2_sum += w * x * x;
        self.x3_sum += w * x * x * x;
        self.x4_sum += w * x * x * x * x;
        self.y_sum += w * y;
        self.x1y_sum += w * y * x;
        self.x2y_sum += w * y * x * x;
        self.w_sum += w;
        self.count += 1;
        if w > 0.0 {
            self.xmin = Some(self.xmin.map_or(x, |old| old.min(x)));
            self.xmax = Some(self.xmax.map_or(x, |old| old.max(x)));
            self.ymin = Some(self.ymin.map_or(y, |old| old.min(y)));
            self.ymax = Some(self.ymax.map_or(y, |old| old.max(y)));
        }
    }

    pub fn det(
        a11: f32,
        a12: f32,
        a13: f32,
        a21: f32,
        a22: f32,
        a23: f32,
        a31: f32,
        a32: f32,
        a33: f32,
    ) -> f32 {
        a11 * a22 * a33 + a12 * a23 * a31 + a21 * a32 * a13
            - a13 * a22 * a31
            - a23 * a32 * a11
            - a12 * a21 * a33
    }

    pub fn d(&self) -> f32 {
        let a0 = self.w_sum;
        let a1 = self.x1_sum;
        let a2 = self.x2_sum;
        let a3 = self.x3_sum;
        let a4 = self.x4_sum;

        Self::det(a4, a3, a2, a3, a2, a1, a2, a1, a0)
    }

    pub fn da(&self) -> f32 {
        let a0 = self.w_sum;
        let a1 = self.x1_sum;
        let a2 = self.x2_sum;
        let a3 = self.x3_sum;
        let a4 = self.x4_sum;
        let b0 = self.y_sum;
        let b1 = self.x1y_sum;
        let b2 = self.x2y_sum;

        Self::det(b2, a3, a2, b1, a2, a1, b0, a1, a0)
    }

    pub fn db(&self) -> f32 {
        let a0 = self.w_sum;
        let a1 = self.x1_sum;
        let a2 = self.x2_sum;
        let a3 = self.x3_sum;
        let a4 = self.x4_sum;
        let b0 = self.y_sum;
        let b1 = self.x1y_sum;
        let b2 = self.x2y_sum;

        Self::det(a4, b2, a2, a3, b1, a1, a2, b0, a0)
    }

    pub fn dc(&self) -> f32 {
        let a0 = self.w_sum;
        let a1 = self.x1_sum;
        let a2 = self.x2_sum;
        let a3 = self.x3_sum;
        let a4 = self.x4_sum;
        let b0 = self.y_sum;
        let b1 = self.x1y_sum;
        let b2 = self.x2y_sum;

        Self::det(a4, a3, b2, a3, a2, b1, a2, a1, b0)
    }

    pub fn coefficients(&self) -> Option<(f32, f32, f32)> {
        let d = self.d();
        if d.abs() > 1.0e-5 {
            Some((self.da() / d, self.db() / d, self.dc() / d))
        } else {
            None
        }
    }

    pub fn quadratic_function(&self) -> Option<QuadraticFunction> {
        self.coefficients()
            .map(|(a, b, c)| QuadraticFunction::new(a, b, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() {
        let mut fit = QuadraticFit::new();
        fit.add(-1.0, 1.0, 1.0);
        fit.add(0.0, 0.0, 1.0);
        fit.add(1.0, 1.0, 1.0);
        let c = fit.coefficients();
        assert_eq!(c, Some((1.0, 0.0, 0.0)));
    }
}
