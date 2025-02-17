use crate::diff::*;
use crate::map::*;
use crate::poly::*;
use crate::vector::*;
use crate::vector2::*;

/// A scalar value.
pub type Scalar = f32;

/// Archimedes' constant (π).
pub const PI: Scalar = std::f32::consts::PI;

impl Differentiate<Scalar> for Scalar {
    fn perturb_mut(&mut self, amount: &Scalar) {
        *self += *amount
    }
}

impl Differentiate<Scalar> for f64 {
    fn perturb_mut(&mut self, amount: &f32) {
        *self += *amount as f64
    }
}

impl Vector for Scalar {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn add_mul_to(&self, scale: Scalar, output: &mut Self) {
        *output += *self * scale;
    }
}

unsafe impl ContiguousVector for Scalar {
    const DIM: usize = 1;
}

/// A [`LinearMap`] from a [`Scalar`] to some vector type.
pub type ScalarMap<Out> = Out;

impl<T: Vector> LinearMap<Scalar> for T {
    type Out = T;

    #[inline]
    fn eval_inplace<Accum>(
        &self,
        input: &Scalar,
        accum: &mut Accum,
        f: impl Copy + Fn(&Self::Out, Scalar, &mut Accum),
    ) {
        f(self, *input, accum)
    }
}

impl MappableBase for Scalar {
    type Map<Out: Vector> = Out;
}

impl Mappable for Scalar {
    #[inline]
    fn map_new<Out: Vector>(f: impl Copy + Fn(&Self) -> Out) -> Map<Self, Out> {
        f(&1.0)
    }

    #[inline]
    fn map_identity() -> Map<Self, Self> {
        1.0
    }

    #[inline]
    fn map_transpose<Other: Mappable, Out: Vector>(
        source: &Map<Other, Map<Self, Out>>,
    ) -> Map<Self, Map<Other, Out>> {
        *source
    }

    #[inline]
    fn map_linear_inplace<A: Vector, B: Vector>(
        map: &Map<Self, A>,
        target: &mut Map<Self, B>,
        f: impl Copy + Fn(&A, &mut B),
    ) {
        f(map, target)
    }
}

impl<Poly: PolyMappable> core::ops::Div<Scalar> for Diff<Poly, Scalar> {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<Poly: PolyMappable> core::ops::Div<Diff<Poly, Scalar>> for Scalar {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn div(self, rhs: Diff<Poly, Scalar>) -> Self::Output {
        rhs.chain_series_with(|x| {
            let i_x = 1.0 / x;
            let mut coeff = self * i_x;
            core::iter::from_fn(move || {
                let res = coeff;
                coeff *= -i_x;
                Some(res)
            })
        })
    }
}

impl<Poly: PolyMappable> core::ops::Div<Diff<Poly, Scalar>> for Diff<Poly, Scalar> {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn div(self, rhs: Diff<Poly, Scalar>) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<Poly: PolyMappable> core::ops::Rem<Scalar> for Diff<Poly, Scalar> {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn rem(self, rhs: Scalar) -> Self::Output {
        Diff {
            value: self.value % rhs,
            poly: self.poly,
        }
    }
}

impl DiffL1<Scalar, Scalar> {
    /// Find an input perturbation, which, when applied to this differential approximation,
    /// yields the given target value.
    #[inline]
    pub fn solve_perturb(&self, target: Scalar) -> Scalar {
        (target - self.value) / self.poly.c_1
    }

    /// Finds the smallest non-negative input perturbation, which, when applied to this
    /// differential approximation, yields the given target value. Returns [`None`] if no such
    /// perturbation exists.
    #[inline]
    pub fn solve_perturb_nn(&self, target: Scalar) -> Option<Scalar> {
        if self.poly.c_1 == 0.0 {
            if self.value == target {
                Some(0.0)
            } else {
                None
            }
        } else {
            let x_0 = (target - self.value) / self.poly.c_1;
            if x_0 >= 0.0 {
                Some(x_0)
            } else {
                None
            }
        }
    }
}

impl DiffL2<Scalar, Scalar> {
    /// Finds the non-negative input perturbations, which, when applied to this differential
    /// approximation, yield the given target value.
    ///
    /// The returned array is guaranteed to be in increasing order.
    pub fn solve_perturb_nn(&self, target: Scalar) -> arrayvec::ArrayVec<Scalar, 2> {
        let mut res = arrayvec::ArrayVec::new();
        let a = self.poly.c_2;
        let b = self.poly.c_1;
        let c = self.value - target;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let n = -b.signum() * (b.abs() + disc.sqrt());
            if n != 0.0 {
                let x_0 = (2.0 * c) / n;
                if x_0 >= 0.0 {
                    res.push(x_0);
                }
                let x_1 = n / (2.0 * a);
                if x_1 >= 0.0 {
                    res.push(x_1);
                }
            }
        }
        res
    }
}

impl<Poly: PolyMappable> Diff<Poly, Scalar> {
    /// Applies a function to this value, given the coefficients of a Taylor series approximation
    /// of the function at any value.
    #[inline]
    pub fn chain_series_with<Iter: Iterator<Item = Out>, Out: Vector>(
        &self,
        diff_f: impl FnOnce(Scalar) -> Iter,
    ) -> Diff<Poly, Out> {
        let mut series = diff_f(self.value);
        let value = series.next().expect("Series must be non-empty");
        let mut offset = Vector::zero();
        let mut offset_exp = self.poly;
        let degree = Poly::DEGREE.expect("Approximation polynomial must have bounded degree");
        for _ in 0..degree {
            let coeff = series
                .next()
                .expect("Series is too short for approximation polynomial");
            offset += Poly::map_linear(&offset_exp, |x| coeff * *x);
            offset_exp = Poly::poly_mul::<Scalar, Scalar, _>(&offset_exp, &self.poly, |x, y| x * y);
        }
        Diff {
            value,
            poly: offset,
        }
    }

    /// Computes the square root of this value.
    pub fn sqrt(&self) -> Diff<Poly, Scalar> {
        self.chain_series_with(|x| {
            let i_x = 1.0 / x;
            let mut coeff = x.sqrt();
            let mut index = 0;
            core::iter::from_fn(move || {
                let res = coeff;
                index += 1;
                coeff *= (3.0 / 2.0) / (index as Scalar) - 1.0;
                coeff *= i_x;
                Some(res)
            })
        })
    }

    /// Computes the `cos` and `sin` of this value, as a vector.
    pub fn cos_sin(&self) -> Diff<Poly, Vector2> {
        self.chain_series_with(|x| {
            let mut coeff = vec2(x.cos(), x.sin());
            let mut index = 0;
            core::iter::from_fn(move || {
                let res = coeff;
                index += 1;
                coeff = coeff.cross() / (index as Scalar);
                Some(res)
            })
        })
    }

    /// Computes the `sin` of this value.
    pub fn sin(&self) -> Diff<Poly, Scalar> {
        self.cos_sin().y()
    }

    /// Computes the `cos` of this value.
    pub fn cos(&self) -> Diff<Poly, Scalar> {
        self.cos_sin().x()
    }

    /// Computes the `tan` of this value.
    pub fn tan(&self) -> Diff<Poly, Scalar> {
        let cos_sin = self.cos_sin();
        cos_sin.y() / cos_sin.x()
    }

    /// Computes the inverse `sin` of this value.
    pub fn asin(&self) -> Diff<Poly, Scalar> {
        self.chain_series_with(|x| {
            let i_sqr_y = 1.0 / (1.0 - x * x);
            let x_i_sqr_y = x * i_sqr_y;
            let i_x_sqr = (1.0 / x / x).min(Scalar::MAX);
            let mut index = 0;
            let mut parts = [0.0; 4]; // TODO: Size from `Poly::DEGREE` once possible.
            parts[0] = i_sqr_y.sqrt();
            core::iter::once(x.asin()).chain(core::iter::from_fn(move || {
                // `parts[i] = c * x^(index - 2 * i)(1 + x^2)^(-1/2 - index)`.
                let mut res = 0.0;
                index += 1;
                let scale = x_i_sqr_y / (index as Scalar + 1.0);
                let mut from_prev = 0.0;
                #[allow(clippy::needless_range_loop)]
                for i in 0..(index / 2 + 1) {
                    res += parts[i];
                    let to_next = parts[i] * ((index - 2 * i) as Scalar - 1.0);
                    parts[i] *= (index * 2) as Scalar - 1.0;
                    parts[i] -= to_next;
                    parts[i] += from_prev;
                    parts[i] *= scale;
                    from_prev = to_next * i_x_sqr;
                }
                Some(res)
            }))
        })
    }

    /// Computes the inverse `cos` of this value.
    pub fn acos(&self) -> Diff<Poly, Scalar> {
        -self.asin() + PI * 0.5
    }

    /// Computes the inverse `tan` of this value.
    pub fn atan(&self) -> Diff<Poly, Scalar> {
        (*self / (*self * *self + 1.0).sqrt()).asin()
    }
}

#[cfg(test)]
use astro_float::ctx::Context;
#[cfg(test)]
use astro_float::{expr, BigFloat, Consts, RoundingMode};

/// Verifies that the given function produces consistent gradients near the given input.
#[cfg(test)]
fn verify_gradients(
    input: Scalar,
    f: impl Fn(DiffL4<Scalar, Scalar>) -> DiffL4<Scalar, Scalar>,
    g: impl Fn(&BigFloat, &mut Context) -> BigFloat,
) {
    let actual = f(DiffL4::about(input));
    let d_actual = actual.gradient();
    let d_d_actual = d_actual.gradient();
    let d_d_d_actual = d_d_actual.gradient();
    let d_d_d_d_actual = d_d_d_actual.gradient();
    let mut ctx = Context::new(
        1024,
        RoundingMode::Down,
        Consts::new().expect("constants cache initialized"),
        -10000,
        10000,
    );
    let z = &astro_float::BigFloat::from(input);
    let d = &astro_float::BigFloat::from(1.0e-64);
    let samples = [
        &expr!(z - 2.0 * d, &mut ctx),
        &expr!(z - d, &mut ctx),
        z,
        &expr!(z + d, &mut ctx),
        &expr!(z + 2.0 * d, &mut ctx),
    ]
    .map(|x| g(x, &mut ctx));
    let expected = fit(&mut ctx, d, samples);
    let d_expected = expected.gradient();
    let d_d_expected = d_expected.gradient();
    let d_d_d_expected = d_d_expected.gradient();
    let d_d_d_d_expected = d_d_d_expected.gradient();
    let mut max_ulps = 0;
    println!("==============================");
    println!("Input: {}", input);
    let mut summarize = |name, expected: Scalar, actual: Scalar| {
        let ulps = expected.to_bits().abs_diff(actual.to_bits());
        max_ulps = max_ulps.max(ulps);
        println!("{name:<15} | {expected:<25} | {actual:<25} | {ulps:<20}");
    };
    summarize("output", expected.value, actual.value);
    summarize("d_output", d_expected.value, d_actual.value);
    summarize("d_d_output", d_d_expected.value, d_d_actual.value);
    summarize("d_d_d_output", d_d_d_expected.value, d_d_d_actual.value);
    summarize(
        "d_d_d_d_output",
        d_d_d_d_expected.value,
        d_d_d_d_actual.value,
    );
    assert!(max_ulps <= 5, "High error (ulps): {}", max_ulps);

    /// Constructs a second-order approximation of a function based on samples at [x - d, x, x + d].
    fn fit(
        ctx: &mut Context,
        d: &astro_float::BigFloat,
        samples: [astro_float::BigFloat; 5],
    ) -> DiffL4<Scalar, Scalar> {
        let [s_0, s_1, s_2, s_3, s_4] = &samples;
        let c_0 = s_2;
        let c_1 = &expr!((s_3 - s_1) / (2.0 * d), &mut *ctx);
        let c_2 = &expr!((s_3 - 2.0 * s_2 + s_1) / (2.0 * d * d), &mut *ctx);
        let c_3 = &expr!(
            (s_4 - 2.0 * s_3 + 2.0 * s_1 - s_0) / (12.0 * d * d * d),
            &mut *ctx
        );
        let c_4 = &expr!(
            (s_4 - 4.0 * s_3 + 6.0 * s_2 - 4.0 * s_1 + s_0) / (24.0 * d * d * d * d),
            &mut *ctx
        );
        DiffL4::new(
            to_scalar(c_0),
            to_scalar(c_1),
            to_scalar(c_2),
            to_scalar(c_3),
            to_scalar(c_4),
        )
    }

    /// Converts a high-precision float into a [`Scalar`].
    fn to_scalar(float: &astro_float::BigFloat) -> Scalar {
        // TODO: Use conversion function when available
        format!("{}", float).parse().unwrap()
    }
}

#[test]
#[ignore] // TODO: fails due to: https://github.com/stencillogic/astro-float/issues/33
fn test_div() {
    for input in [-2.0, -1.3, 0.5, 0.8, 1.2, 3.0] {
        verify_gradients(
            input,
            |x| 1.0 / (x + 1.0),
            |x, mut ctx| expr!(1.0 / (x + 1.0), ctx),
        );
    }
}

#[test]
fn test_sqrt() {
    for input in [0.5, 0.8, 1.2, 3.0, 5.0] {
        assert_eq!(DiffL4::about(input).sqrt().value, input.sqrt());
        verify_gradients(input, |x| x.sqrt(), |x, mut ctx| expr!(sqrt(x), ctx));
    }
}

#[test]
fn test_sin() {
    for input in [-5.0, -3.0, -0.1, 0.2, 0.5, 0.9, 3.5] {
        assert_eq!(DiffL4::about(input).sin().value, input.sin());
        verify_gradients(input, |x| x.sin(), |x, mut ctx| expr!(sin(x), ctx));
    }
}

#[test]
fn test_cos() {
    for input in [-5.0, -3.0, -0.1, 0.2, 0.5, 0.9, 3.5] {
        assert_eq!(DiffL4::about(input).cos().value, input.cos());
        verify_gradients(input, |x| x.cos(), |x, mut ctx| expr!(cos(x), ctx));
    }
}

#[test]
fn test_tan() {
    use approx::assert_relative_eq;
    for input in [-5.0, -3.0, -0.1, 0.2, 0.5, 0.9, 3.5] {
        assert_relative_eq!(DiffL4::about(input).tan().value, input.tan());
        verify_gradients(input, |x| x.tan(), |x, mut ctx| expr!(tan(x), ctx));
    }
}

#[test]
fn test_asin() {
    for input in [-0.9, -0.5, -0.1, 0.2, 0.6, 0.9] {
        assert_eq!(DiffL4::about(input).asin().value, input.asin());
        verify_gradients(input, |x| x.asin(), |x, mut ctx| expr!(asin(x), ctx));
    }
}
