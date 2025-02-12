use crate::*;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// A type for which "small" changes can be described using a vector of type `D`, i.e. the type
/// forms a [Differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold).
pub trait Differentiate<D: Vector>: Clone {
    /// Applies a "small" change to this value.
    fn perturb(&mut self, amount: &D);
}

impl Differentiate<Scalar> for Scalar {
    fn perturb(&mut self, amount: &Scalar) {
        *self += *amount
    }
}

/// A [`Diff`] wrapper around a constant value, with the approximation polynomial set to always
/// be zero. This is useful for executing [`Diff`]-enabled code paths without computing gradients.
pub type DiffZero<Out, DOut = Out> = Diff<Zero, Out, DOut>;

/// A value of type `Out` paired with a first-order approximation of how it changes with respect
/// to a "small" perturbation in a parameter of type `DIn`.
pub type DiffL1<DIn, Out, DOut = Out> = Diff<PolyExp1<DIn>, Out, DOut>;

/// A value of type `Out` paired with a second-order approximation of how it changes with respect
/// to a "small" perturbation in a parameter of type `DIn`.
pub type DiffL2<DIn, Out, DOut = Out> = Diff<PolyExp2<DIn>, Out, DOut>;

/// A value of type `Out` paired with a third-order approximation of how it changes with respect
/// to a "small" perturbation in a parameter of type `DIn`.
pub type DiffL3<DIn, Out, DOut = Out> = Diff<PolyExp3<DIn>, Out, DOut>;

/// A value of type `Out` paired with a fourth-order approximation of how it changes with respect
/// to a "small" perturbation in a parameter of type `DIn`.
pub type DiffL4<DIn, Out, DOut = Out> = Diff<PolyExp4<DIn>, Out, DOut>;

/// A value of type `Out` paired with a polynomial approximation of how it changes with respect
/// to a "small" perturbation in some parameter space.
#[repr(C)]
#[derive(Debug, Clone, Copy, Differentiate, Vector)]
pub struct Diff<Poly: PolyMappable, Out: Differentiate<DOut>, DOut: Vector = Out> {
    pub value: Out,
    pub poly: Poly::Map<DOut>,
}

impl<DIn: Mappable, Out: Differentiate<DOut>, DOut: Vector> DiffL1<DIn, Out, DOut> {
    /// Constructs a [`DiffL1`] from a first-degree approximation polynomial with the given
    /// coefficients.
    #[inline]
    pub const fn new(value: Out, c_1: DIn::Map<DOut>) -> Self {
        Diff {
            value,
            poly: Poly1::new(c_1),
        }
    }
}

impl<DIn: Mappable, Out: Differentiate<DOut>, DOut: Vector> DiffL2<DIn, Out, DOut> {
    /// Constructs a [`DiffL2`] from a second-degree approximation polynomial with the given
    /// coefficients.
    #[inline]
    pub const fn new(value: Out, c_1: DIn::Map<DOut>, c_2: DIn::Map<DIn::Map<DOut>>) -> Self {
        Diff {
            value,
            poly: Poly2::new(c_1, c_2),
        }
    }
}

impl<DIn: Mappable, Out: Differentiate<DOut>, DOut: Vector> DiffL3<DIn, Out, DOut> {
    /// Constructs a [`DiffL3`] from a third-degree approximation polynomial with the given
    /// coefficients.
    #[inline]
    pub const fn new(
        value: Out,
        c_1: DIn::Map<DOut>,
        c_2: DIn::Map<DIn::Map<DOut>>,
        c_3: DIn::Map<DIn::Map<DIn::Map<DOut>>>,
    ) -> Self {
        Diff {
            value,
            poly: Poly3::new(c_1, c_2, c_3),
        }
    }
}

impl<DIn: Mappable, Out: Differentiate<DOut>, DOut: Vector> DiffL4<DIn, Out, DOut> {
    /// Constructs a [`DiffL4`] from a fourth-degree approximation polynomial with the given
    /// coefficients.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub const fn new(
        value: Out,
        c_1: DIn::Map<DOut>,
        c_2: DIn::Map<DIn::Map<DOut>>,
        c_3: DIn::Map<DIn::Map<DIn::Map<DOut>>>,
        c_4: DIn::Map<DIn::Map<DIn::Map<DIn::Map<DOut>>>>,
    ) -> Self {
        Diff {
            value,
            poly: Poly4::new(c_1, c_2, c_3, c_4),
        }
    }
}

impl<DIn: Vector, Poly: IdentityPolyRelation<DIn>> Diff<Poly, DIn, DIn> {
    /// Gets the input perturbation for this [`Diff`].
    #[inline]
    pub fn d_in() -> Self {
        Self {
            value: Vector::zero(),
            poly: Poly::poly_identity(),
        }
    }
}

impl<DIn: Vector, Poly: IdentityPolyRelation<DIn>, Out: Differentiate<DIn>> Diff<Poly, Out, DIn> {
    /// Constructs an approximation around the given value. Inputs to the resulting [`Diff`] are
    /// treated as perturbations to this value.
    #[inline]
    pub fn about(value: Out) -> Self {
        Self {
            value,
            poly: Poly::poly_identity(),
        }
    }
}

impl<Poly: PolyMappable, Out: Differentiate<DOut>, DOut: Vector> Diff<Poly, Out, DOut> {
    /// Constructs a value which does not depend on the differential input.
    #[inline]
    pub fn constant(value: Out) -> Self {
        Self {
            value,
            poly: Vector::zero(),
        }
    }

    /// Losslessly converts the underlying polynomial for this differential approximation.
    #[inline]
    pub fn upgrade<NPoly: PolyMappable>(self) -> Diff<NPoly, Out, DOut>
    where
        Poly::Map<DOut>: Into<NPoly::Map<DOut>>,
    {
        Diff {
            value: self.value,
            poly: self.poly.into(),
        }
    }

    /// Removes terms from the approximation polynomial of this value in order to yield a simpler
    /// approximation polynomial.
    #[inline]
    pub fn downgrade<NPoly: PolyMappable>(self) -> Diff<NPoly, Out, DOut>
    where
        Poly: DowngradePolyMappable<NPoly>,
    {
        Diff {
            value: self.value,
            poly: Poly::poly_downgrade::<DOut>(&self.poly),
        }
    }
}

impl<Poly: PolyMappable, Out: Differentiate<DOut>, DOut: Vector> Diff<Poly, Out, DOut> {
    /// Evaluates this approximation with the given differential input.
    #[inline]
    pub fn eval_perturb<DIn: Vector>(&self, d_in: &DIn) -> Out
    where
        Poly: PolyRelation<DIn>,
    {
        let mut value = self.value.clone();
        value.perturb(&Poly::poly_eval(&self.poly, d_in));
        value
    }

    /// "Shifts" this approximation by the given differential input.
    #[inline]
    pub fn shift<DIn: Vector>(self, offset: DIn) -> Self
    where
        Poly: IdentityPolyRelation<DIn>,
    {
        Diff::about(offset).chain(self)
    }

    /// Computes the gradient of this value with respect to the input perturbation.
    #[inline]
    pub fn gradient<DIn: Mappable>(&self) -> Diff<Poly::Gradient, DIn::Map<DOut>>
    where
        Poly: GradientPolyRelation<DIn>,
    {
        Poly::poly_gradient::<DOut>(&self.poly)
    }
}

impl<Out: Differentiate<DOut>, DOut: Vector> DiffZero<Out, DOut> {
    /// Gets the underlying value for this [`DiffZero`]. Since the polynomial approximation is
    /// always zero, no information is lost in this conversion.
    #[inline]
    pub fn unwrap(self) -> Out {
        self.value
    }
}

impl<Poly: PolyMappable, Out: Vector> Diff<Poly, Out> {
    /// Applies a linear map to this value.
    #[inline]
    pub fn map_linear<NOut: Vector>(&self, f: impl Copy + Fn(&Out) -> NOut) -> Diff<Poly, NOut> {
        Diff {
            value: f(&self.value),
            poly: Poly::map_linear(&self.poly, f),
        }
    }

    /// In-place version of [`Diff::map_linear`].
    #[inline]
    pub fn map_linear_inplace<NOut: Vector>(
        &self,
        accum: &mut Diff<Poly, NOut>,
        f: impl Copy + Fn(&Out, &mut NOut),
    ) {
        f(&self.value, &mut accum.value);
        Poly::map_linear_inplace(&self.poly, &mut accum.poly, f);
    }

    /// Applies a function to this value, given a polynomial approximation of the function for any
    /// value.
    #[inline]
    pub fn chain_with<Other: PolyRelation<Out>, NOut: Differentiate<NDOut>, NDOut: Vector>(
        &self,
        diff_f: impl FnOnce(&Out) -> Diff<Other, NOut, NDOut>,
    ) -> Diff<Poly, NOut, NDOut> {
        let mut perturb = *self;
        let res = diff_f(&self.value);
        perturb.value = Out::zero();
        perturb.chain(res)
    }

    /// Interpreting `diff` as a function of input perturbations to outputs, constructs a
    /// new [`Diff`] which composes this with `diff`.
    #[inline]
    pub fn chain<Other: PolyRelation<Out>, NOut: Differentiate<NDOut>, NDOut: Vector>(
        &self,
        diff: Diff<Other, NOut, NDOut>,
    ) -> Diff<Poly, NOut, NDOut> {
        let Diff { mut value, poly } = diff;
        let p = Other::poly_chain::<Poly, NDOut>(self, &poly);
        value.perturb(&p.value);
        Diff {
            value,
            poly: p.poly,
        }
    }

    
    /// Applies this linear map to the given input.
    #[inline]
    pub fn eval<In: Vector>(&self, input: &Diff<Poly, In>) -> Diff<Poly, Out::Out>
    where
        Out: LinearMap<In>,
    {
        let mut res: Diff<Poly, Out::Out> = Vector::zero();
        self.mul_inplace(input, &mut res, |x, y, accum| {
            LinearMap::eval_inplace(x, y, accum, Vector::add_mul_to)
        });
        res
    }

    /// In-place (and generalized) version of [`Diff::mul`].
    #[inline]
    pub fn mul_inplace<Rhs: Vector, Accum: Vector>(
        &self,
        rhs: &Diff<Poly, Rhs>,
        accum: &mut Diff<Poly, Accum>,
        f: impl Copy + Fn(&Out, &Rhs, &mut Accum),
    ) {
        f(&self.value, &rhs.value, &mut accum.value);
        Poly::map_linear_inplace(&self.poly, &mut accum.poly, |x, accum| {
            f(x, &rhs.value, accum)
        });
        Poly::map_linear_inplace(&rhs.poly, &mut accum.poly, |x, accum| {
            f(&self.value, x, accum)
        });
        Poly::poly_mul_inplace(&self.poly, &rhs.poly, &mut accum.poly, f);
    }
}

impl<Poly: PolyMappable, Out: Differentiate<DOut> + PartialEq, DOut: Vector> PartialEq
    for Diff<Poly, Out, DOut>
where
    Poly::Map<DOut>: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.poly == other.poly
    }
}

impl<Poly: PolyMappable, Out: Differentiate<DOut> + Eq, DOut: Vector> Eq for Diff<Poly, Out, DOut> where
    Poly::Map<DOut>: Eq
{
}

impl<Poly: PolyMappable, Out: Vector> Add<Out> for Diff<Poly, Out> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Out) -> Self {
        Self {
            value: self.value + rhs,
            poly: self.poly,
        }
    }
}

impl<Poly: PolyMappable, Out: Vector> Sub<Out> for Diff<Poly, Out> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Out) -> Self {
        Self {
            value: self.value - rhs,
            poly: self.poly,
        }
    }
}

impl<Poly: PolyMappable, A: Vector, B: LinearMap<A>> LinearMap<Diff<Poly, A>> for Diff<Poly, B> {
    type Out = B::Out;
    fn eval_inplace<Accum>(
        &self,
        input: &Diff<Poly, A>,
        accum: &mut Accum,
        f: impl Copy + Fn(&Self::Out, Scalar, &mut Accum),
    ) {
        todo!()
    }
}

impl<Poly: PolyMappable, T: MappableBase> MappableBase for Diff<Poly, T> {
    type Map<Out: Vector> = Diff<Poly, T::Map<Out>>;
}

impl<Poly: PolyMappable, T: Mappable> Mappable for Diff<Poly, T> {
    fn map_new<Out: Vector>(f: impl Copy + Fn(&Self) -> Out) -> Map<Self, Out> {
        todo!()
    }

    fn map_linear_inplace<A: Vector, B: Vector>(
        map: &Map<Self, A>,
        target: &mut Map<Self, B>,
        f: impl Copy + Fn(&A, &mut B),
    ) {
        todo!()
    }
}

impl<Poly: PolyMappable, A: Vector + Mul<B>, B: Vector> Mul<Diff<Poly, B>> for Diff<Poly, A>
where
    <A as Mul<B>>::Output: Vector,
{
    type Output = Diff<Poly, <A as Mul<B>>::Output>;

    #[inline]
    fn mul(self, rhs: Diff<Poly, B>) -> Self::Output {
        let mut res: Self::Output = Vector::zero();
        self.mul_inplace(&rhs, &mut res, |x, y, accum| *accum += *x * *y);
        res
    }
}

impl<Poly: PolyMappable> Div<Scalar> for Diff<Poly, Scalar> {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<Poly: PolyMappable> Div<Diff<Poly, Scalar>> for Scalar {
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

impl<Poly: PolyMappable> Div<Diff<Poly, Scalar>> for Diff<Poly, Scalar> {
    type Output = Diff<Poly, Scalar>;

    #[inline]
    fn div(self, rhs: Diff<Poly, Scalar>) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<Poly: PolyMappable> Rem<Scalar> for Diff<Poly, Scalar> {
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

impl<Poly: PolyMappable> Diff<Poly, Vector2> {
    /// Gets the X component of this vector.
    #[inline]
    pub fn x(&self) -> Diff<Poly, Scalar> {
        self.map_linear(|vec| vec.x)
    }

    /// Gets the Y component of this vector.
    #[inline]
    pub fn y(&self) -> Diff<Poly, Scalar> {
        self.map_linear(|vec| vec.y)
    }

    /// Gets a vector that is the same length and perpendicular to this vector, equivalent to
    /// rotating the vector 90 degrees counter-clockwise.
    #[inline]
    pub fn cross(&self) -> Diff<Poly, Vector2> {
        self.map_linear(|vec| vec.cross())
    }

    /// Computes the dot product of two vectors.
    #[inline]
    pub fn dot(&self, other: &Self) -> Diff<Poly, Scalar> {
        let mut res = Diff::zero();
        self.mul_inplace(other, &mut res, |x, y, accum| *accum += x.dot(y));
        res
    }

    /// The squared L2 norm of this vector.
    #[inline]
    pub fn norm_squared(&self) -> Diff<Poly, Scalar> {
        Diff {
            value: Vector2::dot(&self.value, &self.value),
            poly: Poly::map_compose::<Vector2, _>(&self.poly, &self.value) * 2.0
                + Poly::poly_mul(&self.poly, &self.poly, Vector2::dot),
        }
    }

    /// The L2 norm of this vector.
    #[inline]
    pub fn norm(&self) -> Diff<Poly, Scalar> {
        self.norm_squared().sqrt()
    }

    /// The direction of this vector.
    #[inline]
    pub fn normalize(&self) -> Diff<Poly, Vector2> {
        *self * (1.0 / self.norm())
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
