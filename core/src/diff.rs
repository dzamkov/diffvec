use crate::*;
use core::ops::{Add, Mul, Sub};

/// A type for which "small" changes can be described using a vector of type `D`, i.e. the type
/// forms a [Differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold).
pub trait Differentiate<D: Vector>: Clone {
    /// Gets this value with a "small" change applied to it.
    fn perturb(&self, amount: &D) -> Self {
        let mut res = self.clone();
        res.perturb_mut(amount);
        res
    }

    /// Applies a "small" change to this value.
    fn perturb_mut(&mut self, amount: &D);
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
        self.value.perturb(&Poly::poly_eval(&self.poly, d_in))
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
        value.perturb_mut(&p.value);
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
