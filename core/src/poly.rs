#![allow(clippy::type_complexity)]
use crate::diff::Diff;
use crate::map::*;
use crate::vector::*;

/// A [`Mappable`] where the components of a linear map correspond to terms in a polynomial. This
/// induces a notion of multiplication between maps, implemented by combining terms and multiplying
/// their coefficients.
pub trait PolyMappable: Mappable {
    /// An upper bound on the number of times a polynomial can be multiplied before it becomes
    /// zero (due to the discarding of high-degree terms). This will be [`None`] for polynomials
    /// that have a constant term and polynomials that due not discard terms during multiplication.
    /// For example, the degree of a linear polynomial is 1, because multiplying two linear
    /// polynomials results only in quadratic terms which are discarded.
    const DEGREE: Option<usize>;

    /// Multiplies the given polynomials using the bilinear function `f` to multiply coefficients.
    /// Terms whose degree is too high to be included in the resulting polynomial will be
    /// discarded.
    #[inline]
    fn poly_mul<A: Vector, B: Vector, Out: Vector>(
        a: &Map<Self, A>,
        b: &Map<Self, B>,
        f: impl Copy + Fn(&A, &B) -> Out,
    ) -> Map<Self, Out> {
        let mut res = Vector::zero();
        Self::poly_mul_inplace(a, b, &mut res, |a, b, out| *out = f(a, b));
        res
    }

    /// In-place version of [`PolyMappable::poly_mul`].
    fn poly_mul_inplace<A: Vector, B: Vector, Out: Vector>(
        a: &Map<Self, A>,
        b: &Map<Self, B>,
        accum: &mut Map<Self, Out>,
        f: impl Copy + Fn(&A, &B, &mut Out),
    );
}

impl PolyMappable for Zero {
    const DEGREE: Option<usize> = Some(0);
    fn poly_mul_inplace<A: Vector, B: Vector, Out: Vector>(
        _: &Map<Self, A>,
        _: &Map<Self, B>,
        _: &mut Map<Self, Out>,
        _: impl Copy + Fn(&A, &B, &mut Out),
    ) {
        // Nothing to do here
    }
}

/// A [`PolyMappable`] whose polynomial variables correspond to components of a vector of
/// type `In`.
pub trait PolyRelation<In: Vector>: PolyMappable {
    /// Interpreting `map` as a polynomial, evaluates it for the given input.
    #[inline]
    fn poly_eval<Out: Vector>(poly: &Map<Self, Out>, input: &In) -> Out {
        Self::poly_chain(&Diff::constant(*input), poly).unwrap()
    }

    /// Composes two polynomials such that evaluating the resulting polynomial is approximately
    /// equivalent to evaluating both polynomials in order.
    fn poly_chain<Other: PolyMappable, Out: Vector>(
        a: &Diff<Other, In>,
        b: &Map<Self, Out>,
    ) -> Diff<Other, Out>;

    /// Shortcut for `poly_chain(a, b).eval(a)`.
    ///
    /// May be optimized by some implementations.
    #[inline]
    fn poly_chain_eval<Other: PolyMappable, Out: Vector>(
        a: &Diff<Other, In>,
        b: &Map<Self, Map<In, Out>>,
    ) -> Diff<Other, Out>
    where
        In: Mappable,
    {
        Self::poly_chain::<Other, Map<In, Out>>(a, b).eval(a)
    }
}

impl<T: Vector> PolyRelation<T> for Zero {
    #[inline]
    fn poly_eval<Out: Vector>(_: &Map<Self, Out>, _: &T) -> Out {
        Out::zero()
    }

    #[inline]
    fn poly_chain<Other: PolyMappable, Out: Vector>(
        _: &Diff<Other, T>,
        _: &Map<Self, Out>,
    ) -> Diff<Other, Out> {
        Diff::zero()
    }

    #[inline]
    fn poly_chain_eval<Other: PolyMappable, Out: Vector>(
        _: &Diff<Other, T>,
        _: &Map<Self, Map<T, Out>>,
    ) -> Diff<Other, Out>
    where
        T: Mappable,
    {
        Diff::zero()
    }
}

/// A [`PolyMappable`] for which an "identity" polynomial exists.
pub trait IdentityPolyRelation<In: Vector>: PolyRelation<In> {
    /// Gets the identity polynomial for this type.
    fn poly_identity() -> Map<Self, In>;
}

impl<T: PolyRelation<T>> IdentityPolyRelation<T> for T {
    #[inline]
    fn poly_identity() -> Map<Self, T> {
        Self::map_identity()
    }
}

/// A [`PolyRelation`] for polynomials that allow the gradient/derivative to be computed.
pub trait GradientPolyRelation<In: Mappable>: PolyRelation<In> {
    /// The [`PolyRelation`] for polynomials produced by `gradient_poly`.
    type Gradient: PolyRelation<In>;

    /// Computes the gradient of the given polynomial.
    fn poly_gradient<Out: Vector>(poly: &Map<Self, Out>) -> Diff<Self::Gradient, In::Map<Out>>;
}

impl<T: PolyRelation<T>> GradientPolyRelation<T> for T {
    type Gradient = Zero;

    #[inline]
    fn poly_gradient<Out: Vector>(poly: &Map<Self, Out>) -> Diff<Zero, T::Map<Out>> {
        Diff {
            value: *poly,
            poly: ZeroMap::new(),
        }
    }
}

/// A [`PolyMappable`] for polynomials which can be "downgraded" by truncating high-degree terms.
pub trait DowngradePolyMappable<ToPoly: PolyMappable>: PolyMappable {
    /// Removes terms from the given polynomial in order to make it a `ToPoly` polynomial.
    fn poly_downgrade<Out: Vector>(poly: &Map<Self, Out>) -> ToPoly::Map<Out>;
}

impl DowngradePolyMappable<Zero> for Zero {
    #[inline]
    fn poly_downgrade<Out: Vector>(poly: &ZeroMap<Out>) -> ZeroMap<Out> {
        *poly
    }
}

/// A first-degree [`PolyMappable`] for the given vector type.
pub type PolyExp1<In> = Poly1<In, Scalar>;

/// A second-degree [`PolyMappable`] for the given vector type.
pub type PolyExp2<In> = Poly2<In, Scalar>;

/// A third-degree [`PolyMappable`] for the given vector type.
pub type PolyExp3<In> = Poly3<In, Scalar>;

/// A fourth-degree [`PolyMappable`] for the given vector type.
pub type PolyExp4<In> = Poly4<In, Scalar>;

/// A first-degree polynomial function from `In` to `Out`.
pub type Poly1<In, Out> = PolyN<In, Out, Zero>;

/// A second-degree polynomial function from `In` to `Out`.
pub type Poly2<In, Out> = PolyN<In, Out, PolyExp1<In>>;

/// A third-degree polynomial function from `In` to `Out`.
pub type Poly3<In, Out> = PolyN<In, Out, PolyExp2<In>>;

/// A fourth-degree polynomial function from `In` to `Out`.
pub type Poly4<In, Out> = PolyN<In, Out, PolyExp3<In>>;

/// A polynomial function from `In` to `Out`.
#[repr(C)]
#[derive(Clone, Copy, crate::Differentiate, crate::Vector)]
pub struct PolyN<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> {
    /// The first degree coefficients for the polynomial.
    pub c_1: In::Map<Out>,

    /// The higher-degree coefficients for the polynomial.
    pub rem: Rem::Map<In::Map<Out>>,
}

impl<In: Mappable, Out: Vector> Poly1<In, Out> {
    /// Constructs a first-degree polynomial from the given coefficient.
    #[inline]
    pub const fn new(c_1: In::Map<Out>) -> Self {
        Self {
            c_1,
            rem: ZeroMap::new(),
        }
    }
}

impl<In: Mappable, Out: Vector> Poly2<In, Out> {
    /// Constructs a second-degree polynomial from the given coefficients.
    #[inline]
    pub const fn new(c_1: In::Map<Out>, c_2: In::Map<In::Map<Out>>) -> Self {
        Self {
            c_1,
            rem: Poly1::new(c_2),
        }
    }
}

impl<In: Mappable, Out: Vector> Poly3<In, Out> {
    /// Constructs a third-degree polynomial from the given coefficients.
    #[inline]
    pub const fn new(
        c_1: In::Map<Out>,
        c_2: In::Map<In::Map<Out>>,
        c_3: In::Map<In::Map<In::Map<Out>>>,
    ) -> Self {
        Self {
            c_1,
            rem: Poly2::new(c_2, c_3),
        }
    }
}

impl<In: Mappable, Out: Vector> Poly4<In, Out> {
    /// Constructs a fourth-degree polynomial from the given coefficients.
    #[inline]
    pub const fn new(
        c_1: In::Map<Out>,
        c_2: In::Map<In::Map<Out>>,
        c_3: In::Map<In::Map<In::Map<Out>>>,
        c_4: In::Map<In::Map<In::Map<In::Map<Out>>>>,
    ) -> Self {
        Self {
            c_1,
            rem: Poly3::new(c_2, c_3, c_4),
        }
    }
}

impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> PolyN<In, Out, Rem> {
    /// Computes the gradient of this polynomial.
    #[inline]
    pub fn gradient(&self) -> Diff<Rem, In::Map<Out>> {
        Diff {
            value: self.c_1,
            poly: Rem::rem_gradient_poly::<Out>(&self.rem),
        }
    }

    /// Removes the highest-degree terms from this polynomial, reducing its degree by 1.
    #[inline]
    pub fn downgrade_1(&self) -> &Rem::Map<Out> {
        util::SimplePolyRelation::downgrade_1_poly(self)
    }
}

/// Defines helper types for [`Poly`].
pub mod util {
    use super::*;
    use std::mem::transmute;
    use std::ops::{Deref, DerefMut};

    /// A [`PolyRelation`] whose polynomials are either [`Zero`] or [`PolyN`]. This ensures that
    /// coefficients are stored in increasing order of degree.
    ///
    /// # Safety
    /// The implementor must ensure that a `&Poly<In, Out, Self>` can be directly cast
    /// to a `&Map<Self, Out>` to remove the highest-degree term. All sensible implementations have
    /// already been implemented.
    pub unsafe trait SimplePolyRelation<In: Mappable>: PolyRelation<In> {
        /// Interpreting the given polynomial as one degree higher than it actually is, computes
        /// its gradient.
        fn rem_gradient_poly<Out: Vector>(
            source: &Map<Self, Map<In, Out>>,
        ) -> Map<Self, Map<In, Out>>;

        /// Reduces the degree of the given polynomial by truncating the highest-degree terms.
        #[inline]
        fn downgrade_1_poly<Out: Vector>(source: &PolyN<In, Out, Self>) -> &Map<Self, Out> {
            // SAFETY: By the requirements of `SimplePolyRelation`, this cast is valid.
            unsafe { std::mem::transmute(source) }
        }

        /// Increases the degree of the given polynomial by adding high-degree terms with
        /// coefficients set to zero.
        fn upgrade_1_poly<Out: Vector>(source: Map<Self, Out>) -> PolyN<In, Out, Self>;
    }

    unsafe impl<In: Mappable> SimplePolyRelation<In> for Zero {
        #[inline]
        fn rem_gradient_poly<Out: Vector>(_: &Map<Self, Map<In, Out>>) -> Map<Self, Map<In, Out>> {
            ZeroMap::new()
        }

        #[inline]
        fn upgrade_1_poly<Out: Vector>(_: Map<Self, Out>) -> PolyN<In, Out, Self> {
            PolyN {
                c_1: Vector::zero(),
                rem: ZeroMap::new(),
            }
        }
    }

    unsafe impl<In: Mappable, Rem: SimplePolyRelation<In>> SimplePolyRelation<In>
        for PolyN<In, Scalar, Rem>
    {
        #[inline]
        fn rem_gradient_poly<Out: Vector>(
            source: &PolyN<In, Map<In, Out>, Rem>,
        ) -> PolyN<In, Map<In, Out>, Rem> {
            PolyN {
                c_1: source.c_1 + In::map_transpose::<In, Out>(&source.c_1),
                rem: source.rem
                    + Rem::rem_gradient_poly::<Map<In, Out>>(&Rem::map_linear(&source.rem, |x| {
                        In::map_transpose::<In, Out>(x)
                    })),
            }
        }

        #[inline]
        fn upgrade_1_poly<Out: Vector>(source: Map<Self, Out>) -> PolyN<In, Out, Self> {
            PolyN {
                c_1: source.c_1,
                rem: Rem::upgrade_1_poly(source.rem),
            }
        }
    }

    #[repr(C)]
    pub struct Poly2<In: Mappable, Out: Vector> {
        c_1: In::Map<Out>,
        pub c_2: In::Map<In::Map<Out>>,
    }

    #[repr(C)]
    pub struct Poly3<In: Mappable, Out: Vector> {
        c_1: In::Map<Out>,
        pub c_2: In::Map<In::Map<Out>>,
        pub c_3: In::Map<In::Map<In::Map<Out>>>,
    }

    #[repr(C)]
    pub struct Poly4<In: Mappable, Out: Vector> {
        c_1: In::Map<Out>,
        pub c_2: In::Map<In::Map<Out>>,
        pub c_3: In::Map<In::Map<In::Map<Out>>>,
        pub c_4: In::Map<In::Map<In::Map<In::Map<Out>>>>,
    }

    impl<In: Mappable, Out: Vector> Deref for super::Poly2<In, Out> {
        type Target = Poly2<In, Out>;

        #[inline]
        fn deref(&self) -> &Self::Target {
            unsafe { transmute(self) }
        }
    }

    impl<In: Mappable, Out: Vector> DerefMut for super::Poly2<In, Out> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { transmute(self) }
        }
    }

    impl<In: Mappable, Out: Vector> Deref for super::Poly3<In, Out> {
        type Target = Poly3<In, Out>;

        #[inline]
        fn deref(&self) -> &Self::Target {
            unsafe { transmute(self) }
        }
    }

    impl<In: Mappable, Out: Vector> DerefMut for super::Poly3<In, Out> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { transmute(self) }
        }
    }

    impl<In: Mappable, Out: Vector> Deref for super::Poly4<In, Out> {
        type Target = Poly4<In, Out>;

        #[inline]
        fn deref(&self) -> &Self::Target {
            unsafe { transmute(self) }
        }
    }

    impl<In: Mappable, Out: Vector> DerefMut for super::Poly4<In, Out> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { transmute(self) }
        }
    }

    pub trait DebugPoly: std::fmt::Debug {
        /// Appends the coefficients of this polynomial to the given `DebugStruct`, and then
        /// flushes its output.
        fn fmt_inline(&self, st: std::fmt::DebugStruct, c_index: usize) -> std::fmt::Result;
    }

    impl<Out: Vector + std::fmt::Debug> DebugPoly for ZeroMap<Out> {
        fn fmt_inline(&self, mut st: std::fmt::DebugStruct, _: usize) -> std::fmt::Result {
            st.finish()
        }
    }

    impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> DebugPoly for PolyN<In, Out, Rem>
    where
        In::Map<Out>: std::fmt::Debug,
        Rem::Map<In::Map<Out>>: DebugPoly,
    {
        fn fmt_inline(&self, mut st: std::fmt::DebugStruct, c_index: usize) -> std::fmt::Result {
            fn as_dyn_debug<T: std::fmt::Debug>(x: &T) -> &dyn std::fmt::Debug {
                x
            }
            st.field(&format!("c_{}", c_index), as_dyn_debug(&self.c_1));
            self.rem.fmt_inline(st, c_index + 1)
        }
    }
}

impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> PartialEq for PolyN<In, Out, Rem>
where
    In::Map<Out>: PartialEq,
    Rem::Map<In::Map<Out>>: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.c_1 == other.c_1 && self.rem == other.rem
    }
}

impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> Eq for PolyN<In, Out, Rem>
where
    In::Map<Out>: Eq,
    Rem::Map<In::Map<Out>>: Eq,
{
}

impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> std::fmt::Debug
    for PolyN<In, Out, Rem>
where
    PolyN<In, Out, Rem>: util::DebugPoly,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = f.debug_struct("PolyN");
        util::DebugPoly::fmt_inline(self, st, 1)
    }
}

impl<In: Mappable, Out: Vector, Rem: util::SimplePolyRelation<In>> From<PolyN<In, Out, Rem>>
    for PolyN<In, Out, PolyN<In, Scalar, Rem>>
{
    #[inline]
    fn from(source: PolyN<In, Out, Rem>) -> Self {
        util::SimplePolyRelation::upgrade_1_poly(source)
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>, A: Vector, B: LinearMap<A>>
    LinearMap<PolyN<In, A, Rem>> for PolyN<In, B, Rem>
{
    type Out = B::Out;
    fn eval_inplace<Accum>(
        &self,
        input: &PolyN<In, A, Rem>,
        accum: &mut Accum,
        f: impl Copy + Fn(&Self::Out, Scalar, &mut Accum),
    ) {
        todo!()
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>, T: MappableBase> MappableBase
    for PolyN<In, T, Rem>
{
    type Map<Out: Vector> = PolyN<In, T::Map<Out>, Rem>;
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>> Mappable for PolyN<In, Scalar, Rem> {
    #[inline]
    fn map_new<Out: Vector>(f: impl Copy + Fn(&Self) -> Out) -> Map<Self, Out> {
        PolyN {
            c_1: In::map_new(|c_1| {
                f(&PolyN {
                    c_1: *c_1,
                    rem: Vector::zero(),
                })
            }),
            rem: Rem::map_new(|rem| {
                In::map_new(|x| {
                    f(&PolyN {
                        c_1: Vector::zero(),
                        rem: Rem::map_linear(rem, |y| *x * *y),
                    })
                })
            }),
        }
    }

    #[inline]
    fn map_linear_inplace<A: Vector, B: Vector>(
        map: &Map<Self, A>,
        target: &mut Map<Self, B>,
        f: impl Copy + Fn(&A, &mut B),
    ) {
        In::map_linear_inplace(&map.c_1, &mut target.c_1, f);
        Rem::map_linear_inplace(&map.rem, &mut target.rem, |a, b| {
            In::map_linear_inplace(a, b, f)
        });
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>> PolyMappable for PolyN<In, Scalar, Rem> {
    const DEGREE: Option<usize> = match Rem::DEGREE {
        Some(n) => Some(n + 1),
        None => None,
    };

    #[inline]
    fn poly_mul_inplace<A: Vector, B: Vector, Out: Vector>(
        a: &PolyN<In, A, Rem>,
        b: &PolyN<In, B, Rem>,
        accum: &mut PolyN<In, Out, Rem>,
        f: impl Copy + Fn(&A, &B, &mut Out),
    ) {
        Rem::map_linear_inplace(b.downgrade_1(), &mut accum.rem, |b, accum| {
            In::map_linear_inplace(&a.c_1, accum, |a, accum| f(a, b, accum))
        });
        Rem::poly_mul_inplace(&a.rem, b.downgrade_1(), &mut accum.rem, |a, b, accum| {
            In::map_linear_inplace(a, accum, |a, accum| f(a, b, accum))
        });
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>> PolyRelation<In> for PolyN<In, Scalar, Rem> {
    #[inline]
    fn poly_eval<Out: Vector>(map: &PolyN<In, Out, Rem>, input: &In) -> Out {
        // TODO: Rewrite to avoid creating intermediate map
        (map.c_1 + Rem::poly_eval(&map.rem, input)).eval(input)
    }

    #[inline]
    fn poly_chain<Other: PolyMappable, Out: Vector>(
        a: &Diff<Other, In>,
        b: &PolyN<In, Out, Rem>,
    ) -> Diff<Other, Out> {
        let mut res = Rem::poly_chain_eval::<Other, Out>(a, &b.rem);
        a.map_linear_inplace(&mut res, |a, r| {
            b.c_1.eval_inplace(a, r, Vector::add_mul_to);
        });
        res
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>> IdentityPolyRelation<In>
    for PolyN<In, Scalar, Rem>
{
    #[inline]
    fn poly_identity() -> PolyN<In, In, Rem> {
        PolyN {
            c_1: In::map_identity(),
            rem: Vector::zero(),
        }
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In>> GradientPolyRelation<In>
    for PolyN<In, Scalar, Rem>
{
    type Gradient = Rem;

    #[inline]
    fn poly_gradient<Out: Vector>(
        poly: &PolyN<In, Out, Rem>,
    ) -> Diff<Self::Gradient, In::Map<Out>> {
        poly.gradient()
    }
}

impl<In: Mappable, Rem: util::SimplePolyRelation<In> + DowngradePolyMappable<Zero>>
    DowngradePolyMappable<Zero> for PolyN<In, Scalar, Rem>
{
    #[inline]
    fn poly_downgrade<Out: Vector>(poly: &Map<Self, Out>) -> ZeroMap<Out> {
        Rem::poly_downgrade::<Out>(poly.downgrade_1())
    }
}

impl<
        In: Mappable,
        Rem: util::SimplePolyRelation<In> + DowngradePolyMappable<ToRem>,
        ToRem: util::SimplePolyRelation<In>,
    > DowngradePolyMappable<PolyN<In, Scalar, ToRem>> for PolyN<In, Scalar, Rem>
{
    #[inline]
    fn poly_downgrade<Out: Vector>(poly: &Map<Self, Out>) -> PolyN<In, Out, ToRem> {
        PolyN {
            c_1: poly.c_1,
            rem: Rem::poly_downgrade::<Map<In, Out>>(&poly.rem),
        }
    }
}
