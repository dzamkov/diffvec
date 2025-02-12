use crate::vector::*;
use crate::scalar::*;
use crate::map::*;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A zero-sized scalar whose value is always 0.
pub type Zero = ZeroMap<Scalar>;

/// A [`LinearMap`] from [`Zero`] to some vector type.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct ZeroMap<Out: Vector>([Out; 0]);

impl<Out: Vector> ZeroMap<Out> {
    /// Constructs a new [`ZeroMap`].
    #[inline]
    pub const fn new() -> Self {
        ZeroMap([])
    }
}

impl<Out: Vector> Default for ZeroMap<Out> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<Out: Vector> AddAssign<Self> for ZeroMap<Out> {
    #[inline]
    fn add_assign(&mut self, _: Self) {
        // Nothing to do here
    }
}

impl<Out: Vector> MulAssign<Scalar> for ZeroMap<Out> {
    #[inline]
    fn mul_assign(&mut self, _: Scalar) {
        // Nothing to do here
    }
}

impl<Out: Vector> SubAssign<Self> for ZeroMap<Out> {
    #[inline]
    fn sub_assign(&mut self, _: Self) {
        // Nothing to do here
    }
}

impl<Out: Vector> Add<Self> for ZeroMap<Out> {
    type Output = Self;

    #[inline]
    fn add(self, _: Self) -> Self::Output {
        self
    }
}

impl<Out: Vector> Sub<Self> for ZeroMap<Out> {
    type Output = Self;

    #[inline]
    fn sub(self, _: Self) -> Self::Output {
        self
    }
}

impl<Out: Vector> Neg for ZeroMap<Out> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

impl<Out: Vector> Mul<Scalar> for ZeroMap<Out> {
    type Output = Self;

    #[inline]
    fn mul(self, _: Scalar) -> Self::Output {
        self
    }
}

impl<Out: Vector> crate::diff::Differentiate<ZeroMap<Out>> for ZeroMap<Out> {
    #[inline]
    fn perturb(&mut self, _: &ZeroMap<Out>) {
        // Nothing to do here
    }
}

impl<Out: Vector> Vector for ZeroMap<Out> {
    #[inline]
    fn zero() -> Self {
        Self([])
    }

    #[inline]
    fn add_mul_to(&self, _: Scalar, _: &mut Self) {
        // Nothing to do here
    }
}

impl<A: LinearMap<B>, B: Vector> LinearMap<ZeroMap<B>> for ZeroMap<A> {
    type Out = A::Out;

    #[inline]
    fn eval_inplace<Accum>(
        &self,
        _: &ZeroMap<B>,
        _: &mut Accum,
        _: impl Copy + Fn(&Self::Out, Scalar, &mut Accum),
    ) {
        // Nothing to do here
    }
}

unsafe impl<Out: Vector> ContiguousVector for ZeroMap<Out> {
    const DIM: usize = 0;
}

impl MappableBase for Zero {
    type Map<Out: Vector> = ZeroMap<Out>;
}

impl Mappable for Zero {
    #[inline]
    fn map_new<Out: Vector>(_: impl Copy + Fn(&Self) -> Out) -> Map<Self, Out> {
        ZeroMap::new()
    }

    #[inline]
    fn map_transpose<Other: Mappable, Out: Vector>(
        _: &Map<Other, Map<Self, Out>>,
    ) -> Map<Self, Map<Other, Out>> {
        ZeroMap::new()
    }

    #[inline]
    fn map_linear_inplace<A: Vector, B: Vector>(
        _: &Map<Self, A>,
        _: &mut Map<Self, B>,
        _: impl Copy + Fn(&A, &mut B),
    ) {
        // Nothing to do here
    }
}