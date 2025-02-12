use crate::diff;
use crate::map::*;
use crate::poly::PolyMappable;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use serdere::{Deserialize, Serialize};

/// A scalar value.
pub type Scalar = f32;

/// A vector of two [`Scalar`]s.
pub type Vector2 = Vector2Map<Scalar>;

/// Archimedes' constant (π).
pub const PI: Scalar = std::f32::consts::PI;

/// Shortcut for constructing a vector from its components.
#[inline(always)]
pub const fn vec2(x: Scalar, y: Scalar) -> Vector2 {
    Vector2::new(x, y)
}

/// A type which satisfies all the mathematical properties of a vector space.
pub trait Vector:
    Sized
    + diff::Differentiate<Self>
    + Copy
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Scalar>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + Mul<Scalar, Output = Self>
{
    /// The "zero" value for this vector type.
    fn zero() -> Self;

    /// Adds `self * scale` to `output`.
    fn add_mul_to(&self, scale: Scalar, output: &mut Self) {
        *output += *self * scale;
    }
}

/// A [`Vector`] type which satisfies certain memory layout requirements.
///
/// # Safety
/// The implementor must ensure that values of the type can safely be interpreted as a
/// `[Scalar; Self::DIM]`.
pub unsafe trait ContiguousVector: Vector {
    /// The number of [`Scalar`]s needed to represent a value of this type.
    const DIM: usize = std::mem::size_of::<Self>() / std::mem::size_of::<Scalar>();

    /// Interprets this vector as an array of scalar components.
    #[inline]
    fn components(&self) -> &[Scalar] {
        // Safety: it is up to the implementor to ensure that this cast is valid
        unsafe { core::slice::from_raw_parts(self as *const Self as *const Scalar, Self::DIM) }
    }

    /// Interprets this vector as a mutable array of scalar components.
    #[inline]
    fn components_mut(&mut self) -> &mut [Scalar] {
        // Safety: it is up to the implementor to ensure that this cast is valid
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut Scalar, Self::DIM) }
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

impl Vector2 {
    /// Gets a vector that is the same length and perpendicular to this vector, equivalent to
    /// rotating the vector 90 degrees counter-clockwise.
    #[inline]
    pub fn cross(&self) -> Vector2 {
        vec2(-self.y, self.x)
    }

    /// Computes the dot product of the given vectors.
    #[inline]
    pub fn dot(&self, b: &Vector2) -> Scalar {
        self.eval(b)
    }

    /// Gets the square of the L2 norm of this vector.
    #[inline]
    pub fn norm_squared(&self) -> Scalar {
        self.dot(self)
    }

    /// Gets the L2 norm of this vector.
    #[inline]
    pub fn norm(&self) -> Scalar {
        self.norm_squared().sqrt()
    }

    /// The direction of this vector.
    #[inline]
    pub fn normalize(&self) -> Vector2 {
        *self * (1.0 / self.norm())
    }

    /// Gets the signed angle, in radians, between this vector and another, in the range (-π, π].
    /// This will be positive if `other` is to the left (counter-clockwise) from this vector.
    pub fn angle_between(&self, other: &Vector2) -> Scalar {
        let x = self.dot(other);
        let y = self.cross().dot(other);
        y.atan2(x)
    }

    /// Determines whether three vectors, when placed at a common base point, are in
    /// counter-clockwise order. This will return `false` for degenerate cases (colinear and/or
    /// zero vectors).
    pub fn in_ccw_order(a: &Vector2, b: &Vector2, c: &Vector2) -> bool {
        // Consider the positive angles between `(a, b)`, `(b, c)` and `(c, a)`. If these are
        // in counter-clockwise order, their sum will be 360 degrees. If, however, they are in
        // clockwise order, their sum will be 720 degrees. Thus, they are in counter-clockwise
        // order if and only if there is at most one angle that is at least 180 degrees. We
        // can quickly check this using cross products.
        if a.cross().dot(b) > 0.0 {
            b.cross().dot(c) > 0.0 || c.cross().dot(a) > 0.0
        } else {
            b.cross().dot(c) > 0.0 && c.cross().dot(a) > 0.0
        }
    }
}

impl From<[Scalar; 2]> for Vector2 {
    #[inline]
    fn from(source: [Scalar; 2]) -> Self {
        vec2(source[0], source[1])
    }
}

impl From<Vector2> for [Scalar; 2] {
    #[inline]
    fn from(source: Vector2) -> Self {
        [source.x, source.y]
    }
}

impl Mul<Vector2> for Scalar {
    type Output = Vector2;
    #[inline]
    fn mul(self, rhs: Vector2) -> Self::Output {
        vec2(rhs.x * self, rhs.y * self)
    }
}

impl std::ops::Div<Scalar> for Vector2 {
    type Output = Vector2;
    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        vec2(self.x / rhs, self.y / rhs)
    }
}

impl<Out: Vector + approx::AbsDiffEq> approx::AbsDiffEq for Vector2Map<Out>
where
    Out::Epsilon: Copy,
{
    type Epsilon = Out::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        Out::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon) && self.y.abs_diff_eq(&other.y, epsilon)
    }
}

impl<Out: Vector + approx::RelativeEq> approx::RelativeEq for Vector2Map<Out>
where
    Out::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        Out::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
    }
}

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

impl<Out: Vector> diff::Differentiate<ZeroMap<Out>> for ZeroMap<Out> {
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

/// A [`LinearMap`] from a [`Scalar`] to some vector type.
pub type ScalarMap<Out> = Out;

/// A [`LinearMap`] from a [`Vector2`] to some vector type.
#[repr(C)]
#[derive(
    PartialEq,
    Eq,
    Default,
    Clone,
    Copy,
    crate::Differentiate,
    crate::Vector,
    crate::Mappable,
    Serialize,
    Deserialize,
)]
pub struct Vector2Map<Out: Vector> {
    pub x: Out,
    pub y: Out,
}

impl<Out: Vector> Vector2Map<Out> {
    /// Constructs a linear map with the given coefficients for each component.
    #[inline]
    pub const fn new(x: Out, y: Out) -> Self {
        Self { x, y }
    }

    /// Constructs a new [`Vector2Map`] in a differential context.
    #[inline]
    pub fn new_diff<Poly: PolyMappable>(
        x: diff::Diff<Poly, Out>,
        y: diff::Diff<Poly, Out>,
    ) -> diff::Diff<Poly, Self> {
        x.map_linear(|x| Self::new(*x, Out::zero())) + y.map_linear(|y| Self::new(Out::zero(), *y))
    }
}

impl Vector2Map<Vector2> {
    /// Constructs a linear map which returns its input vector unchanged.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            x: vec2(1.0, 0.0),
            y: vec2(0.0, 1.0),
        }
    }

    /// Interpreting this map as a 2x2 matrix, computes the inverse matrix.
    #[inline]
    pub fn inverse(&self) -> Self {
        let det = self.x.x * self.y.y - self.x.y * self.y.x;
        Self {
            x: vec2(self.y.y, -self.x.y) / det,
            y: vec2(-self.y.x, self.x.x) / det,
        }
    }
}

impl Mul<Vector2> for Vector2Map<Vector2> {
    type Output = Vector2;

    #[inline]
    fn mul(self, rhs: Vector2) -> Self::Output {
        self.eval(&rhs)
    }
}

impl Mul<Vector2Map<Vector2>> for Vector2Map<Vector2> {
    type Output = Vector2Map<Vector2>;

    #[inline]
    fn mul(self, rhs: Vector2Map<Vector2>) -> Self::Output {
        Vector2Map {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl<Out: Vector + std::fmt::Debug> std::fmt::Debug for Vector2Map<Out> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("vec2").field(&self.x).field(&self.y).finish()
    }
}

#[test]
fn test_in_ccw_order() {
    assert!(Vector2::in_ccw_order(
        &vec2(1.0, 0.0),
        &vec2(0.0, 1.0),
        &vec2(-1.0, 0.0)
    ));
    assert!(!Vector2::in_ccw_order(
        &vec2(1.0, 0.0),
        &vec2(0.0, -1.0),
        &vec2(-1.0, 0.0)
    ));
    assert!(Vector2::in_ccw_order(
        &vec2(1.0, -1.0),
        &vec2(0.0, 1.0),
        &vec2(-1.0, 1.0)
    ));
    assert!(!Vector2::in_ccw_order(
        &vec2(1.0, 0.0),
        &vec2(1.0, 0.0),
        &vec2(0.0, 1.0)
    ));
}
