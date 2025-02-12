use crate::diff::*;
use crate::map::*;
use crate::poly::*;
use crate::scalar::*;
use crate::vector::*;

/// Shortcut for constructing a vector from its components.
#[inline(always)]
pub const fn vec2(x: Scalar, y: Scalar) -> Vector2 {
    Vector2::new(x, y)
}

/// A vector of two [`Scalar`]s.
pub type Vector2 = Vector2Map<Scalar>;

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

impl core::ops::Mul<Vector2> for Scalar {
    type Output = Vector2;
    #[inline]
    fn mul(self, rhs: Vector2) -> Self::Output {
        vec2(rhs.x * self, rhs.y * self)
    }
}

impl core::ops::Div<Scalar> for Vector2 {
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

/// A 2-by-2 matrix, i.e. a linear map from a [`Vector2`] to a [`Vector2`].
pub type Matrix2 = Vector2Map<Vector2>;

impl Matrix2 {
    /// The identity [`Matrix2`].
    #[inline]
    pub const fn identity() -> Self {
        Self {
            x: vec2(1.0, 0.0),
            y: vec2(0.0, 1.0),
        }
    }

    /// Computes the inverse of this [`Matrix2`].
    #[inline]
    pub fn inverse(&self) -> Self {
        let det = self.x.x * self.y.y - self.x.y * self.y.x;
        Self {
            x: vec2(self.y.y, -self.x.y) / det,
            y: vec2(-self.y.x, self.x.x) / det,
        }
    }
}

impl core::ops::Mul<Vector2> for Matrix2 {
    type Output = Vector2;

    #[inline]
    fn mul(self, rhs: Vector2) -> Self::Output {
        self.eval(&rhs)
    }
}

impl core::ops::Mul<Matrix2> for Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn mul(self, rhs: Matrix2) -> Self::Output {
        Matrix2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

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
    serdere::Serialize,
    serdere::Deserialize,
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
        x: Diff<Poly, Out>,
        y: Diff<Poly, Out>,
    ) -> Diff<Poly, Self> {
        x.map_linear(|x| Self::new(*x, Out::zero())) + y.map_linear(|y| Self::new(Out::zero(), *y))
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
