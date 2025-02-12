use crate::diff::*;
use crate::map::*;
use crate::poly::*;
use crate::scalar::*;
use crate::vector::*;

/// Shortcut for constructing a vector from its components.
#[inline(always)]
pub const fn vec4(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Vector4 {
    Vector4::new(x, y, z, w)
}

/// A vector of four [`Scalar`]s.
pub type Vector4 = Vector4Map<Scalar>;

impl Vector4 {
    /// Computes the dot product of the given vectors.
    #[inline]
    pub fn dot(&self, b: &Vector4) -> Scalar {
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
    pub fn normalize(&self) -> Vector4 {
        *self * (1.0 / self.norm())
    }
}

impl From<[Scalar; 4]> for Vector4 {
    #[inline]
    fn from(source: [Scalar; 4]) -> Self {
        vec4(source[0], source[1], source[2], source[3])
    }
}

impl From<Vector4> for [Scalar; 4] {
    #[inline]
    fn from(source: Vector4) -> Self {
        [source.x, source.y, source.z, source.w]
    }
}

impl core::ops::Mul<Vector4> for Scalar {
    type Output = Vector4;
    #[inline]
    fn mul(self, rhs: Vector4) -> Self::Output {
        vec4(rhs.x * self, rhs.y * self, rhs.z * self, rhs.w * self)
    }
}

impl core::ops::Div<Scalar> for Vector4 {
    type Output = Vector4;
    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        vec4(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl<Out: Vector + approx::AbsDiffEq> approx::AbsDiffEq for Vector4Map<Out>
where
    Out::Epsilon: Copy,
{
    type Epsilon = Out::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        Out::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
            && self.w.abs_diff_eq(&other.w, epsilon)
    }
}

impl<Out: Vector + approx::RelativeEq> approx::RelativeEq for Vector4Map<Out>
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
            && self.z.relative_eq(&other.z, epsilon, max_relative)
            && self.w.relative_eq(&other.w, epsilon, max_relative)
    }
}

/// A 4-by-4 matrix, i.e. a linear map from a [`Vector4`] to a [`Vector4`].
pub type Matrix4 = Vector4Map<Vector4>;

impl Matrix4 {
    /// The identity [`Matrix4`].
    #[inline]
    pub const fn identity() -> Self {
        Self {
            x: vec4(1.0, 0.0, 0.0, 0.0),
            y: vec4(0.0, 1.0, 0.0, 0.0),
            z: vec4(0.0, 0.0, 1.0, 0.0),
            w: vec4(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Computes the transpose of this [`Matrix4`].
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            x: vec4(self.x.x, self.y.x, self.z.x, self.w.x),
            y: vec4(self.x.y, self.y.y, self.z.y, self.w.y),
            z: vec4(self.x.z, self.y.z, self.z.z, self.w.z),
            w: vec4(self.x.w, self.y.w, self.z.w, self.w.w),
        }
    }
}

impl core::ops::Mul<Vector4> for Matrix4 {
    type Output = Vector4;

    #[inline]
    fn mul(self, rhs: Vector4) -> Self::Output {
        self.eval(&rhs)
    }
}

impl core::ops::Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    #[inline]
    fn mul(self, rhs: Matrix4) -> Self::Output {
        Matrix4 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}

/// A [`LinearMap`] from a [`Vector4`] to some vector type.
#[repr(C)]
#[derive(
    PartialEq, Eq, Default, Clone, Copy, crate::Differentiate, crate::Vector, crate::Mappable,
)]
#[cfg_attr(feature = "serdere", derive(serdere::Serialize, serdere::Deserialize))]
pub struct Vector4Map<Out: Vector> {
    pub x: Out,
    pub y: Out,
    pub z: Out,
    pub w: Out,
}

impl<Out: Vector> Vector4Map<Out> {
    /// Constructs a linear map with the given coefficients for each component.
    #[inline]
    pub const fn new(x: Out, y: Out, z: Out, w: Out) -> Self {
        Self { x, y, z, w }
    }

    /// Constructs a new [`Vector4Map`] in a differential context.
    #[inline]
    pub fn new_diff<Poly: PolyMappable>(
        x: Diff<Poly, Out>,
        y: Diff<Poly, Out>,
        z: Diff<Poly, Out>,
        w: Diff<Poly, Out>,
    ) -> Diff<Poly, Self> {
        x.map_linear(|x| Self::new(*x, Out::zero(), Out::zero(), Out::zero()))
            + y.map_linear(|y| Self::new(Out::zero(), *y, Out::zero(), Out::zero()))
            + z.map_linear(|z| Self::new(Out::zero(), Out::zero(), *z, Out::zero()))
            + w.map_linear(|w| Self::new(Out::zero(), Out::zero(), Out::zero(), *w))
    }
}

impl<Out: Vector + std::fmt::Debug> std::fmt::Debug for Vector4Map<Out> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("vec4")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<Out: Vector + bytemuck::Zeroable> bytemuck::Zeroable for Vector4Map<Out> {}

#[cfg(feature = "bytemuck")]
unsafe impl<Out: ContiguousVector + bytemuck::Zeroable + 'static> bytemuck::Pod
    for Vector4Map<Out>
{
}

impl<Poly: PolyMappable> Diff<Poly, Vector4> {
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

    /// Gets the Z component of this vector.
    #[inline]
    pub fn z(&self) -> Diff<Poly, Scalar> {
        self.map_linear(|vec| vec.z)
    }

    /// Gets the W component of this vector.
    #[inline]
    pub fn w(&self) -> Diff<Poly, Scalar> {
        self.map_linear(|vec| vec.w)
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
        self.dot(self)
    }

    /// The L2 norm of this vector.
    #[inline]
    pub fn norm(&self) -> Diff<Poly, Scalar> {
        self.norm_squared().sqrt()
    }

    /// The direction of this vector.
    #[inline]
    pub fn normalize(&self) -> Diff<Poly, Vector4> {
        *self * (1.0 / self.norm())
    }
}