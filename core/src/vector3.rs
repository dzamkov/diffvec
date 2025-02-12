use crate::diff::*;
use crate::map::*;
use crate::poly::*;
use crate::scalar::*;
use crate::vector::*;

/// Shortcut for constructing a vector from its components.
#[inline(always)]
pub const fn vec3(x: Scalar, y: Scalar, z: Scalar) -> Vector3 {
    Vector3::new(x, y, z)
}

/// A vector of three [`Scalar`]s.
pub type Vector3 = Vector3Map<Scalar>;

impl Vector3 {
    /// Computes the dot product of the given vectors.
    #[inline]
    pub fn dot(&self, b: &Vector3) -> Scalar {
        self.eval(b)
    }

    /// Computes the cross product of the given vectors.
    #[inline]
    pub fn cross(&self, b: &Vector3) -> Vector3 {
        vec3(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
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
    pub fn normalize(&self) -> Vector3 {
        *self * (1.0 / self.norm())
    }
}

impl From<[Scalar; 3]> for Vector3 {
    #[inline]
    fn from(source: [Scalar; 3]) -> Self {
        vec3(source[0], source[1], source[2])
    }
}

impl From<Vector3> for [Scalar; 3] {
    #[inline]
    fn from(source: Vector3) -> Self {
        [source.x, source.y, source.z]
    }
}

impl core::ops::Mul<Vector3> for Scalar {
    type Output = Vector3;
    #[inline]
    fn mul(self, rhs: Vector3) -> Self::Output {
        vec3(rhs.x * self, rhs.y * self, rhs.z * self)
    }
}

impl core::ops::Div<Scalar> for Vector3 {
    type Output = Vector3;
    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        vec3(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<Out: Vector + approx::AbsDiffEq> approx::AbsDiffEq for Vector3Map<Out>
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
    }
}

impl<Out: Vector + approx::RelativeEq> approx::RelativeEq for Vector3Map<Out>
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
    }
}

/// A 3-by-3 matrix, i.e. a linear map from a [`Vector3`] to a [`Vector3`].
pub type Matrix3 = Vector3Map<Vector3>;

impl Matrix3 {
    /// The identity [`Matrix3`].
    #[inline]
    pub const fn identity() -> Self {
        Self {
            x: vec3(1.0, 0.0, 0.0),
            y: vec3(0.0, 1.0, 0.0),
            z: vec3(0.0, 0.0, 1.0),
        }
    }

    /// Computes the transpose of this [`Matrix3`].
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            x: vec3(self.x.x, self.y.x, self.z.x),
            y: vec3(self.x.y, self.y.y, self.z.y),
            z: vec3(self.x.z, self.y.z, self.z.z),
        }
    }

    /// Computes the cofactor matrix of this [`Matrix3`].
    #[inline]
    pub fn cofactor(&self) -> Self {
        Self {
            x: vec3(
                self.y.y * self.z.z - self.z.y * self.y.z,
                self.y.z * self.z.x - self.z.z * self.y.x,
                self.y.x * self.z.y - self.z.x * self.y.y,
            ),
            y: vec3(
                self.z.y * self.x.z - self.x.y * self.z.z,
                self.z.z * self.x.x - self.x.z * self.z.x,
                self.z.x * self.x.y - self.x.x * self.z.y,
            ),
            z: vec3(
                self.x.y * self.y.z - self.y.y * self.x.z,
                self.x.z * self.y.x - self.y.z * self.x.x,
                self.x.x * self.y.y - self.y.x * self.x.y,
            ),
        }
    }

    /// Computes the inverse of this [`Matrix3`].
    #[inline]
    pub fn inverse(&self) -> Self {
        let cofactor = self.cofactor();
        let det = self.x.dot(&cofactor.x);
        cofactor.transpose() * (1.0 / det)
    }
}

impl core::ops::Mul<Vector3> for Matrix3 {
    type Output = Vector3;

    #[inline]
    fn mul(self, rhs: Vector3) -> Self::Output {
        self.eval(&rhs)
    }
}

impl core::ops::Mul<Matrix3> for Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn mul(self, rhs: Matrix3) -> Self::Output {
        Matrix3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

#[test]
fn test_inverse() {
    let test = Matrix3 {
        x: vec3(9.0, 5.0, 2.0),
        y: vec3(3.0, 6.0, 7.0),
        z: vec3(4.0, 8.0, 1.0),
    };
    let inv = test.inverse();
    approx::assert_abs_diff_eq!(inv * test, Matrix3::identity(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(inv * (test * test), test, epsilon = 1e-6);
}

/// A [`LinearMap`] from a [`Vector3`] to some vector type.
#[repr(C)]
#[derive(
    PartialEq, Eq, Default, Clone, Copy, crate::Differentiate, crate::Vector, crate::Mappable,
)]
#[cfg_attr(feature = "serdere", derive(serdere::Serialize, serdere::Deserialize))]
pub struct Vector3Map<Out: Vector> {
    pub x: Out,
    pub y: Out,
    pub z: Out,
}

impl<Out: Vector> Vector3Map<Out> {
    /// Constructs a linear map with the given coefficients for each component.
    #[inline]
    pub const fn new(x: Out, y: Out, z: Out) -> Self {
        Self { x, y, z }
    }

    /// Constructs a new [`Vector3Map`] in a differential context.
    #[inline]
    pub fn new_diff<Poly: PolyMappable>(
        x: Diff<Poly, Out>,
        y: Diff<Poly, Out>,
        z: Diff<Poly, Out>,
    ) -> Diff<Poly, Self> {
        x.map_linear(|x| Self::new(*x, Out::zero(), Out::zero()))
            + y.map_linear(|y| Self::new(Out::zero(), *y, Out::zero()))
            + z.map_linear(|z| Self::new(Out::zero(), Out::zero(), *z))
    }
}

impl<Out: Vector + std::fmt::Debug> std::fmt::Debug for Vector3Map<Out> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("vec3")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<Out: Vector + bytemuck::Zeroable> bytemuck::Zeroable for Vector3Map<Out> {}

#[cfg(feature = "bytemuck")]
unsafe impl<Out: ContiguousVector + bytemuck::Zeroable + 'static> bytemuck::Pod
    for Vector3Map<Out>
{
}

impl<Poly: PolyMappable> Diff<Poly, Vector3> {
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

    /// Computes the dot product of two vectors.
    #[inline]
    pub fn dot(&self, other: &Self) -> Diff<Poly, Scalar> {
        let mut res = Diff::zero();
        self.mul_inplace(other, &mut res, |x, y, accum| *accum += x.dot(y));
        res
    }

    /// Computes the cross product of two vectors.
    #[inline]
    pub fn cross(&self, other: &Self) -> Diff<Poly, Vector3> {
        let mut res = Diff::zero();
        self.mul_inplace(other, &mut res, |x, y, accum| *accum += x.cross(y));
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
    pub fn normalize(&self) -> Diff<Poly, Vector3> {
        *self * (1.0 / self.norm())
    }
}
