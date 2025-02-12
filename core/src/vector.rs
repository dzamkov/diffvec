use crate::diff;
use crate::Scalar;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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