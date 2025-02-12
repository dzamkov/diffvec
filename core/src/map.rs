use crate::scalar::*;
use crate::vector::*;

/// A [`Vector`] which can be interpreted as a linear map from one vector type to another.
pub trait LinearMap<In: Vector>: Vector {
    /// The type of [`Vector`] produced by this linear map.
    type Out: Vector;

    /// Applies this linear map to the given input.
    #[inline]
    fn eval(&self, input: &In) -> Self::Out {
        let mut res: Self::Out = Vector::zero();
        self.eval_inplace(input, &mut res, Vector::add_mul_to);
        res
    }

    /// In-place version of [`LinearMap::eval`].
    fn eval_inplace<Accum>(
        &self,
        input: &In,
        accum: &mut Accum,
        f: impl Copy + Fn(&Self::Out, Scalar, &mut Accum),
    );
}

/// A [`LinearMap`] type which satisfies certain memory layout requirements.
///
/// # Safety
/// The implementor must ensure that values of the type can safely be interpreted as a
/// `[Self::Out; In::DIM]`s, where each element corresponds to a scalar component of `In`.
pub unsafe trait ContiguousLinearMap<In: ContiguousVector>: LinearMap<In> {
    /// Interprets this map as an array of output vectors, each corresponding to a component of the
    /// input vector.
    #[inline]
    fn columns(&self) -> &[Self::Out] {
        // Safety: it is up to the implementor to ensure that this cast is valid
        unsafe { core::slice::from_raw_parts(&*(self as *const Self as *const Self::Out), In::DIM) }
    }

    /// Interprets this map as a mutable array of output vectors, each corresponding to a component
    /// of the input vector.
    #[inline]
    fn columns_mut(&mut self) -> &mut [Self::Out] {
        // Safety: it is up to the implementor to ensure that this cast is valid
        unsafe {
            core::slice::from_raw_parts_mut(&mut *(self as *mut Self as *mut Self::Out), In::DIM)
        }
    }

    // NOTE: Every `ContiguousLinearMap` whose output type is a `ContiguousVector`
    // should also be a `ContiguousVector`. However, there's currently no way to express this
    // in the type system. Instead, we duplicate the `ContiguousVector` methods here with
    // appropriate bounds on `Self::Out`.

    /// Interprets this vector as an array of scalar components.
    #[inline]
    fn components(&self) -> &[Scalar]
    where
        Self::Out: ContiguousVector,
    {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const Scalar,
                In::DIM * Self::Out::DIM,
            )
        }
    }

    /// Interprets this vector as a mutable array of scalar components.
    #[inline]
    fn components_mut(&mut self) -> &mut [Scalar]
    where
        Self::Out: ContiguousVector,
    {
        unsafe {
            core::slice::from_raw_parts_mut(
                self as *mut Self as *mut Scalar,
                In::DIM * Self::Out::DIM,
            )
        }
    }
}

/// A linear map from `In` to `Out`.
pub type Map<In, Out> = <In as MappableBase>::Map<Out>;

/// The base trait for [`Mappable`]. This trait exists for technical reasons. Don't look into it
/// too much and instead look at [`Mappable`].
pub trait MappableBase: Vector {
    /// The type of [`LinearMap`]s from this type to the given vector type.
    type Map<Out: Vector>: LinearMap<Self, Out = Out>;
}

/// A [`Vector`] type for which there is a comprehensive [`LinearMap`] type from it to every
/// possible output [`Vector`] type.
pub trait Mappable: MappableBase<Map<Scalar> = Self> + LinearMap<Self, Out = Scalar> {
    /// Constructs a new map from the given linear function.
    fn map_new<Out: Vector>(f: impl Copy + Fn(&Self) -> Out) -> Map<Self, Out>;

    /// Applies a linear function to the results of a given map.
    #[inline]
    fn map_linear<A: Vector, B: Vector>(
        map: &Map<Self, A>,
        f: impl Copy + Fn(&A) -> B,
    ) -> Map<Self, B> {
        let mut res = Vector::zero();
        Self::map_linear_inplace(map, &mut res, |a, b| *b = f(a));
        res
    }

    /// Transposes a nested linear map.
    #[inline]
    fn map_transpose<Other: Mappable, Out: Vector>(
        source: &Map<Other, Map<Self, Out>>,
    ) -> Map<Self, Map<Other, Out>> {
        Self::map_new(|x| Other::map_linear::<Map<Self, Out>, Out>(source, |y| y.eval(x)))
    }

    /// Gets the identity map for this vector type.
    #[inline]
    fn map_identity() -> Map<Self, Self> {
        Self::map_new(|x| *x)
    }

    /// Composes two linear maps.
    #[inline]
    fn map_compose<Inter: Vector, Other: LinearMap<Inter>>(
        a: &Map<Self, Inter>,
        b: &Other,
    ) -> Map<Self, Other::Out> {
        Self::map_linear(a, |x| b.eval(x))
    }

    /// In-place version of [`Mappable::map_linear`].
    fn map_linear_inplace<A: Vector, B: Vector>(
        map: &Map<Self, A>,
        accum: &mut Map<Self, B>,
        f: impl Copy + Fn(&A, &mut B),
    );
}

/// A [`Mappable`] whose maps are [`ContiguousLinearMap`]s.
pub trait ContiguousMappable: Mappable + ContiguousVector {}

impl<T: Mappable + ContiguousVector> ContiguousMappable for T {}

/// Applies a linear function to the results of a linear map.
#[inline]
pub fn map_linear<A: Mappable, B: Vector, Out: Vector>(
    map: &Map<A, B>,
    f: impl Copy + Fn(&B) -> Out,
) -> Map<A, Out> {
    A::map_linear(map, f)
}

/// Applies a linear function to the results of a doubly-nested linear map.
#[inline]
pub fn map_linear_2<A: Mappable, B: Mappable, C: Vector, Out: Vector>(
    map: &Map<A, Map<B, C>>,
    f: impl Copy + Fn(&C) -> Out,
) -> Map<A, Map<B, Out>> {
    map_linear::<A, Map<B, C>, Map<B, Out>>(map, |map| map_linear::<B, C, Out>(map, f))
}

/// Transposes a linear map.
#[inline]
pub fn transpose<A: Mappable, B: Mappable, C: Vector>(
    map: &Map<A, Map<B, C>>,
) -> Map<B, Map<A, C>> {
    B::map_transpose::<A, C>(map)
}

/// Transposes a doubly-nested linear map.
#[inline]
pub fn transpose_2<A: Mappable, B: Mappable, C: Mappable, D: Vector>(
    map: &Map<A, Map<B, Map<C, D>>>,
) -> Map<C, Map<A, Map<B, D>>> {
    transpose::<A, C, Map<B, D>>(&map_linear::<A, Map<B, Map<C, D>>, _>(
        map,
        transpose::<B, C, D>,
    ))
}

unsafe impl<In: MappableBase<Map<T::Out> = T> + ContiguousVector, T: LinearMap<In>>
    ContiguousLinearMap<In> for T
{
}
