#![allow(clippy::type_complexity)]
use crate::diff::{Diff, Differentiate};
use crate::map::*;
use crate::poly::PolyMappable;
use crate::vector::*;

/// A product type consisting of `A`, `B`, which can be interpreted as a product vector
/// space.
#[repr(C)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, crate::Vector, crate::Mappable)]
pub struct Prod2<A, B>(pub A, pub B);

impl<Poly: PolyMappable, DA: Vector, DB: Vector, A: Differentiate<DA>, B: Differentiate<DB>>
    Diff<Poly, Prod2<A, B>, Prod2<DA, DB>>
{
    /// Joins two components into a product type.
    #[inline]
    pub fn join_2(
        a: &Diff<Poly, A, DA>,
        b: &Diff<Poly, B, DB>,
    ) -> Diff<Poly, Prod2<A, B>, Prod2<DA, DB>>
    where
        A: Clone,
        B: Clone,
    {
        let mut poly = Vector::zero();
        Poly::map_linear_inplace::<DA, Prod2<DA, DB>>(&a.poly, &mut poly, |x, y| y.0 = *x);
        Poly::map_linear_inplace::<DB, Prod2<DA, DB>>(&b.poly, &mut poly, |x, y| y.1 = *x);
        Diff {
            value: Prod2(a.value.clone(), b.value.clone()),
            poly,
        }
    }

    /// Splits a product type into its constituent components.
    #[inline]
    pub fn split(self) -> (Diff<Poly, A, DA>, Diff<Poly, B, DB>) {
        (
            Diff {
                value: self.value.0,
                poly: map_linear::<Poly, Prod2<DA, DB>, DA>(&self.poly, |x| x.0),
            },
            Diff {
                value: self.value.1,
                poly: map_linear::<Poly, Prod2<DA, DB>, DB>(&self.poly, |x| x.1),
            },
        )
    }
}

impl<DA: Vector, DB: Vector, A: Differentiate<DA>, B: Differentiate<DB>>
    Differentiate<Prod2<DA, DB>> for Prod2<A, B>
{
    #[inline]
    fn perturb(&mut self, amount: &Prod2<DA, DB>) {
        self.0.perturb(&amount.0);
        self.1.perturb(&amount.1);
    }
}

/// A product type consisting of `A`, `B` and `C`, which can be interpreted as a product vector
/// space.
#[repr(C)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, crate::Vector, crate::Mappable)]
pub struct Prod3<A, B, C>(pub A, pub B, pub C);

impl<
        Poly: PolyMappable,
        DA: Vector,
        DB: Vector,
        DC: Vector,
        A: Differentiate<DA>,
        B: Differentiate<DB>,
        C: Differentiate<DC>,
    > Diff<Poly, Prod3<A, B, C>, Prod3<DA, DB, DC>>
{
    /// Splits a product type into its constituent components.
    #[inline]
    pub fn split(self) -> (Diff<Poly, A, DA>, Diff<Poly, B, DB>, Diff<Poly, C, DC>) {
        (
            Diff {
                value: self.value.0,
                poly: map_linear::<Poly, Prod3<DA, DB, DC>, DA>(&self.poly, |x| x.0),
            },
            Diff {
                value: self.value.1,
                poly: map_linear::<Poly, Prod3<DA, DB, DC>, DB>(&self.poly, |x| x.1),
            },
            Diff {
                value: self.value.2,
                poly: map_linear::<Poly, Prod3<DA, DB, DC>, DC>(&self.poly, |x| x.2),
            },
        )
    }
}

impl<
        DA: Vector,
        DB: Vector,
        DC: Vector,
        A: Differentiate<DA>,
        B: Differentiate<DB>,
        C: Differentiate<DC>,
    > Differentiate<Prod3<DA, DB, DC>> for Prod3<A, B, C>
{
    #[inline]
    fn perturb(&mut self, amount: &Prod3<DA, DB, DC>) {
        self.0.perturb(&amount.0);
        self.1.perturb(&amount.1);
        self.2.perturb(&amount.2);
    }
}