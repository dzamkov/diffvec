use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{quote, ToTokens, TokenStreamExt};
use std::collections::{HashMap, HashSet};
use std::iter::zip;

#[proc_macro_derive(Differentiate)]
pub fn derive_differentiate(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    expand_differentiate(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(Vector)]
pub fn derive_vector(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    expand_vector(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(Mappable)]
pub fn derive_mappable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    expand_mappable(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Generates a `Differentiate` implementation.
fn expand_differentiate(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let diffvec = quote! { ::diffvec };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let syn::Data::Struct(syn::DataStruct { fields, .. }) = input.data else {
        return Err(syn::Error::new(Span::call_site(), "expected a struct"));
    };
    let ident = input.ident;
    let field_tys = fields.iter().map(|f| &f.ty).collect::<HashSet<_>>();
    let where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: #diffvec::Vector }).unwrap()),
    );
    let body = fields
        .members()
        .map(|member| {
            quote! { self.#member.perturb_mut(&rhs.#member); }
        })
        .collect::<TokenStream>();
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #diffvec::Differentiate<Self> for #ident #ty_generics #where_clause {
            fn perturb_mut(&mut self, rhs: &Self) {
                #body
            }
        }
    })
}

/// Generates a `Vector` implementation.
fn expand_vector(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let diffvec = quote! { ::diffvec };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let syn::Data::Struct(syn::DataStruct { fields, .. }) = &input.data else {
        return Err(syn::Error::new(Span::call_site(), "expected a struct"));
    };
    let ident = input.ident.clone();
    let field_tys = fields.iter().map(|f| &f.ty).collect::<HashSet<_>>();
    let add_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: ::core::ops::Add<#ty, Output = #ty> }).unwrap()),
    );
    let add_body = build_struct(&ident, fields, |_, member| {
        quote! { self.#member + rhs.#member }
    });
    let sub_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: ::core::ops::Sub<#ty, Output = #ty> }).unwrap()),
    );
    let sub_body = build_struct(&ident, fields, |_, member| {
        quote! { self.#member - rhs.#member }
    });
    let neg_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: ::core::ops::Neg<Output = #ty> }).unwrap()),
    );
    let neg_body = build_struct(&ident, fields, |_, member| {
        quote! { -self.#member }
    });
    let mul_where_clause = add_predicates(
        where_clause,
        field_tys.iter().map(|ty| {
            syn::parse2(quote! { #ty: ::core::ops::Mul<#diffvec::Scalar, Output = #ty> }).unwrap()
        }),
    );
    let mul_body = build_struct(&ident, fields, |_, member| {
        quote! { self.#member * rhs }
    });
    let add_assign_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: ::core::ops::AddAssign<#ty> }).unwrap()),
    );
    let add_assign_body = fields
        .members()
        .map(|member| {
            quote! { self.#member += rhs.#member; }
        })
        .collect::<TokenStream>();
    let sub_assign_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: ::core::ops::SubAssign<#ty> }).unwrap()),
    );
    let sub_assign_body = fields
        .members()
        .map(|member| {
            quote! { self.#member -= rhs.#member; }
        })
        .collect::<TokenStream>();
    let mul_assign_where_clause = add_predicates(
        where_clause,
        field_tys.iter().map(|ty| {
            syn::parse2(quote! { #ty: ::core::ops::MulAssign<#diffvec::Scalar> }).unwrap()
        }),
    );
    let mul_assign_body = fields
        .members()
        .map(|member| {
            quote! { self.#member *= rhs; }
        })
        .collect::<TokenStream>();
    let vector_where_clause = add_predicates(
        where_clause,
        field_tys
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: #diffvec::Vector }).unwrap()),
    );
    let zero_body = build_struct(&ident, fields, |_, _| quote! { #diffvec::Vector::zero() });
    let add_mul_to_body = fields
        .members()
        .map(|member| {
            quote! { self.#member.add_mul_to(scale, &mut output.#member); }
        })
        .collect::<TokenStream>();
    let contiguous_vector_impl = if is_possibly_contiguous(&input) {
        let contiguous_vector_where_clause = add_predicates(
            where_clause,
            field_tys
                .iter()
                .map(|ty| syn::parse2(quote! { #ty: #diffvec::ContiguousVector }).unwrap()),
        );
        Some(quote! {
            #[automatically_derived]
            unsafe impl #impl_generics #diffvec::ContiguousVector
                for #ident #ty_generics #contiguous_vector_where_clause
            {}
        })
    } else {
        None
    };
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::core::ops::Add<Self> for #ident #ty_generics #add_where_clause {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                #add_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::Sub<Self> for #ident #ty_generics #sub_where_clause {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                #sub_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::Neg for #ident #ty_generics #neg_where_clause {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                #neg_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::Mul<#diffvec::Scalar>
            for #ident #ty_generics #mul_where_clause
        {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: #diffvec::Scalar) -> Self {
                #mul_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::AddAssign<Self>
            for #ident #ty_generics #add_assign_where_clause
        {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                #add_assign_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::SubAssign<Self>
            for #ident #ty_generics #sub_assign_where_clause
        {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                #sub_assign_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::ops::MulAssign<#diffvec::Scalar>
            for #ident #ty_generics #mul_assign_where_clause
        {
            #[inline]
            fn mul_assign(&mut self, rhs: #diffvec::Scalar) {
                #mul_assign_body
            }
        }

        #[automatically_derived]
        impl #impl_generics #diffvec::Vector for #ident #ty_generics #vector_where_clause {
            #[inline]
            fn zero() -> Self {
                #zero_body
            }

            #[inline]
            fn add_mul_to(&self, scale: #diffvec::Scalar, output: &mut Self) {
                #add_mul_to_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::core::iter::Sum for #ident #ty_generics #vector_where_clause {
            #[inline]
            fn sum<I: ::core::iter::IntoIterator<Item = Self>>(iter: I) -> Self {
                let mut res = <Self as #diffvec::Vector>::zero();
                for item in iter {
                    res += item;
                }
                res
            }
        }

        #contiguous_vector_impl
    })
}

/// Generates a `Mappable` implementation.
fn expand_mappable(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let diffvec = quote! { ::diffvec };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let syn::Data::Struct(syn::DataStruct { fields, .. }) = &input.data else {
        return Err(syn::Error::new(Span::call_site(), "expected a struct"));
    };
    let ident = input.ident;
    let field_tys = fields.iter().map(|f| &f.ty).collect::<HashSet<_>>();
    let all_generics = input
        .generics
        .type_params()
        .map(|param| &param.ident)
        .collect::<HashSet<_>>();
    let mut idents =
        IdentAllocator::new(all_generics.iter().map(|ident| ident.to_string()).collect());
    let mut mapped_generics = HashSet::new();
    for field_ty in &field_tys {
        get_mapped_generics(&all_generics, &mut mapped_generics, field_ty);
    }
    let other_ty = idents.alloc("Other".to_string());
    let out_ty = idents.alloc("Out".to_string());
    let accum_ty = idents.alloc("Accum".to_string());
    let a_ty = idents.alloc("A".to_string());
    let b_ty = idents.alloc("B".to_string());
    let mut alt_generics = HashMap::new();
    let mut linear_map_generic_params: Vec<syn::GenericParam> = Vec::new();
    let mut linear_map_add_preds = Vec::new();
    linear_map_generic_params.extend(input.generics.lifetimes().map(|p| p.clone().into()));
    for ty_param in input.generics.type_params() {
        linear_map_generic_params.push(ty_param.clone().into());
        if mapped_generics.contains(&ty_param.ident) {
            let mut alt_ty_param = ty_param.clone();
            alt_ty_param.ident = idents.alloc(ty_param.ident.to_string());
            let ty_ident = &ty_param.ident;
            let alt_ty_ident = &alt_ty_param.ident;
            linear_map_add_preds.push(
                syn::parse2(quote! {
                    #ty_ident: #diffvec::Vector + #diffvec::LinearMap<#alt_ty_ident, Out = #out_ty>
                })
                .unwrap(),
            );
            linear_map_add_preds.push(
                syn::parse2(quote! {
                    #alt_ty_ident: #diffvec::Vector
                })
                .unwrap(),
            );
            alt_generics.insert(&ty_param.ident, alt_ty_param.ident.clone());
            linear_map_generic_params.push(alt_ty_param.into());
        }
    }
    linear_map_generic_params.push(syn::parse2(quote! { #out_ty: #diffvec::Vector }).unwrap());
    linear_map_generic_params.extend(input.generics.const_params().map(|p| p.clone().into()));
    let linear_map_generic_params: syn::punctuated::Punctuated<_, syn::Token![,]> =
        linear_map_generic_params.into_iter().collect();
    let alt_ty_generics = substitute(ty_generics.to_token_stream(), |dest, ident| {
        if let Some(alt_ident) = alt_generics.get(ident) {
            dest.append(TokenTree::Ident(alt_ident.clone()));
        } else {
            dest.append(TokenTree::Ident(ident.clone()));
        }
    });
    let linear_map_where_clause = add_predicates(where_clause, linear_map_add_preds.into_iter());
    let eval_inplace_body = zip(fields.iter().map(|field| &field.ty), fields.members())
        .map(|(field_ty, member)| {
            quote! {
                <#field_ty as #diffvec::LinearMap<_>>::eval_inplace(
                    &self.#member,
                    &input.#member,
                    accum,
                    f
                );
            }
        })
        .collect::<TokenStream>();
    let mappable_base_where_clause = add_predicates(
        where_clause,
        mapped_generics
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: #diffvec::MappableBase }).unwrap()),
    );
    let mapped_ty_generics = substitute(ty_generics.to_token_stream(), |dest, ident| {
        if mapped_generics.contains(&ident) {
            dest.append_all(quote! { <#ident as #diffvec::MappableBase>::Map<#out_ty> });
        } else {
            dest.append(TokenTree::Ident(ident.clone()));
        }
    });
    let mappable_where_clause = add_predicates(
        where_clause,
        mapped_generics
            .iter()
            .map(|ty| syn::parse2(quote! { #ty: #diffvec::Mappable }).unwrap()),
    );
    let map_new_body = build_struct(&ident, fields, |_, member| {
        let inner = build_struct(&ident, fields, |_, inner_member| {
            if member == inner_member {
                quote! { *a }
            } else {
                quote! { #diffvec::Vector::zero() }
            }
        });
        quote! { #diffvec::Mappable::map_new(|a| f(&#inner)) }
    });
    let map_transpose_body = build_struct(&ident, fields, |field_ty, member| {
        quote! {
            <#field_ty as #diffvec::Mappable>::map_transpose::<#other_ty, #out_ty>(
                &#other_ty::map_linear::<
                    <Self as #diffvec::MappableBase>::Map<#out_ty>,
                    <#field_ty as #diffvec::MappableBase>::Map<#out_ty>
                >(source, |a| a.#member)
            )
        }
    });
    let map_linear_inplace_body = zip(fields.iter().map(|field| &field.ty), fields.members())
        .map(|(field_ty, member)| {
            quote! {
                <#field_ty as #diffvec::Mappable>::map_linear_inplace(
                    &map.#member,
                    &mut accum.#member,
                    f
                );
            }
        })
        .collect::<TokenStream>();
    Ok(quote! {
        #[automatically_derived]
        impl <#linear_map_generic_params> #diffvec::LinearMap<#ident #alt_ty_generics>
            for #ident #ty_generics #linear_map_where_clause
        {
            type Out = #out_ty;

            #[inline]
            fn eval_inplace<#accum_ty>(
                &self,
                input: &#ident #alt_ty_generics,
                accum: &mut #accum_ty,
                f: impl ::core::marker::Copy + Fn(&Self::Out, #diffvec::Scalar, &mut #accum_ty),
            ) {
                #eval_inplace_body
            }
        }

        #[automatically_derived]
        impl #impl_generics #diffvec::MappableBase
            for #ident #ty_generics #mappable_base_where_clause
        {
            type Map<#out_ty: #diffvec::Vector> = #ident #mapped_ty_generics;
        }

        #[automatically_derived]
        impl #impl_generics #diffvec::Mappable for #ident #ty_generics #mappable_where_clause {
            #[inline]
            fn map_new<#out_ty: #diffvec::Vector>(
                f: impl Copy + Fn(&Self) -> #out_ty
            ) -> #diffvec::Map<Self, #out_ty> {
                #map_new_body
            }

            #[inline]
            fn map_transpose<#other_ty: #diffvec::Mappable, #out_ty: #diffvec::Vector>(
                source: &#diffvec::Map<#other_ty, #diffvec::Map<Self, #out_ty>>,
            ) -> #diffvec::Map<Self, #diffvec::Map<#other_ty, #out_ty>> {
                #map_transpose_body
            }

            #[inline]
            fn map_linear_inplace<#a_ty: #diffvec::Vector, #b_ty: #diffvec::Vector>(
                map: &#diffvec::Map<Self, #a_ty>,
                accum: &mut #diffvec::Map<Self, #b_ty>,
                f: impl Copy + Fn(&#a_ty, &mut #b_ty),
            ) {
                #map_linear_inplace_body
            }
        }
    })
}

/// Adds predicates to a `where` clause.
fn add_predicates(
    where_clause: Option<&syn::WhereClause>,
    preds: impl Iterator<Item = syn::WherePredicate>,
) -> Option<syn::WhereClause> {
    match where_clause {
        Some(where_clause) => {
            let mut where_clause = where_clause.clone();
            where_clause.predicates.extend(preds);
            Some(where_clause)
        }
        None => Some(syn::WhereClause {
            where_token: syn::Token![where](Span::call_site()),
            predicates: preds.collect(),
        }),
    }
}

/// Generates code for building an instance of a struct.
fn build_struct(
    ident: &syn::Ident,
    fields: &syn::Fields,
    f: impl Fn(&syn::Type, &syn::Member) -> TokenStream,
) -> TokenStream {
    match fields {
        syn::Fields::Named(syn::FieldsNamed { named, .. }) => {
            let data = named.iter().map(|field| {
                let ident = field.ident.as_ref().unwrap();
                let value = f(&field.ty, &syn::Member::Named(ident.clone()));
                quote! { #ident: #value }
            });
            quote! { #ident { #(#data,)* } }
        }
        syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) => {
            let data = unnamed.iter().enumerate().map(|(i, field)| {
                f(
                    &field.ty,
                    &syn::Member::Unnamed(syn::Index {
                        index: i as u32,
                        span: Span::call_site(),
                    }),
                )
            });
            quote! { #ident(#(#data,)*) }
        }
        syn::Fields::Unit => quote! { #ident },
    }
}

/// Given the type of a field in a struct, determines which generics it uses should be "mapped"
/// when generating a `MappableBase` implementation.
fn get_mapped_generics<'a>(
    all_generics: &HashSet<&syn::Ident>,
    mapped_generics: &mut HashSet<&'a syn::Ident>,
    ty: &'a syn::Type,
) {
    match ty {
        syn::Type::Array(type_array) => {
            get_mapped_generics(all_generics, mapped_generics, &type_array.elem);
        }
        syn::Type::Group(type_group) => {
            get_mapped_generics(all_generics, mapped_generics, &type_group.elem)
        }
        syn::Type::Paren(type_paren) => {
            get_mapped_generics(all_generics, mapped_generics, &type_paren.elem)
        }
        syn::Type::Path(type_path) => {
            let last_segment = type_path.path.segments.last().unwrap();
            match &last_segment.arguments {
                syn::PathArguments::AngleBracketed(args) => {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(ty) = arg {
                            get_mapped_generics(all_generics, mapped_generics, ty);
                        }
                    }
                }
                syn::PathArguments::None if type_path.path.segments.len() == 1 => {
                    if all_generics.contains(&last_segment.ident) {
                        mapped_generics.insert(&last_segment.ident);
                    }
                }
                _ => (),
            }
        }
        syn::Type::Ptr(type_ptr) => {
            get_mapped_generics(all_generics, mapped_generics, &type_ptr.elem)
        }
        syn::Type::Reference(type_reference) => {
            get_mapped_generics(all_generics, mapped_generics, &type_reference.elem)
        }
        syn::Type::Slice(type_slice) => {
            get_mapped_generics(all_generics, mapped_generics, &type_slice.elem)
        }
        syn::Type::Tuple(type_tuple) => {
            for field in &type_tuple.elems {
                get_mapped_generics(all_generics, mapped_generics, field);
            }
        }
        _ => (),
    }
}

/// Determines whether the type for a [`syn::DeriveInput`] can possibly be contiguous.
fn is_possibly_contiguous(input: &syn::DeriveInput) -> bool {
    input.attrs.iter().any(|attr| {
        if attr.path().is_ident("repr") {
            let mut is_possibly_contiguous = false;
            attr.parse_nested_meta(|meta| {
                is_possibly_contiguous |=
                    meta.path.is_ident("C") || meta.path.is_ident("transparent");
                Ok(())
            })
            .unwrap();
            is_possibly_contiguous
        } else {
            false
        }
    })
}

/// Substitutes the identifiers in the given [`TokenStream`] using the given function.
fn substitute(
    source: TokenStream,
    write_sub: impl Copy + Fn(&mut TokenStream, &syn::Ident),
) -> TokenStream {
    let mut res = TokenStream::new();
    for token in source {
        match token {
            TokenTree::Group(group) => {
                res.append(TokenTree::Group(proc_macro2::Group::new(
                    group.delimiter(),
                    substitute(group.stream(), write_sub),
                )));
            }
            TokenTree::Ident(ident) => write_sub(&mut res, &ident),
            token => res.append(token),
        }
    }
    res
}

/// A helper for allocating unique [`syn::Ident`]s that don't conflict with existing identifiers.
struct IdentAllocator {
    taken: HashSet<String>,
}

impl IdentAllocator {
    /// Creates a new [`IdentAllocator`] that will not generate identifiers that conflict with the
    /// given set.
    pub fn new(taken: HashSet<String>) -> Self {
        Self { taken }
    }

    /// Gets a new identifier based on the given hint.
    pub fn alloc(&mut self, hint: String) -> syn::Ident {
        let mut name = hint;
        name.insert(0, '_');
        while self.taken.contains(&name) {
            name.insert(0, '_');
        }
        self.taken.insert(name.clone());
        syn::Ident::new(&name, Span::call_site())
    }
}
