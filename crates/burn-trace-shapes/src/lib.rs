use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    FnArg, GenericArgument, ItemFn, Pat, PathArguments, ReturnType, Type,
    parse_macro_input,
    punctuated::Punctuated,
    token::Comma,
};

/// Attribute macro that traces input and output tensor shapes.
///
/// Automatically retrieves the struct name at runtime via `std::any::type_name::<Self>()`.
/// No arguments needed.
///
/// Emits a single line per call:
///   `[forward] WNConv1d (x: [1, 32, 30500]) -> [1, 32, 30500]`
///
/// # Usage
/// ```rust,ignore
/// #[trace_shapes]
/// pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> { .. }
/// ```
#[proc_macro_attribute]
pub fn trace_shapes(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
    //let input_fn = parse_macro_input!(item as ItemFn);
    //match transform_fn(input_fn) {
    //    Ok(ts) => ts.into(),
    //    Err(e) => e.to_compile_error().into(),
    //}
}

fn transform_fn(mut func: ItemFn) -> syn::Result<TokenStream2> {
    let fn_name = func.sig.ident.to_string();
    let original_block = func.block.clone();
    let ret_type = func.sig.output.clone();

    let tensor_params = collect_tensor_params(&func.sig.inputs);

    // Build format string for inputs: "(x: {:?}, mask: {:?})"
    let input_fmt_parts: Vec<String> = tensor_params
        .iter()
        .map(|(name, _)| format!("{{:?}}"))
        .collect();
    let inputs_fmt = format!("[{}]", input_fmt_parts.join(","));

    // Capture dims into locals BEFORE the closure moves the tensors
    let input_capture_stmts: Vec<TokenStream2> = tensor_params
        .iter()
        .map(|(name, kind)| {
            let src = syn::Ident::new(name, proc_macro2::Span::call_site());
            let dst = syn::Ident::new(
                &format!("__trace_in_{}", name),
                proc_macro2::Span::call_site(),
            );
            match kind {
                TensorKind::Plain => quote! { let #dst = #src.dims(); },
                TensorKind::Option => quote! {
                    let #dst = #src.as_ref().map(|__t| __t.dims());
                },
                TensorKind::None => unreachable!(),
            }
        })
        .collect();

    let input_dim_idents: Vec<TokenStream2> = tensor_params
        .iter()
        .map(|(name, _)| {
            let ident = syn::Ident::new(
                &format!("__trace_in_{}", name),
                proc_macro2::Span::call_site(),
            );
            quote! { #ident }
        })
        .collect();

    let (output_fmt, output_expr) = build_output_parts(&ret_type);

    // "[{fn_name}] {struct_name} {inputs_fmt}{output_fmt}"
    // struct_name is resolved at runtime from type_name::<Self>()
    let full_fmt = format!("[{}] {{}} {}{}", fn_name, inputs_fmt, output_fmt);

    let new_block = syn::parse2(quote! {
        {
            // Resolve struct name at runtime, strip module path and generics:
            // "my_crate::model::WNConv1d<NdArray<f32>>" -> "WNConv1d"
            let __trace_struct_name = {
                let full = std::any::type_name::<Self>();
                // Take the last `::` segment, then strip anything from `<` onward
                full.split("<").next().unwrap_or(full)
                    .split("::").last().unwrap_or(full)
            };

            // Capture input dims before the closure moves the tensors
            #(#input_capture_stmts)*

            let __trace_result = (|| #original_block)();

            eprintln!(#full_fmt, __trace_struct_name, #(#input_dim_idents,)* #output_expr);

            __trace_result
        }
    })?;

    func.block = Box::new(new_block);
    Ok(quote! { #func })
}

// ---------------------------------------------------------------------------
// Input collection
// ---------------------------------------------------------------------------

fn collect_tensor_params(inputs: &Punctuated<FnArg, Comma>) -> Vec<(String, TensorKind)> {
    let mut result = Vec::new();
    for arg in inputs {
        let FnArg::Typed(pat_type) = arg else { continue };
        let Pat::Ident(pat_ident) = pat_type.pat.as_ref() else { continue };
        let kind = classify_type(&pat_type.ty);
        if !matches!(kind, TensorKind::None) {
            result.push((pat_ident.ident.to_string(), kind));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Output format + expression builder
// ---------------------------------------------------------------------------

fn build_output_parts(ret: &ReturnType) -> (String, TokenStream2) {
    let ReturnType::Type(_, ty) = ret else {
        return (String::new(), quote! {});
    };
    match classify_output_type(ty) {
        OutputKind::PlainTensor => (
            " -> {:?}".to_string(),
            quote! { __trace_result.dims() },
        ),
        OutputKind::TupleWithTensors(indices) => {
            let fmt_parts: Vec<String> = indices
                .iter()
                .map(|i| format!("{{:?}}"))
                .collect();
            let fmt = format!(" -> ({})", fmt_parts.join(", "));
            let exprs: Vec<TokenStream2> = indices
                .iter()
                .map(|i| {
                    let idx = syn::Index::from(*i);
                    quote! { __trace_result.#idx.dims() }
                })
                .collect();
            (fmt, quote! { #(#exprs),* })
        }
        OutputKind::None => (String::new(), quote! {}),
    }
}

// ---------------------------------------------------------------------------
// Type classification
// ---------------------------------------------------------------------------

enum TensorKind { Plain, Option, None }
enum OutputKind { PlainTensor, TupleWithTensors(Vec<usize>), None }

fn is_tensor_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else { return false };
    type_path.path.segments.last()
        .map(|seg| seg.ident == "Tensor")
        .unwrap_or(false)
}

fn classify_type(ty: &Type) -> TensorKind {
    if is_tensor_type(ty) { return TensorKind::Plain; }
    let Type::Path(type_path) = ty else { return TensorKind::None };
    let Some(seg) = type_path.path.segments.last() else { return TensorKind::None };
    if seg.ident != "Option" { return TensorKind::None; }
    let PathArguments::AngleBracketed(ref args) = seg.arguments else { return TensorKind::None };
    if let Some(GenericArgument::Type(inner_ty)) = args.args.first() {
        if is_tensor_type(inner_ty) { return TensorKind::Option; }
    }
    TensorKind::None
}

fn classify_output_type(ty: &Type) -> OutputKind {
    if is_tensor_type(ty) { return OutputKind::PlainTensor; }
    if let Type::Tuple(type_tuple) = ty {
        let indices: Vec<usize> = type_tuple.elems.iter().enumerate()
            .filter_map(|(i, e)| if is_tensor_type(e) { Some(i) } else { None })
            .collect();
        if !indices.is_empty() { return OutputKind::TupleWithTensors(indices); }
    }
    OutputKind::None
}
