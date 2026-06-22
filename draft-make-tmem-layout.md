---
layout: post
permalink: /draft/make-tmem-layout/
sitemap: false
---

Draft: Explain make_tmem_layout: an application for CuTe Layout left_inverse

Take the example of Op `SM100_TMEM_LOAD_32dp32b4x`.

An example host code to show results:

```c++
auto tmem_layout = make_layout(
  make_shape (make_shape(_128{},        _256{}), _1{}, _1{}),
  make_stride(make_stride(Int<65536>{}, _1{}),   _0{}, _0{}));

// tmem_[32b](0x0000.0000)  -> 32-bit element (float) TMEM pointer at address 0
Tensor tCtAcc = make_tensor(make_tmem_ptr<float>(0), tmem_layout);

print(tCtAcc); print("\n");  // tmem_[32b](0x0000.0000) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)

//   template <>
// struct Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>
//   {
//     using ThrID = Layout<_32>;
//     using ValID = Layout<Shape <_128,       _32>,
//                          Stride<  _1,TMEM::DP_b>>;
//     using SrcLayout = Layout<Shape <_32,_4096>,
//                              Stride< _0,   _1>>;
//     using DstLayout = Layout<Shape < _32,_128>,
//                              Stride<_128,  _1>>;
//     using RefLayout = SrcLayout;
//   };
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b4x{}, tCtAcc);

print("tiled_t2r_copy:\n"); print(tiled_t2r_copy); print("\n");
//tiled_t2r_copy:	
//TiledCopy
//  Tiler_MN:       ((_16,_32):(_32,_1),_1:_0,_1:_0)
//  TiledLayout_TV: ((_32,_4),(_4,_32)):((_0,_1),(_4,_16))
//Copy_Atom
//  ThrID:        _32:_1
//  ValLayoutSrc: (_32,_128):(_0,_1)
//  ValLayoutDst: (_32,_4):(_4,_1)
//  ValLayoutRef: (_32,_128):(_0,_1)
//  ValueType:    32b
```

Explaining the `Traits`: `ValID` is a layout of value (bit) idx to physical TMEM address for every bit. 
`TMEM::DP_b` is `1 << 21` (e.g. address offset of two neighbor 32 bit cell on two rows 0x0001.0000 and 0x0002.0000, times 32 for bit addressing)


Library code needed:
```c++
/** Generate a TiledCopy from a CopyAtom and a TMEM tensor
 * Example:
 *   Tensor gmem_tensor = ...                                            // (M,N,...)
 *   Tensor tmem_tensor = ...                                            // (M,N,...)
 *   auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD_Operation, tmem_tensor);
 *   auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
 *
 *   Tensor tDtC = thr_tmem_load.partition_S(tmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDgC = thr_tmem_load.partition_D(gmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDrC = make_tensor<ElementAccumulator>(shape(tDgD));         // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *
 *   copy(tiled_tmem_load, tDtC, tDrC);       // tmem -> rmem
 *   copy(tDrC, tDgC);                   // rmem -> gmem
 */
template <class CopyOp, class CopyT,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(Copy_Atom<CopyOp,CopyT> const& atom,
               Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T      = typename TEngine::value_type;
  using Traits = typename Copy_Atom<CopyOp, CopyT>::Traits;
  static_assert(sizeof_bits_v<CopyT> == sizeof_bits_v<T>,
                "Expected a CopyAtom with the same type-width as the Tensor.");

  // atom thr idx -> tmem addr    4warps where each warp points to the same position within it's own subpartition
  auto atom_t_layout = Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>{};
  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(atom, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}
```
`atom_t_layout` has shape `(32, 4)`: 32 threads per warp, 4 warps. Stride for thread mode is 0 because each warp accesses 
the same tmem address (though each thread gets different values back). Stride for warp mode is 32 (rows) times the 
data-type-casted row stride(e.g. `TMEM::DP<T>` gets `1<<16` for 32b elements).

`atom_v_layout` is just upcast of `Traits::ValID` above from per-bit val idx to the right data type. So `(128, 32):(1, 2^21)`
will become `(4, 32):(1, 2^16)` for 32b elements.

`make_layout` will concatenate them.

```c++
/** Produce a TiledCopy from thread and value offset maps.
 * The TV Layout maps threads and values to the codomain of the data_layout.
 * It is verified that the intended codomain is valid within data_layout.
 * Useful when threads and values don't care about owning specific coordinates, but
 *   care more about the vector-width and offsets between them.
 */
template <class... Args, class AtomTVLayout, class DataLayout>
CUTE_HOST_DEVICE constexpr
auto
make_cotiled_copy(Copy_Atom<Args...> const& copy_atom,
                  AtomTVLayout const& atom_tv_layout,   // atom (thr,val) -> data addr
                  DataLayout   const& data_layout)      // coord          -> data addr    The target layout
{
  static_assert(is_static<AtomTVLayout>::value);
  static_assert(is_static<DataLayout>::value);

  // data addr -> data coord    Append 1:0 so off-the-ends get the stride-0
  auto inv_data_layout = make_layout(left_inverse(data_layout), Layout<_1,_0>{});

  // (tid,vid) -> data_coord
  auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);

  // Check validity
  // Append 1:0 to data_layout so that OOB coordinates get the stride-0
  CUTE_STATIC_ASSERT_V(coalesce(composition(make_layout(data_layout, Layout<_1,_0>{}), layout<1>(layout_tv_data))) == coalesce(layout<1>(atom_tv_layout)),
                       "The memory pointed to by AtomTVLayout does not exist in the DataLayout.");
  //
  // Tiler -- Find the active elements in the DATA tensor and generate a tiler to extract them
  //

  // Convert to the awkward by-mode tiler to preserve the modes of the tiled DATA
  auto flat_data_shape = product_each(shape(data_layout));
  auto flat_data_zeros = repeat<rank(flat_data_shape)>(Int<0>{});

  auto tiler = transform(make_seq<rank(flat_data_shape)>{}, [&](auto i) {
    return filter(composition(make_layout(flat_data_shape, replace<i>(flat_data_zeros, Int<1>{})), layout_tv_data));
  });

  //
  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  //

  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> data_coord
  auto tile2data = composition(make_layout(flat_data_shape), tiler);

  // (tid,vid) -> tile_coord
  auto layout_tv = composition(left_inverse(tile2data), layout_tv_data);
  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

`atom_tv_layout`: concatenated layouts above. Mapping (thr idx, v idx) to tmem physical addr.

`data_layout`: mapping tmem logical coord to physical addr.

Now to construct of a mapping from (tid, vid) to tmem logical coord (`data_coord`), we'd need to do a left_inverse and then 
compose them:
```c++
// data addr -> data coord    Append 1:0 so off-the-ends get the stride-0
auto inv_data_layout = make_layout(left_inverse(data_layout), Layout<_1,_0>{});

// (tid,vid) -> data_coord
auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);
```

Left inverse: from the paper:
If we have a layout $L : \mathbb{Z}_{|L|} \to D$, 
then its left inverse is
$$L^\dagger : D_{L^\dagger} \to \mathbb{Z}_{|L|}$$
 and satisfies
$$\forall k \in \mathbb{Z}_{|L|},\; L(L^\dagger(L(k))) = L(k)$$

If we define `data_layout` as $L$, and `atom_tv_layout` as $T$, then `layout_tv_data` will be $L^\dagger \circ T$.
We want to make sure
$$\forall i \in \mathbb{Z}_{|T|},\; L(L^\dagger(T(i))) = T(i)$$

by checking it in code
```c++
// Check validity
// Append 1:0 to data_layout so that OOB coordinates get the stride-0
CUTE_STATIC_ASSERT_V(coalesce(composition(make_layout(data_layout, Layout<_1,_0>{}), layout<1>(layout_tv_data))) == coalesce(layout<1>(atom_tv_layout)),
                   "The memory pointed to by AtomTVLayout does not exist in the DataLayout.");
```
Implication: For any possible value $T(i)$, it's in the image of $L$ (note the equation is of the form $L(...)=T(i)$).
In other words, any output tmem addr of `atom_tv_layout` is also an output addr of `data_layout`, meaning that the 
TMEM access Atom will not load/store out of bounds physical addresses pointed to by the TMEM data layout.

In my opinion, there's also an implicit assumption that $L$ is injective, meaning for each physical addr, there's at most 
one data coord mapped to it. If we have this assumption, we could then say that for any $T(i)$ there's a unique $y$ such 
that $L(y) = T(i)$, and since we already have $L(L^\dagger(T(i))) = T(i)$, $L^\dagger(T(i))$ will be that $y$. i.e. Given 
any tvid input `i`, if we compute tmem addr by `atom_tv_layout`, there will be exactly one unique data coord `y` in the 
domain of `data_layout` that will map to the same tmem addr. And `y = layout_tv_data(i)`.

Now that we have a layout mapping tvid to data coord, next step is to add one more layer in the middle for tiling.

First compute a tiler
```c++
  // Convert to the awkward by-mode tiler to preserve the modes of the tiled DATA
  auto flat_data_shape = product_each(shape(data_layout));
  auto flat_data_zeros = repeat<rank(flat_data_shape)>(Int<0>{});

  auto tiler = transform(make_seq<rank(flat_data_shape)>{}, [&](auto i) {
    return filter(composition(make_layout(flat_data_shape, replace<i>(flat_data_zeros, Int<1>{})), layout_tv_data));
  });
```
The result tiler will be of multiple modes. For each mode `i`, we try to map domain of `layout_tv_data` onto a 1d coord space
of shape `flat_data_shape[i]` by setting its stride to 1 leaving other strides 0. Then `filter` out stride 0 modes, such 
as thread mode of shape 32. This `tiler` is in fact the `Tiler_MN` of `TiledCopy`.

Then we can use the tiler's shape to define the domain of `"tile_coord"`, and use the tiler's codomain to represent the 
`data_coord` mapped to from the `tild_coord`.

Note the tiler's shape here is `(16, 32)`, which really doesn't agree with tv layouts anymore. We shouldn't interpret it 
as any kind of composition of tid and vid. Instead, it's just another logical coord space (`tild_coord`) that we can later map to.

Now we have `layout_tv_data: (tid, vid) -> data_coord`, and `tiler: (tile_coord) -> data_coord`. 
What we want to construct as `TiledLayout_TV` for `TiledCopy` will be `(tid, vid) -> tile_coord`. So similar to the 
process above, we use compose+left_inverse to combine the two together. Note `tiler` is in fact a tiler instead of a 
function. To invert it, we first compose with `flat_data_shape` to make it a function.
```c++
  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  //

  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> data_coord
  auto tile2data = composition(make_layout(flat_data_shape), tiler);

  // (tid,vid) -> tile_coord
  auto layout_tv = composition(left_inverse(tile2data), layout_tv_data);
```
And `layout_tv` is `TiledLayout_TV`.

As a result, here's how we should interpret such a `TiledCopy`:
```
tiled_t2r_copy:	
TiledCopy
  Tiler_MN:       ((_16,_32):(_32,_1),_1:_0,_1:_0)
  TiledLayout_TV: ((_32,_4),(_4,_32)):((_0,_1),(_4,_16))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_128):(_0,_1)
  ValLayoutDst: (_32,_4):(_4,_1)
  ValLayoutRef: (_32,_128):(_0,_1)
  ValueType:    32b
```

- `TiledLayout_TV`: this is a layout of `(tid, vid)` to `tile_coord` in `Tiler_MN`'s domain space
- `Tiler_MN`: this is a layout of `tile_coord` to `data_coord` in tmem tensor's domain space
- `ValLayoutSrc`: I think it is a layout of `(tid', vid')` to `(tid, vid)` in `ValID`'s domain

Example0: 

Given `((t0, w0), (v0, v1))` in `TiledLayout_TV`'s domain, 

- we first apply its stride to calculate `flat_tiled_coord = 0*t0 + 1*w0 + 4*v0 + 16*v1`
- then we convert it to natural coord before feeding into `Tiler_MN`: `m0 = flat_tiler_coord % 16 = w0+4*v0`,
`n0 = flat_tiler_coord // 16 = v1`
- then apply `Tiler_MN`'s stride: `flat_data_coord = 32*m0+n0 = 32*w0 + 128*v0 + v1`
- then convert to natural data coord: `M0 = flat_data_coord % 128 = 32*w0 + v1`, `N0 = flat_data_coord // 128 = v0`
- then apply `data_layout` (e.g. `tCtAcc`) to get the TMEM address: `addr = M0 * 2^16 + N0*1 = (32*w0 + v1) * 2^16 + v0`

This means a thread mode change in `TiledLayout_TV` does not affect tmem addr, a warp mode change by 1 causes a tmem 
addr change by 32*2^16 (e.g. 0x0000.0000->0x0020.0000, 32 physical rows). A change in v0 mode causes a tmem addr change 
by 1, i.e. one column, meaning v0 mode represents the 4x part of the Atom. A change in v1 mode causes a tmem addr change 
by 2^16, i.e. one row, meaning v1 mode represents the in-warp 32dp part of the Atom.

Printed:
```
atom_tv_layout:	((_32,_4),(_4,_32)):((_0,_2097152),(_1,_65536))
data_layout:	((_128,_256),_1,_1):((_65536,_1),_0,_0)
inv_data_layout:	((_65536,_128),_1):((_128,_1),_0)
layout_tv_data:	((_32,_4),(_4,_32)):((_0,_32),(_128,_1))

flat_data_shape:	(_32768,_1,_1)
flat_data_zeros:	(_0,_0,_0)
tiler:	((_16,_32):(_32,_1),_1:_0,_1:_0)

tile2data:	((_16,_32),_1,_1):((_32,_1),_0,_0)

layout_tv:	((_32,_4),(_4,_32)):((_0,_1),(_4,_16))
```