## DESCRIPTION

*i.sen2sr.opencl* increases the spatial resolution of Sentinel-2 bands
with the SEN2SR super-resolution models, returning the same bands on a
finer grid. It runs the "Lite" (convolutional) SEN2SR networks on an
OpenCL device (GPU or CPU), reading the published model weights
directly. Neither Python nor PyTorch is needed at run time.

SEN2SR is designed to be radiometrically consistent: after the neural
network has added spatial detail, a hard constraint replaces the low
spatial frequencies of the output with those of the input. Averaging the
output back to the input grid therefore gives the input reflectances
again, and the result can be used for quantitative work such as spectral
indices.

The three SEN2SRLite models are installed with the module and chosen
with the **model** option:

- **main**: all ten 10 m and 20 m bands to 2.5 m. The 20 m bands are
  first brought to 10 m, the 10 m bands (B02, B03, B04, B08) are
  super-resolved to 2.5 m, and a fusion network transfers their detail
  to the 20 m bands.
- **rgbn_x4**: the four 10 m bands (B02, B03, B04, B08) to 2.5 m.
- **rswir_x2**: the six 20 m bands (B05, B06, B07, B8A, B11, B12) to
  10 m, guided by the four 10 m bands.

With **model**=*auto*, the default, the model follows the bands given:
**main** if any 20 m band is given, **rgbn_x4** otherwise. **rswir_x2**
produces a different result, the 20 m bands on the 10 m grid, and is
only used when requested.

Each band is given through its own option: **blue** (B02), **green**
(B03), **red** (B04), **nir** (B08), **rededge1** (B05), **rededge2**
(B06), **rededge3** (B07), **nir08** (B8A), **swir16** (B11) and
**swir22** (B12), following the STAC common band names. The four 10 m
bands are always needed. The six 20 m bands are all needed by **main**
and **rswir_x2** and refused by **rgbn_x4**: a band is never silently
ignored.

The output maps are named after **output**, suffixed with the band name,
e.g. `output_B02`. They cover the current computational region with cells
4 times smaller (2 times for **rswir_x2**, whose output is on
the 10 m grid of the region). The computational region itself is not
changed. Output values are in the units of the input, as floating point
(FCELL) maps with the colour table of the matching input band.

### Models

The weights are those published under the CC0 licence at
[huggingface.co/tacofoundation/SEN2SR](https://huggingface.co/tacofoundation/SEN2SR)
(`SEN2SRLite/main`, `SEN2SRLite/NonReference_RGBN_x4` and
`SEN2SRLite/Reference_RSWIR_x2`), about 15 MB in total. They are
installed with the module in its `etc` directory
(`models/main`, `models/rgbn_x4`, `models/rswir_x2`), where
*i.sen2sr.opencl* finds them; no download is needed.

**model_dir** replaces the installed weights with those of another
directory holding one SEN2SRLite variant, e.g. retrained weights with the
same architecture: `model.safetensor` and `hard_constraint.safetensor`,
plus `sr_model`, `sr_hard_constraint`, `f2_model` and
`f2_hard_constraint` for the main model. The variant is recognised from
the files; if **model** is also set, the directory must hold that
variant. The full, Mamba-based SEN2SR models (the `SEN2SR/` directories)
are not supported.

### Input values

The models work on surface reflectance (Sentinel-2 L2A) in the 0-1
range. Input values are converted with reflectance = (value +
**offset**) / **scale** and the outputs are converted back to the input
units. The defaults (**scale**=10000, **offset**=0) match L2A digital
numbers from processing baselines before 04.00. For baseline 04.00 and
later (products from 25 January 2022), set **offset**=-1000. For inputs
already in reflectance, set **scale**=1.

Almost all Sentinel-2 L2A data in use today is of baseline 04.00 or
later, so **offset**=-1000 is usually needed: with the default, every
reflectance is 0.1 too high and the output is wrong. The default is right
only for older products and for bands imported with *r.in.sentinel* and
its **o** flag, which already removes the offset.

Input NULL cells are passed to the network as zero reflectance, as in
the reference implementation. The output cells they cover are set to
NULL in all output bands, so that no invented values are returned.

## NOTES

### Region and grid

The models are trained on 10 m Sentinel-2 cells. The current region must
therefore be on the 10 m grid of the scene, and the module warns if the
region resolution is not 10 m. The 20 m bands are read on that 10 m grid
(each 20 m cell covers 2 x 2 region cells). For the **main** and
**rswir_x2** models, the region must also be aligned with the
20 m cells: the model rebuilds the 20 m grid from the region origin, and
the module warns when a 20 m input map is not aligned with the region.
A suitable region is set in two steps: align the extent with a 20 m band,
which also sets a 20 m resolution, then refine the resolution to 10 m,
which keeps the extent:

```sh
g.region n=4845000 s=4840000 w=375000 e=380000 align=B11
g.region res=10
```

### Tiling

The networks work on tiles of 128 x 128 region cells, the size they
were trained on and the only size their hard-constraint filters are
defined for. Larger regions are processed in tiles overlapping by
**overlap** cells (32 by default, as in the reference
`predict_large()`); the last tile along each axis is moved back to end
on the region edge. Neighbouring tiles hand over in the middle of their
overlap, so that no output cell comes from within **overlap**/2 cells of
an inner tile edge. Regions smaller than a tile, or with an
odd number of rows or columns, are padded by repeating the last row and
column. The tile origins are kept on even cells so that they stay on the
20 m grid, which is why **overlap** must be even.

### OpenCL device

With **device**=auto, the first GPU is used, or a CPU device if there is
no GPU. **device**=gpu or **device**=cpu forces the choice.
**platform** restricts the search to OpenCL platforms whose name
contains the given text, e.g. `rusticl`, `Clover`, `AMD` or `Portable`
(PoCL, CPU). If the kernels do not build on one device, the next
candidate is tried. The kernels need OpenCL 1.1 in single precision.

The device memory needed is about 50 MiB for **rgbn_x4**,
15 MiB for **rswir_x2** and 200 MiB for **main**, whose fusion
network works at 2.5 m. On the host, one row of tiles of output is held
in memory, about 800 MiB for the ten bands of a full 10980 x 10980 cell
Sentinel-2 granule (proportionally less for narrower regions).

### Performance

With PoCL on a 4-core laptop CPU (Intel i7-1165G7), one 128 x 128 tile
takes about 0.35 s with **rgbn_x4**, 0.2 s with
**rswir_x2** and 5 s with **main**. With the default overlap,
a 10 x 10 km area (1000 x 1000 cells) is about 120 tiles. The work is
dominated by convolutions, which a GPU runs much faster than a CPU,
especially for **main**.

On an AMD Radeon Pro WX 7100 GPU (Polaris, 16 GiB, Mesa rusticl), the
same 10 x 10 km area takes 3.8 s with **rgbn_x4**, 1.9 s
with **rswir_x2** and 22 s with **main**, reading and writing
the maps included: about 0.03 s, 0.016 s and 0.18 s per tile, 11 to 27
times faster than the laptop CPU.

### Accuracy

The module reproduces the sen2sr PyTorch package: on the example data
shipped with each model, tiled into a 256 x 256 cell image, the outputs
of all three variants agree with PyTorch to within 2e-6 in reflectance,
tile stitching included.
The script `tests/reference_compare.py` in the source tree repeats this
check.

## EXAMPLES

### All bands to 2.5 m

Super-resolve an L2A scene of processing baseline 04.00 or later,
imported with *r.import* (map names are illustrative), on a 5 x 5 km
area:

```sh
g.region n=4845000 s=4840000 w=375000 e=380000 align=T31TCJ_B11
g.region res=10
i.sen2sr.opencl blue=T31TCJ_B02 green=T31TCJ_B03 red=T31TCJ_B04 \
    nir=T31TCJ_B08 rededge1=T31TCJ_B05 rededge2=T31TCJ_B06 \
    rededge3=T31TCJ_B07 nir08=T31TCJ_B8A swir16=T31TCJ_B11 \
    swir22=T31TCJ_B12 offset=-1000 output=T31TCJ_sr
d.rgb red=T31TCJ_sr_B04 green=T31TCJ_sr_B03 blue=T31TCJ_sr_B02
```

### RGB and NIR only

With the four 10 m bands only, the **rgbn_x4** model is used:

```sh
i.sen2sr.opencl blue=B02 green=B03 red=B04 nir=B08 offset=-1000 \
    output=rgbn
```

### 20 m bands to 10 m

```sh
i.sen2sr.opencl blue=B02 green=B03 red=B04 nir=B08 rededge1=B05 \
    rededge2=B06 rededge3=B07 nir08=B8A swir16=B11 swir22=B12 \
    model=rswir_x2 offset=-1000 output=s2_10m
```

## REFERENCES

Aybar, C., Contreras, J., et al. SEN2SR: a radiometrically and
spatially consistent super-resolution framework for Sentinel-2.
Preprint,
[SSRN 5247739](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5247739).

SEN2SR source code:
[github.com/ESAOpenSR/sen2sr](https://github.com/ESAOpenSR/sen2sr)

## SEE ALSO

*[i.fusion.hpf](i.fusion.hpf.md), [i.pansharpen](i.pansharpen.md),
[i.sentinel.import](i.sentinel.import.md), [r.in.sentinel](r.in.sentinel.md),
[r.resamp.interp](r.resamp.interp.md)*

## AUTHORS

Yann Chemin
