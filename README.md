# i.sen2sr.opencl

GRASS GIS addon that super-resolves Sentinel-2 bands with the
[SEN2SR](https://github.com/ESAOpenSR/sen2sr) "Lite" (convolutional)
models on an OpenCL device (GPU or CPU). The published model weights
(`*.safetensor`) are read directly, so neither Python nor PyTorch is
needed at run time.

Supported model variants (detected from the files in `model_dir`):

| Variant | Input | Output |
|---|---|---|
| `main` | all ten 10 m and 20 m bands | 2.5 m |
| `NonReference_RGBN_x4` | B02, B03, B04, B08 (10 m) | 2.5 m |
| `Reference_RSWIR_x2` | B05, B06, B07, B8A, B11, B12 (20 m), guided by the 10 m bands | 10 m |

The outputs are radiometrically consistent with the inputs: a hard
constraint restores the low spatial frequencies of the input after the
network has added detail.

See [i.sen2sr.opencl.md](i.sen2sr.opencl.md) for the full manual.

## Requirements

- GRASS GIS 8.x source tree (for building the addon)
- An OpenCL 1.1 implementation with single-precision support, e.g.
  Mesa rusticl/Clover, a vendor GPU driver, or PoCL for CPUs
- OpenCL headers and ICD loader (`libOpenCL`), e.g. Debian packages
  `opencl-headers` and `ocl-icd-opencl-dev`
- SEN2SRLite model weights (CC0) from
  [huggingface.co/tacofoundation/SEN2SR](https://huggingface.co/tacofoundation/SEN2SR),
  one directory per variant under `SEN2SRLite/`

## Build and install

Standalone, against a GRASS source tree:

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

Or from within GRASS:

```sh
g.extension extension=i.sen2sr.opencl url=/path/to/i.sen2sr.opencl
```

The OpenCL kernels in `sen2sr_kernels.cl` are embedded into the binary
at build time.

## Getting the models

```sh
base=https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main
mkdir -p SEN2SRLite_main
for f in model hard_constraint sr_model sr_hard_constraint f2_model \
         f2_hard_constraint; do
    curl -L -o SEN2SRLite_main/$f.safetensor $base/$f.safetensor
done
```

`NonReference_RGBN_x4` and `Reference_RSWIR_x2` only need `model` and
`hard_constraint`.

## Usage

The region must be on the scene's 10 m grid and, for models using the
20 m bands, aligned with the 20 m cells:

```sh
g.region n=4845000 s=4840000 w=375000 e=380000 align=T31TCJ_B11
g.region res=10
i.sen2sr.opencl blue=T31TCJ_B02 green=T31TCJ_B03 red=T31TCJ_B04 \
    nir=T31TCJ_B08 rededge1=T31TCJ_B05 rededge2=T31TCJ_B06 \
    rededge3=T31TCJ_B07 nir08=T31TCJ_B8A swir16=T31TCJ_B11 \
    swir22=T31TCJ_B12 offset=-1000 output=T31TCJ_sr \
    model_dir=$HOME/models/SEN2SRLite_main
```

Outputs are FCELL maps named `<output>_<band>` (e.g. `T31TCJ_sr_B02`).
Use `offset=-1000` for L2A products of processing baseline 04.00 and
later, and `scale=1` for inputs already in reflectance. The OpenCL
device is chosen with `device=auto|gpu|cpu` and `platform=<name>`.

## Tests

Tests use pytest and need an OpenCL device (PoCL is enough). Tests that
run a model are skipped unless a model directory is given:

```sh
export I_SEN2SR_OPENCL_RGBN=$HOME/models/SEN2SRLite_NonReference_RGBN_x4
export I_SEN2SR_OPENCL_MAIN=$HOME/models/SEN2SRLite_main
export I_SEN2SR_OPENCL_PLATFORM=Portable  # optional
PYTHONPATH=$(grass --config python_path) python3 -m pytest tests/
```

The module must be installed (or on `PATH`) for the tests to find it.

`tests/reference_compare.py` is a developer check against the sen2sr
PyTorch package (needs PyTorch and a sen2sr checkout); run it inside a
GRASS session on a metric project:

```sh
python3 tests/reference_compare.py SEN2SR_SOURCE MODEL_DIR [MODEL_DIR ...]
```

## Reference

Aybar, C., Contreras, J., et al. SEN2SR: a radiometrically and spatially
consistent super-resolution framework for Sentinel-2. Preprint,
[SSRN 5247739](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5247739).

## License

Code: GPL-2.0-or-later. Model weights: CC0 (distributed separately).

## Author

Yann Chemin
