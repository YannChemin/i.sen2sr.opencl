# TODO

## Adapt the OpenCL code to the device it runs on

Requested 2026-10-26, done the same day except the items left open below.
Rules: `~/.claude/CLAUDE.md`, "OpenCL code: adapt to the device it runs on".

- [x] Device queried at start-up (type, unified memory, PCI address via
      `cl_khr_pci_bus_info` or the NVIDIA query).
- [x] `profile=auto|cpu|igpu|dgpu|egpu`, detected by default, chosen
      profile and reason printed.
- [x] `precision=fp32|fp16`: CNN feature maps stored as half
      (`vload_half`/`vstore_half`, no `cl_khr_fp16` needed), arithmetic,
      input/output tiles and hard constraint in float. Max relative
      difference to fp32 in the tests: 1.4e-3.
- [x] `cpu`: host-memory buffers; `igpu`/`cpu`: transfers by map/unmap
      (host-memory buffers on `igpu` slowed Intel kernels by about 15 %).
- [x] `egpu`: pinned staging buffer for large transfers.
- [x] `cpu`: `conv2d_cpu`, 8 pixels per work-item with `float8` loads
      (5.8x faster convolutions on PoCL, i7-1165G7, mains power; fp16
      gives no speed gain there).
- [x] `I_SEN2SR_OPENCL_PROFILE=1`: event and wall-clock time per kernel,
      GFLOP/s for the convolutions.
- [x] Manual and tests (30 pass on PoCL and on Intel compute-runtime).
- [ ] `egpu`: overlap the tile transfers with compute (second queue,
      double-buffered input/output tiles); untested on a real eGPU.
- [ ] Fuse the element-wise passes of the hard constraint (`hc_diff`,
      `hc_mask`, `hc_add`) into the FFT passes.
- [ ] `conv2d` reaches about 90 GFLOP/s on the Iris Xe (mains power):
      try a tiled version (shared input rows in local memory).
