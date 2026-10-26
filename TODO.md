# TODO

## Adapt the OpenCL code to the device it runs on (next session)

Requested 2026-10-26. Bring this code to the device-adaptation rules of
`~/.claude/CLAUDE.md` ("OpenCL code: adapt to the device it runs on"), as
done first in `i.sam.opencl`, `i.sam2.opencl` and `i.sam3.opencl`.

- [ ] Query the device at start-up (type, `CL_DEVICE_HOST_UNIFIED_MEMORY`,
      global and max-allocation memory, local memory, compute units,
      preferred work-group multiple, extensions such as `cl_khr_fp16`,
      `cl_khr_fp64`, `cl_khr_subgroups`, `cl_khr_pci_bus_info`).
- [ ] Derive a device profile and expose it as an option with automatic
      detection as default: `profile=auto|cpu|igpu|dgpu|egpu` (GRASS
      parser option, or the equivalent CLI flag / environment variable).
      Report the chosen profile and why in a verbose message.
- [ ] Data type option: `precision=` with the types that make sense here
      (fp32/fp16 storage for weights and activations; int16/uint16 for
      hyperspectral DN, converted in the kernel; keep fp64 where the
      numerics need it, and say so in the manual). Validate every reduced
      precision against the full-precision path.
- [ ] `igpu`: zero-copy buffers (`CL_MEM_ALLOC_HOST_PTR`, map/unmap
      instead of write/read copies).
- [ ] `egpu` / remote: keep data resident on the device, pinned staging,
      overlap transfers with compute (two queues, double-buffered tiles);
      never stream a large cube through the link per kernel.
- [ ] `cpu`: kernels without local memory, vectorised along the
      contiguous axis (PoCL runs GPU-shaped local-memory kernels at about
      1 % of peak), or a host BLAS on mapped buffers.
- [ ] Kernel fusion of element-wise passes into their producer/consumer.
- [ ] Profiling switch reporting per-kernel time (wall clock with
      `clFinish`: event timestamps are wrong on rusticl iris), achieved
      GB/s and GFLOP/s.
- [ ] A forced profile or precision that cannot run fails loudly; no
      silent fallback.
- [ ] Manual (`.md` and `.html`) and tests updated for the new options.
