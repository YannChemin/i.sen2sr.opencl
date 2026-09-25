"""Compare i.sen2sr.opencl with the sen2sr PyTorch reference.

Developer check, not part of the pytest suite: it needs PyTorch, a
checkout of https://github.com/ESAOpenSR/sen2sr and the SEN2SRLite model
directories. For each model, the example input shipped with the model is
tiled into a larger image, super-resolved by the module, and every tile
core is compared with the PyTorch output of the tile it comes from.

Usage (inside a GRASS session on a metric project):

    python3 reference_compare.py SEN2SR_SOURCE MODEL_DIR [MODEL_DIR ...]
"""

import importlib.util
import sys
from pathlib import Path

import grass.script as gs
import numpy as np
from grass.script import array as garray

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
KEYS = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
]
TILE = 128


def tile_layout(length, overlap):
    """Same tile origins and cores as tile_layout() in main.c."""
    starts, s = [], 0
    while True:
        starts.append(s)
        if s + TILE >= length:
            break
        s += TILE - overlap
        s = min(s, length - TILE)
    cores = [0] + [(a + TILE + b) // 2 for a, b in zip(starts, starts[1:])]
    return starts, cores + [length]


def main():
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    sys.path.insert(0, sys.argv[1])
    overlap = 32
    # 2 x 2 mirrored copies of the 128 x 128 example: 256 x 256 cells.
    size = 2 * TILE
    for model_dir in map(Path, sys.argv[2:]):
        spec = importlib.util.spec_from_file_location("load", model_dir / "load.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        model = loader.compiled_model(model_dir, "cpu")

        lr = load_file(model_dir / "example_data.safetensor")["lr"][0].numpy()
        lr = np.nan_to_num(lr.astype(np.float32))
        if lr.shape[0] == 4:
            # NonReference_RGBN_x4 examples are B04 B03 B02 B08.
            lr = lr[[2, 1, 0, 3]]
            names = ["B02", "B03", "B04", "B08"]
        else:
            names = BANDS
        big = np.concatenate([lr, lr[:, ::-1]], axis=1)
        big = np.concatenate([big, big[:, :, ::-1]], axis=2)

        gs.run_command("g.region", n=size * 10, s=0, e=size * 10, w=0, res=10)
        opts = {}
        for i, name in enumerate(names):
            a = garray.array(dtype=np.float32)
            a[...] = big[i]
            a.write(f"ref_in_{name}", overwrite=True)
            opts[KEYS[BANDS.index(name)]] = f"ref_in_{name}"
        gs.run_command(
            "i.sen2sr.opencl",
            output="ref_out",
            model_dir=str(model_dir),
            scale=1,
            overlap=overlap,
            overwrite=True,
            **opts,
        )

        net_in = big[[2, 1, 0, 3]] if lr.shape[0] == 4 else big
        starts, cores = tile_layout(size, overlap)
        ref = None
        with torch.no_grad():
            for ty, y0 in enumerate(starts):
                for tx, x0 in enumerate(starts):
                    t = torch.from_numpy(
                        np.ascontiguousarray(net_in[:, y0 : y0 + TILE, x0 : x0 + TILE])
                    )
                    out = model(t[None])[0].numpy()
                    if lr.shape[0] == 4:
                        out = out[[2, 1, 0, 3]]
                    elif out.shape[0] == 10 and out.shape[1] == TILE:
                        out = out[[3, 4, 5, 7, 8, 9]]
                    s = out.shape[1] // TILE
                    if ref is None:
                        ref = np.zeros((out.shape[0], size * s, size * s), np.float32)
                    ys, ye, xs, xe = cores[ty], cores[ty + 1], cores[tx], cores[tx + 1]
                    ref[:, ys * s : ye * s, xs * s : xe * s] = out[
                        :, (ys - y0) * s : (ye - y0) * s, (xs - x0) * s : (xe - x0) * s
                    ]

        out_names = sorted(
            gs.list_strings("raster", pattern="ref_out_*", mapset="."),
            key=lambda n: BANDS.index(n.split("@")[0].rsplit("_", 1)[1]),
        )
        gs.run_command("g.region", n=size * 10, s=0, e=size * 10, w=0, res=10 / s)
        worst = 0.0
        for i, name in enumerate(out_names):
            got = garray.array(name.split("@")[0], dtype=np.float32)
            diff = float(np.abs(np.asarray(got) - ref[i]).max())
            worst = max(worst, diff)
            print(f"  {name}: max |diff| = {diff:.3g}")
        gs.run_command("g.remove", type="raster", pattern="ref_*", flags="f")
        print(f"{model_dir.name}: {len(out_names)} bands, worst {worst:.3g}")


if __name__ == "__main__":
    main()
