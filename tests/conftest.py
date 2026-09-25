"""pytest fixtures for i.sen2sr.opencl.

The module always needs an OpenCL device (PoCL is enough). The three
SEN2SRLite models are installed with the module, so the model tests need
no download. I_SEN2SR_OPENCL_PLATFORM optionally selects the OpenCL
platform.
"""

import os
from pathlib import Path

import grass.script as gs
import pytest
from grass.tools import Tools

# A 10 m UTM grid of odd size, so that padding and tile clamping are used.
ROWS = 150
COLS = 131
RES = 10

BAND_KEYS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12",
}


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session on a UTM project with ten smooth synthetic bands <s2_B*>
    in L2A digital numbers (reflectance x 10000), and a null cell in
    <s2_B04>."""
    project = tmp_path_factory.mktemp("i_sen2sr_opencl") / "project"
    gs.create_project(project, epsg="32631")
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(
            n=5000000 + ROWS * RES,
            s=5000000,
            w=500000,
            e=500000 + COLS * RES,
            res=RES,
        )
        for i, band in enumerate(BAND_KEYS.values()):
            tools.r_mapcalc(
                expression=(
                    f"s2_{band} = 1500 + 300 * {i % 4} + "
                    f"800 * sin(row() * {7 + i}) * cos(col() * {5 + i})"
                )
            )
        tools.r_mapcalc(
            expression="s2_B04 = if(row() == 10 && col() == 20, null(), s2_B04)",
            overwrite=True,
        )
        yield session


def bundled_model(env, name):
    """Installed directory of a bundled model, found where the module
    looks for it (GRASS_ADDON_ETC, $GISBASE/etc, GRASS_ADDON_BASE/etc)."""
    rel = Path("i.sen2sr.opencl", "models", name)
    roots = env.get("GRASS_ADDON_ETC", "").split(":")
    roots.append(str(Path(env["GISBASE"], "etc")))
    if env.get("GRASS_ADDON_BASE"):
        roots.append(str(Path(env["GRASS_ADDON_BASE"], "etc")))
    for root in filter(None, roots):
        if (Path(root) / rel / "model.safetensor").is_file():
            return str(Path(root) / rel)
    pytest.skip(f"bundled model <{name}> is not installed")


@pytest.fixture(scope="module")
def device_options():
    platform = os.environ.get("I_SEN2SR_OPENCL_PLATFORM")
    return {"platform": platform} if platform else {}


def band_options(*keys):
    """Input options for the given band keys."""
    return {key: f"s2_{BAND_KEYS[key]}" for key in keys}
