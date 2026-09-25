"""Tests for i.sen2sr.opencl."""

import grass.script as gs
import pytest
from conftest import BAND_KEYS, COLS, RES, ROWS, band_options
from grass.tools import ToolError, Tools

RGBN = ("blue", "green", "red", "nir")


def native_univar(session, name):
    """r.univar of a map on its own grid, leaving the region alone."""
    env = session.env.copy()
    env["GRASS_REGION"] = gs.region_env(raster=name, env=session.env)
    return Tools(env=env).r_univar(map=name, format="json").json


def test_overlap_must_be_even(session, tmp_path):
    """An odd overlap would break the 20 m grid alignment of the tiles."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="even"):
        tools.i_sen2sr_opencl(
            output="out", model_dir=str(tmp_path), overlap=31, **band_options(*RGBN)
        )


def test_rejects_directory_without_model(session, tmp_path, device_options):
    """A directory without SEN2SRLite weights fails loudly."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not found"):
        tools.i_sen2sr_opencl(
            output="out",
            model_dir=str(tmp_path),
            **band_options(*RGBN),
            **device_options,
        )


def test_rejects_file_that_is_not_safetensors(session, tmp_path, device_options):
    """A corrupt weight file is reported as such."""
    tools = Tools(session=session)
    (tmp_path / "model.safetensor").write_bytes(b"\xff" * 64)

    with pytest.raises(ToolError, match="not a valid safetensors"):
        tools.i_sen2sr_opencl(
            output="out",
            model_dir=str(tmp_path),
            **band_options(*RGBN),
            **device_options,
        )


def test_rgbn_model_refuses_unused_band(session, rgbn_model, device_options):
    """Bands the model does not read are refused, not silently dropped."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not used"):
        tools.i_sen2sr_opencl(
            output="out",
            model_dir=rgbn_model,
            **band_options(*RGBN, "swir16"),
            **device_options,
        )


def test_main_model_needs_all_bands(session, main_model, device_options):
    """The main model reads all ten bands."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="rededge1"):
        tools.i_sen2sr_opencl(
            output="out",
            model_dir=main_model,
            **band_options(*RGBN),
            **device_options,
        )


@pytest.fixture(scope="module")
def rgbn_run(session, rgbn_model, device_options):
    """Run the RGBN x4 model once for the result checks."""
    tools = Tools(session=session)
    region = tools.g_region(flags="g", format="json").json
    tools.i_sen2sr_opencl(
        output="sr", model_dir=rgbn_model, **band_options(*RGBN), **device_options
    )
    return session, tools, region


def test_rgbn_region_is_not_changed(rgbn_run):
    """The computational region is left as it was."""
    _, tools, before = rgbn_run
    after = tools.g_region(flags="g", format="json").json

    assert after == before


@pytest.mark.parametrize("band", ["B02", "B03", "B04", "B08"])
def test_rgbn_output_grid(rgbn_run, band):
    """Outputs cover the region with cells four times finer."""
    _, tools, region = rgbn_run
    info = tools.r_info(map=f"sr_{band}", format="json").json

    assert info["rows"] == 4 * ROWS
    assert info["cols"] == 4 * COLS
    assert info["nsres"] == pytest.approx(RES / 4)
    assert info["ewres"] == pytest.approx(RES / 4)
    assert info["north"] == pytest.approx(region["north"])
    assert info["west"] == pytest.approx(region["west"])
    assert info["datatype"] == "FCELL"


@pytest.mark.parametrize("band", ["B02", "B03", "B04", "B08"])
def test_rgbn_keeps_input_units(rgbn_run, band):
    """The hard constraint keeps the low frequencies of the input, so the
    band means agree and the outputs stay in digital numbers."""
    session, tools, _ = rgbn_run
    lr = tools.r_univar(map=f"s2_{band}", format="json").json
    hr = native_univar(session, f"sr_{band}")

    assert hr["mean"] == pytest.approx(lr["mean"], rel=0.02)


def test_rgbn_null_input_cell_gives_null_block(rgbn_run):
    """A null input cell makes the matching 4 x 4 output block null in
    every band, and nothing else."""
    session, tools, _ = rgbn_run
    for band in ("B02", "B03", "B04", "B08"):
        stats = native_univar(session, f"sr_{band}")
        assert stats["null_cells"] == 16, band

    # Input cell (row 10, col 20) covers output rows 37-40, cols 77-80.
    x = 500000 + 19 * RES + RES / 8
    y = 5000000 + ROWS * RES - 9 * RES - RES / 8
    values = tools.r_what(map="sr_B02", coordinates=(x, y), format="json").json
    assert values[0]["sr_B02"]["value"] is None


def test_existing_output_needs_overwrite(rgbn_run, rgbn_model, device_options):
    """Output maps are only replaced with --overwrite."""
    _, tools, _ = rgbn_run

    with pytest.raises(ToolError, match="already exists"):
        tools.i_sen2sr_opencl(
            output="sr",
            model_dir=rgbn_model,
            **band_options(*RGBN),
            **device_options,
        )


def test_main_model_all_bands(session, main_model, device_options):
    """The main model writes all ten bands at 2.5 m."""
    tools = Tools(session=session)
    tools.i_sen2sr_opencl(
        output="all",
        model_dir=main_model,
        **band_options(*BAND_KEYS),
        **device_options,
    )
    for band in BAND_KEYS.values():
        info = tools.r_info(map=f"all_{band}", format="json").json
        assert info["rows"] == 4 * ROWS
        lr = tools.r_univar(map=f"s2_{band}", format="json").json
        hr = native_univar(session, f"all_{band}")
        assert hr["mean"] == pytest.approx(lr["mean"], rel=0.02), band
