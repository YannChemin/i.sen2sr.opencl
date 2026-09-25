"""Tests for i.sen2sr.opencl."""

import grass.script as gs
import pytest
from conftest import BAND_KEYS, COLS, RES, ROWS, band_options, bundled_model
from grass.tools import ToolError, Tools

RGBN = ("blue", "green", "red", "nir")
TWENTY_M = ("rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22")


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


def test_rgbn_model_refuses_unused_band(session, device_options):
    """Bands the model does not read are refused, not silently dropped."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not used"):
        tools.i_sen2sr_opencl(
            output="out",
            model="rgbn_x4",
            **band_options(*RGBN, "swir16"),
            **device_options,
        )


def test_main_model_needs_all_bands(session, device_options):
    """The main model reads all ten bands."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="rededge1"):
        tools.i_sen2sr_opencl(
            output="out",
            model="main",
            **band_options(*RGBN),
            **device_options,
        )


def test_auto_with_some_20m_bands_asks_for_all(session, device_options):
    """With any 20 m band, auto chooses the main model, which then asks for
    the missing ones instead of dropping the given band."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="main"):
        tools.i_sen2sr_opencl(
            output="out", **band_options(*RGBN, "swir16"), **device_options
        )


def test_model_dir_must_hold_requested_model(session, device_options):
    """Custom weights of another variant than <model> are refused."""
    tools = Tools(session=session)
    rgbn_dir = bundled_model(session.env, "rgbn_x4")

    with pytest.raises(ToolError, match=r"holds\s+the"):
        tools.i_sen2sr_opencl(
            output="out",
            model="main",
            model_dir=rgbn_dir,
            **band_options(*BAND_KEYS),
            **device_options,
        )


@pytest.fixture(scope="module")
def rgbn_run(session, device_options):
    """Run once with the four 10 m bands, for which auto chooses the RGBN
    x4 model, for the result checks."""
    tools = Tools(session=session)
    region = tools.g_region(flags="g", format="json").json
    tools.i_sen2sr_opencl(output="sr", **band_options(*RGBN), **device_options)
    return session, tools, region


def test_auto_with_10m_bands_uses_rgbn(rgbn_run):
    """Auto with only the 10 m bands writes exactly those four bands."""
    _, tools, _ = rgbn_run
    maps = tools.g_list(type="raster", pattern="sr_*", format="json").json

    assert sorted(m["name"] for m in maps) == ["sr_B02", "sr_B03", "sr_B04", "sr_B08"]


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


def test_existing_output_needs_overwrite(rgbn_run, device_options):
    """Output maps are only replaced with --overwrite."""
    _, tools, _ = rgbn_run

    with pytest.raises(ToolError, match="already exists"):
        tools.i_sen2sr_opencl(output="sr", **band_options(*RGBN), **device_options)


def test_main_model_all_bands(session, device_options):
    """With all ten bands, auto chooses the main model, which writes all
    ten bands at 2.5 m."""
    tools = Tools(session=session)
    tools.i_sen2sr_opencl(
        output="all",
        **band_options(*BAND_KEYS),
        **device_options,
    )
    for band in BAND_KEYS.values():
        info = tools.r_info(map=f"all_{band}", format="json").json
        assert info["rows"] == 4 * ROWS
        lr = tools.r_univar(map=f"s2_{band}", format="json").json
        hr = native_univar(session, f"all_{band}")
        assert hr["mean"] == pytest.approx(lr["mean"], rel=0.02), band


def test_rswir_model_on_request(session, device_options):
    """model=rswir_x2 writes the six 20 m bands on the 10 m region grid."""
    tools = Tools(session=session)
    tools.i_sen2sr_opencl(
        output="ten",
        model="rswir_x2",
        **band_options(*BAND_KEYS),
        **device_options,
    )
    maps = tools.g_list(type="raster", pattern="ten_*", format="json").json
    assert sorted(m["name"] for m in maps) == sorted(
        f"ten_{BAND_KEYS[key]}" for key in TWENTY_M
    )
    for key in TWENTY_M:
        band = BAND_KEYS[key]
        info = tools.r_info(map=f"ten_{band}", format="json").json
        assert info["rows"] == ROWS
        assert info["cols"] == COLS
        lr = tools.r_univar(map=f"s2_{band}", format="json").json
        hr = native_univar(session, f"ten_{band}")
        assert hr["mean"] == pytest.approx(lr["mean"], rel=0.02), band


def test_custom_model_dir_is_used(session, device_options):
    """Weights given with model_dir replace the installed ones."""
    tools = Tools(session=session)
    tools.i_sen2sr_opencl(
        output="custom",
        model_dir=bundled_model(session.env, "rgbn_x4"),
        **band_options(*RGBN),
        **device_options,
    )
    info = tools.r_info(map="custom_B02", format="json").json
    assert info["rows"] == 4 * ROWS
