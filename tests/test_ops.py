import numpy as np
import pytest
from chunkmirage.ops import op_from_spec
from zarr_proxy.ops import (
    CTWindow,
    Dilate,
    DynamicRange,
    Erode,
    Frangi,
    GammaCorrection,
    Identity,
    Laplace,
    Li,
    Median,
    Meijering,
    Otsu,
    Passthrough,
    Sato,
    Scharr,
    Sobel,
    UnsharpMask,
    WhiteTopHat,
)


def test_frangi_op():
    op = Frangi(sigmas=[1.0, 2.0])
    assert op.name == "frangi"
    assert op.halo == 6  # ceil(3 * 2.0)
    assert op.output_dtype(np.dtype("uint8")) == np.dtype("float32")

    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[16, :, 16] = 100.0

    filtered = op.apply(vol)
    assert filtered.shape == (32, 32, 32)
    assert filtered.dtype == np.float32
    assert filtered[16, 16, 16] > filtered[0, 0, 0]


def test_sato_op():
    op = Sato(sigmas=[1.0, 2.0])
    assert op.name == "sato"
    assert op.halo == 6

    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[16, :, 16] = 100.0

    filtered = op.apply(vol)
    assert filtered.shape == (32, 32, 32)
    assert filtered.dtype == np.float32
    assert filtered[16, 16, 16] > filtered[0, 0, 0]


def test_meijering_op():
    op = Meijering(sigmas=[1.0])
    assert op.name == "meijering"
    assert op.halo == 3

    vol = np.zeros((24, 24, 24), dtype=np.float32)
    vol[12, :, 12] = 100.0
    res = op.apply(vol)
    assert res.shape == (24, 24, 24)
    assert res.dtype == np.float32
    assert res[12, 12, 12] > res[0, 0, 0]


def test_sobel_and_scharr_ops():
    sobel = Sobel()
    assert sobel.name == "sobel"
    assert sobel.halo == 1

    scharr = Scharr()
    assert scharr.name == "scharr"
    assert scharr.halo == 1

    # Step function boundary
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[:, :, 8:] = 100.0

    s_out = sobel.apply(vol)
    assert s_out.shape == (16, 16, 16)
    assert s_out[:, :, 7].max() > 0 or s_out[:, :, 8].max() > 0

    sc_out = scharr.apply(vol)
    assert sc_out.shape == (16, 16, 16)
    assert sc_out[:, :, 7].max() > 0 or sc_out[:, :, 8].max() > 0


def test_laplace_and_unsharp_mask():
    lap = Laplace()
    assert lap.name == "laplace"
    assert lap.halo == 1

    unsharp = UnsharpMask(radius=1.5, amount=2.0)
    assert unsharp.name == "unsharp_mask"
    assert unsharp.halo == 5

    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[8, 8, 8] = 50.0

    assert lap.apply(vol).shape == (16, 16, 16)
    u_out = unsharp.apply(vol)
    assert u_out.shape == (16, 16, 16)
    assert u_out[8, 8, 8] > 50.0  # sharpened impulse center is amplified


def test_morphology_ops():
    vol = np.zeros((15, 15, 15), dtype=np.uint8)
    vol[7, 7, 7] = 200  # single bright spike

    # Median should remove single spike
    med = Median(size=3)
    assert med.halo == 1
    m_out = med.apply(vol)
    assert m_out[7, 7, 7] == 0

    # Dilate should expand it
    dil = Dilate(radius=1)
    assert dil.halo == 1
    d_out = dil.apply(vol)
    assert d_out[6, 7, 7] == 200
    assert d_out[8, 7, 7] == 200

    # White top-hat should isolate it
    wth = WhiteTopHat(radius=2)
    assert wth.halo == 2
    w_out = wth.apply(vol)
    assert w_out[7, 7, 7] == 200


def test_intensity_and_windowing_ops():
    ct = CTWindow(vmin=18000.0, vmax=26000.0, target_dtype="uint8")
    assert ct.name == "ct_window"
    assert ct.output_dtype(np.dtype("uint16")) == np.dtype("uint8")

    data = np.array([18000, 22000, 26000, 30000], dtype=np.uint16)
    out = ct.apply(data)
    assert out[0] == 0
    assert 120 <= out[1] <= 135
    assert out[2] == 255
    assert out[3] == 255  # clipped

    gamma = GammaCorrection(gamma=2.0)
    g_out = gamma.apply(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.isclose(g_out[0], 0.0)
    assert np.isclose(g_out[1], 0.25)
    assert np.isclose(g_out[2], 1.0)


def test_segmentation_ops():
    # Otsu
    otsu = Otsu(value=1)
    assert otsu.name == "otsu"
    assert otsu.halo == 4

    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[:10] = 10.0
    vol[10:] = 100.0
    mask_o = otsu.apply(vol)
    assert np.all(mask_o[:10] == 0)
    assert np.all(mask_o[10:] == 1)

    # Li
    li = Li(value=2)
    assert li.name == "li"
    assert li.halo == 4
    mask_li = li.apply(vol)
    assert np.all(mask_li[:10] == 0)
    assert np.all(mask_li[10:] == 2)


def test_passthrough_ops():
    p = Passthrough()
    assert p.name == "passthrough"
    assert p.halo == 0
    data = np.arange(10, dtype=np.uint16)
    assert np.array_equal(p.apply(data), data)

    ident = Identity()
    assert ident.name == "identity"
    assert np.array_equal(ident.apply(data), data)


def test_ops_discoverable_via_chunkmirage():
    for op_name, expected_type in [
        ("sato", Sato),
        ("frangi", Frangi),
        ("meijering", Meijering),
        ("sobel", Sobel),
        ("scharr", Scharr),
        ("laplace", Laplace),
        ("unsharp_mask", UnsharpMask),
        ("median", Median),
        ("erode", Erode),
        ("dilate", Dilate),
        ("white_tophat", WhiteTopHat),
        ("ct_window", CTWindow),
        ("gamma", GammaCorrection),
        ("otsu", Otsu),
        ("li", Li),
        ("passthrough", Passthrough),
    ]:
        op_inst = op_from_spec({"op": op_name})
        assert isinstance(op_inst, expected_type)
