"""Round-trip .aseprite files through a minimal independent reader of the spec."""
import struct
import zlib

import numpy as np

from pixel_pipeline.aseprite import write_aseprite


def read_ase(path):
    data = path.read_bytes()
    (size, magic, frames, w, h, depth, _flags, _speed, _z1, _z2, transparent, ncolors,
     *_rest) = struct.unpack_from("<IHHHHHIHIIB3xHBBhhHH84x", data, 0)
    assert magic == 0xA5E0 and size == len(data)
    out = {"w": w, "h": h, "depth": depth, "transparent": transparent, "ncolors": ncolors,
           "frames": [], "palette": None, "layers": [], "tags": []}
    pos = 128
    for _ in range(frames):
        fsize, fmagic, _old, duration, nchunks = struct.unpack_from("<IHHH2xI", data, pos)
        assert fmagic == 0xF1FA
        frame = {"duration": duration, "cels": []}
        cpos = pos + 16
        for _ in range(nchunks):
            csize, ctype = struct.unpack_from("<IH", data, cpos)
            body = data[cpos + 6:cpos + csize]
            if ctype == 0x2019:
                n, first, last = struct.unpack_from("<III", body, 0)
                entries = [struct.unpack_from("<HBBBB", body, 20 + 6 * i)[1:]
                           for i in range(last - first + 1)]
                out["palette"] = entries
            elif ctype == 0x2004:
                (name_len,) = struct.unpack_from("<H", body, 16)
                out["layers"].append(body[18:18 + name_len].decode())
            elif ctype == 0x2005:
                layer, x, y, opacity, kind, _z = struct.unpack_from("<HhhBHh5x", body, 0)
                cw, ch = struct.unpack_from("<HH", body, 16)
                assert kind == 2
                frame["cels"].append((layer, x, y, cw, ch, zlib.decompress(body[20:])))
            elif ctype == 0x2018:
                (ntags,) = struct.unpack_from("<H", body, 0)
                tpos = 10
                for _ in range(ntags):
                    start, end, direction = struct.unpack_from("<HHB", body, tpos)
                    (name_len,) = struct.unpack_from("<H", body, tpos + 17)
                    name = body[tpos + 19:tpos + 19 + name_len].decode()
                    out["tags"].append((name, start, end, direction))
                    tpos += 19 + name_len
            cpos += csize
        assert cpos == pos + fsize
        out["frames"].append(frame)
        pos += fsize
    return out


def _frames():
    frames = []
    for i in range(3):
        f = np.zeros((6, 9, 4), np.uint8)
        f[1:5, i:i + 4] = [255, 0, 77, 255]
        f[0, 0] = [41, 173, 255, 255]
        frames.append(f)
    return frames


def test_indexed_animation_round_trip(tmp_path):
    frames = _frames()
    pal = np.array([[255, 0, 77], [41, 173, 255], [0, 0, 0]], np.uint8)
    path = tmp_path / "anim.aseprite"
    write_aseprite(path, frames, pal, durations=[100, 120, 140], tags=[("walk", 0, 2)])
    ase = read_ase(path)
    assert (ase["w"], ase["h"], ase["depth"], ase["transparent"]) == (9, 6, 8, 0)
    assert ase["palette"] == [(0, 0, 0, 0), (255, 0, 77, 255), (41, 173, 255, 255), (0, 0, 0, 255)]
    assert ase["layers"] == ["Sprite"]
    assert ase["tags"] == [("walk", 0, 2, 0)]
    assert [f["duration"] for f in ase["frames"]] == [100, 120, 140]
    for f, frame in zip(frames, ase["frames"]):
        (_layer, _x, _y, cw, ch, raw), = frame["cels"]
        idx = np.frombuffer(raw, np.uint8).reshape(ch, cw)
        expected = np.zeros((6, 9), np.uint8)
        expected[(f[..., :3] == [255, 0, 77]).all(-1) & (f[..., 3] == 255)] = 1
        expected[(f[..., :3] == [41, 173, 255]).all(-1)] = 2
        assert np.array_equal(idx, expected)


def test_rgba_fallback_for_soft_alpha(tmp_path):
    f = _frames()[0]
    f[5, 8] = [10, 20, 30, 128]
    path = tmp_path / "soft.aseprite"
    write_aseprite(path, [f])
    ase = read_ase(path)
    assert ase["depth"] == 32
    (_layer, _x, _y, cw, ch, raw), = ase["frames"][0]["cels"]
    assert np.array_equal(np.frombuffer(raw, np.uint8).reshape(ch, cw, 4), f)


def test_colour_outside_palette_falls_back_to_rgba(tmp_path):
    path = tmp_path / "x.aseprite"
    write_aseprite(path, [_frames()[0]], palette=np.array([[1, 2, 3]], np.uint8))
    assert read_ase(path)["depth"] == 32
