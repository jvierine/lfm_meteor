"""Copy the original ENU geometry, not rendered graphics, into scientific HDF5."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "sanya_tristatic_paper/figures/sanya_hainan_system_map_webgl.html"
OUTPUT = ROOT / "results/native_paper_data/system_geometry.h5"


def main():
    text = SOURCE.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT, "w") as h:
        h.attrs["source"] = str(SOURCE)
        h.attrs["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        h.attrs["preparation_program"] = Path(__file__).name
        h.attrs["coordinate_system"] = "Sanya local east/north/up, km"
        for key in ("sites", "target", "meteors", "center", "beamCone", "beamRing", "beamEdges"):
            h[key] = np.asarray(data[key], dtype=float)
        for key in ("beamWidthDeg", "beamWidthScale", "beamAltitudeKm", "radius"):
            h.attrs[key] = data[key]
        h.attrs["trajectory_file"] = data["trajectoryFile"]
        for key in ("coastlines", "gridLines"):
            group = h.create_group(key)
            for i, values in enumerate(data[key]):
                group[str(i)] = np.asarray(values, dtype=float)
        for beam in data["receiveBeams"]:
            group = h.create_group(f"receiver_{beam['siteIndex']}")
            for key in ("cone", "ring", "edges"):
                group[key] = np.asarray(beam[key], dtype=float)
        for kind, ends in (("lonTicks", ("south", "north")), ("latTicks", ("west", "east"))):
            group = h.create_group(kind)
            for i, tick in enumerate(data["tickLabels"][kind]):
                item = group.create_group(str(i))
                item.attrs["degrees"] = float(tick["text"].split()[1])
                for end in ends:
                    item[end] = tick[end]
    print(f"Original 3-D geometry: {len(data['meteors'])} meteor points -> {OUTPUT}")


if __name__ == "__main__":
    main()
