"""Build a local, lazy-loading index of the generated November figures."""
import argparse
from html import escape
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    figures = [(2, "Detections and receiver noise"), (3, "Heights and beam coverage"),
               (4, "Path residuals"), (7, "Dynamic masses"),
               (8, "Radar cross sections and mass comparison"), (9, "Sun-centred radiants"),
               (10, "Heliocentric orbits")]
    sections = []
    for number, name in figures:
        png = root / ("figure%d.png" % number)
        if png.exists():
            sections.append('<section id="figure%d"><h2>Figure %d — %s</h2>'
                            '<a href="figure%d.pdf"><img src="%s" loading="lazy" alt="%s"></a></section>'
                            % (number, number, escape(name), number, png.name, escape(name)))
    events = sorted((root / "events").glob("tri_*/event.png"))
    sections.append('<h2 id="events">Individual events (%d rendered)</h2><div class="events">' % len(events))
    for png in events:
        path = png.relative_to(root).as_posix()
        sections.append('<section><h3>%s</h3><a href="%s"><img src="%s" loading="lazy" alt="%s"></a></section>'
                        % (png.parent.name, path.replace(".png", ".pdf"), path, png.parent.name))
    sections.append('</div>')
    text = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>November SYISR figures</title><style>
body{font:16px system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1280px;color:#202020;background:white}
nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#006a9c}h1{font-size:26px}h2{font-size:21px;margin-top:36px}
img{display:block;width:100%;height:auto}section{scroll-margin-top:20px}.events{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}h3{font-size:16px}
</style><h1>17 November 2026: SYISR tri-static observations</h1>
<p>Provisional fits. Hardware-delay and absolute RCS calibration remain unverified; the orbit and mass plots include candidates requiring further review.</p>
<nav>''' + ''.join('<a href="#figure%d">Figure %d</a>' % (n, n) for n, _ in figures)
    text += '<a href="#events">Events</a></nav>' + '\n'.join(sections) + '</html>\n'
    (root / "index.html").write_text(text)
    print("Gallery:", root / "index.html", "events:", len(events))


if __name__ == "__main__":
    main()
