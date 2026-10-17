use super::common::*;
use anyhow::Result;
use rustplotlib::*;

fn xyz(h: &hdf5::Group, name: &str) -> Result<Vec<[f64; 3]>> {
    Ok(array(h, name)?
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect())
}
fn screen(c: &PerspectiveCamera, p: [f64; 3]) -> Option<[f64; 2]> {
    c.project(p)
        .map(|v| [v[0] + 0.162552189, 1. - (v[1] - 0.320086549)])
}
fn path(
    p: &mut Panel,
    c: &PerspectiveCamera,
    pts: &[[f64; 3]],
    color: &str,
    width: f64,
    opacity: f64,
) {
    let mut l = Line::new(
        pts.iter()
            .map(|&v| screen(c, v).unwrap_or([f64::NAN; 2]))
            .collect(),
        color,
    );
    l.width_pt = width;
    l.opacity = opacity;
    p.push(Layer::Line(l));
}
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let width = 200.;
    let height = width * 946. / 1706.;
    let center = xyz(h, "center")?[0];
    // Reference-view reconstruction from the three station markers and common volume.
    let (yaw, pitch, roll, dist): (f64, f64, f64, f64) =
        (-1.06463159, 0.200315451, 0.0169415374, 249.232042);
    let z = [
        pitch.cos() * yaw.sin(),
        -pitch.cos() * yaw.cos(),
        pitch.sin(),
    ];
    let right = [yaw.cos(), yaw.sin(), 0.];
    let up = [
        -z[2] * right[1],
        z[2] * right[0],
        z[0] * right[1] - z[1] * right[0],
    ];
    let rolled = std::array::from_fn(|i| up[i] * roll.cos() + right[i] * roll.sin());
    let camera = PerspectiveCamera::new(
        std::array::from_fn(|i| center[i] + dist * z[i]),
        center,
        rolled,
        42.,
        width / height,
    )?;
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: width,
        font_pt: 10. * width / 184.,
    });
    f.height_mm = height;
    let mut p = Panel::new(
        Rect::new(0., 0., width, height),
        Axes::new([0., 1.], [0., 1.], "", ""),
    );
    p.axes_visible = false;
    for (name, color, line_width) in [
        ("gridLines", "#dde6ee", 0.25),
        ("coastlines", "#8c8475", 0.35),
    ] {
        let g = h.group(name)?;
        for name in g.member_names()? {
            path(&mut p, &camera, &xyz(&g, &name)?, color, line_width, 1.);
            if color == "#dde6ee" {
                if let Some(Layer::Line(line)) = p.layers.last_mut() {
                    line.protect_from_legend = false;
                }
            }
        }
    }
    let mut triangles: Vec<(f64, [[f64; 2]; 3], &str)> = Vec::new();
    for (group, name, color) in [
        (h.group("/")?, "beamCone", "#7a23a3"),
        (h.group("receiver_1")?, "cone", "#1671c4"),
        (h.group("receiver_2")?, "cone", "#378541"),
    ] {
        for triangle in xyz(&group, name)?.chunks_exact(3) {
            if let (Some(a), Some(b), Some(d)) = (
                screen(&camera, triangle[0]),
                screen(&camera, triangle[1]),
                screen(&camera, triangle[2]),
            ) {
                let depth = triangle
                    .iter()
                    .map(|&v| camera.project(v).unwrap()[2])
                    .sum::<f64>()
                    / 3.;
                triangles.push((depth, [a, b, d], color));
            }
        }
    }
    triangles.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, v, color) in triangles {
        let mut l = Line::new(vec![v[0], v[1], v[2], v[0]], color);
        l.width_pt = 0.;
        l.fill = Some(color.into());
        l.fill_opacity = 0.12;
        p.push(Layer::Line(l));
    }
    let sites = xyz(h, "sites")?;
    let target = xyz(h, "target")?[0];
    for (i, color) in ["#7a23a3", "#1671c4", "#378541"].iter().enumerate() {
        let g = if i == 0 {
            h.group("/")?
        } else {
            h.group(&format!("receiver_{i}"))?
        };
        path(
            &mut p,
            &camera,
            &xyz(&g, if i == 0 { "beamRing" } else { "ring" })?,
            color,
            0.45,
            0.65,
        );
        for edge in xyz(&g, if i == 0 { "beamEdges" } else { "edges" })?.chunks_exact(2) {
            path(&mut p, &camera, edge, color, 0.35, 0.55);
        }
        path(&mut p, &camera, &[sites[i], target], color, 0.5, 0.7);
    }
    let mut meteors: Vec<_> = xyz(h, "meteors")?
        .into_iter()
        .filter_map(|v| camera.project(v).map(|q| (q[2], v)))
        .collect();
    meteors.sort_by(|a, b| b.0.total_cmp(&a.0));
    let pts: Vec<_> = meteors
        .iter()
        .filter_map(|(_, v)| screen(&camera, *v))
        .collect();
    p.scatter_colored(
        &pts,
        &vec![0.5; pts.len()],
        &solid_scale("#dc680e"),
        1.55,
        0.55,
        "none",
    )?;
    for (i, (name, color)) in [
        ("Sanya TX/RX", "#c72828"),
        ("Danzhou RX", "#1469b8"),
        ("Wenchang RX", "#2d8035"),
    ]
    .iter()
    .enumerate()
    {
        let q = screen(&camera, sites[i]).unwrap();
        p.scatter_colored(&[q], &[0.5], &solid_scale(color), 3.5, 1., "none")?;
        let index = f.annotate(&p, format!("\\textbf{{{name}}}"), q)?;
        f.labels[index].color = color.to_string();
    }
    for (kind, end, suffix) in [("lonTicks", "north", "E"), ("latTicks", "east", "N")] {
        let g = h.group(kind)?;
        for name in g.member_names()? {
            let item = g.group(&name)?;
            let degrees = item.attr("degrees")?.read_scalar::<f64>()?;
            // Keep the same sparse geographic references as the original view.
            if (kind == "lonTicks" && degrees > 109.5) || (kind == "latTicks" && degrees > 18.5) {
                continue;
            }
            let q = screen(&camera, xyz(&item, end)?[0]).unwrap();
            let y = 1. - q[1];
            if q[0] > 0.06 && q[0] < 0.93 && y > 0.08 && y < 0.88 {
                let mut l = Label::new(
                    format!("${degrees:.1}^\\circ\\,\\mathrm{{{suffix}}}$"),
                    q[0] * width,
                    y * height - 1.,
                );
                l.color = "#454545".into();
                f.labels.push(l);
            }
        }
    }
    f.panels.push(p);
    article_figure(f, 1.)
}
