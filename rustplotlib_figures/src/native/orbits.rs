use super::common::*;
use anyhow::Result;
use hdf5::types::VarLenUnicode;
use rustplotlib::*;
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let names = h.dataset("event_id")?.read_raw::<VarLenUnicode>()?;
    let a = array(h, "a_au")?;
    let e = array(h, "e")?;
    let inc = array(h, "i_deg")?;
    let period = array(h, "period_years")?;
    let n = h.dataset("a_au")?.shape()[1];
    let summary = |name: &str, scale: f64| -> Result<Vec<[f64; 3]>> {
        Ok(array(h, name)?
            .chunks(n)
            .map(|v| {
                let values: Vec<_> = v.iter().map(|v| v / scale).collect();
                [
                    quantile(&values, 0.5),
                    quantile(&values, 0.025),
                    quantile(&values, 0.975),
                ]
            })
            .collect())
    };
    let aa = summary("a_au", 1.)?;
    let ee = summary("e", 1.)?;
    let ii = summary("i_deg", 1.)?;
    let qq = summary("q_au", 1.)?;
    let ap = summary("Q_au", 1.)?;
    let energy = summary("specific_energy_j_kg", 1e6)?;
    let keep: Vec<_> = (0..names.len())
        .filter(|&i| {
            ee[i][0] >= 1.
                || (ap[i].iter().all(|v| v.is_finite())
                    && ap[i][2] - ap[i][1] <= 20.
                    && energy[i][2] - energy[i][1] <= 100.)
        })
        .collect();
    anyhow::ensure!(!keep.is_empty(), "No finite orbit intervals for display");
    let class: Vec<_> = (0..names.len())
        .map(|row| {
            let mut votes = 0.;
            let mut total = 0.;
            for k in row * n..(row + 1) * n {
                if ![a[k], e[k], inc[k]]
                    .iter()
                    .all(|v| v.is_finite())
                {
                    continue;
                }
                let high = inc[k] > 75. || e[k] >= 1.;
                let tj = 1. / a[k]
                    + 2. * 5.2038_f64.powf(-1.5)
                        * (a[k] * (1. - e[k] * e[k])).max(0.).sqrt()
                        * inc[k].to_radians().cos();
                let kk = (a[k] * (1. + e[k]) / (1. - e[k])).ln() - 1.;
                let eps = -0.01720209895_f64.powi(2) / (2. * a[k]);
                for flag in [
                    tj < 0.58,
                    kk > 0.,
                    a[k] * (1. + e[k]) > 4.6,
                    period[k] * e[k] > 2.5,
                    eps > -5.28e-5,
                ] {
                    votes += if flag || high { 1. } else { 0. };
                    total += 1.;
                }
            }
            total > 0. && votes / total >= 0.5
        })
        .collect();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.height_mm = 132.;
    f.font_pt = 8.;
    f.latex.font_pt = 8.;
    let panels = [
        (
            &aa,
            &ee,
            r"Semimajor axis $a$ (AU)",
            r"Eccentricity $e$",
            false,
        ),
        (
            &qq,
            &ii,
            r"Perihelion distance $q$ (AU)",
            r"Inclination $i$ (deg)",
            false,
        ),
        (
            &ap,
            &ii,
            r"Aphelion distance $Q$ (AU)",
            r"Inclination $i$ (deg)",
            true,
        ),
        (
            &energy,
            &ii,
            r"Specific orbital energy $\varepsilon$ ($\mathrm{km^2\,s^{-2}}$)",
            r"Inclination $i$ (deg)",
            false,
        ),
    ];
    for (index, (xx, yy, xlabel, ylabel, log)) in panels.iter().enumerate() {
        let good: Vec<_> = keep
            .iter()
            .copied()
            .filter(|&i| {
                xx[i].iter().chain(&yy[i]).all(|v| v.is_finite()) && (!log || xx[i][1] > 0.)
            })
            .collect();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for &i in &good {
            xv.extend(xx[i]);
            yv.extend(yy[i]);
        }
        let xr = if *log {
            let v: Vec<_> = xv.iter().map(|v| v.log10()).collect();
            limits(&v, 0.06).map(|v| 10_f64.powf(v))
        } else {
            limits(&xv, 0.06)
        };
        let mut axes = Axes::new(xr, limits(&yv, 0.06), *xlabel, *ylabel);
        if *log {
            axes.set_xscale(AxisScale::Log10)?;
        }
        if index > 0 {
            axes.ylim = [-5., 185.];
            axes.yticks = vec![0., 45., 90., 135., 180.];
        }
        let mut p = Panel::new(
            Rect::new(
                20. + (index % 2) as f64 * 82.,
                10. + (index / 2) as f64 * 61.,
                66.,
                48.,
            ),
            axes,
        );
        p.major_grid("#e2e2e2", 0.4);
        for (category, color) in [(1, COLORS[1]), (0, COLORS[2]), (2, "#222222")] {
            let ids: Vec<_> = good
                .iter()
                .copied()
                .filter(|&i| (if ee[i][0] >= 1. { 2 } else if class[i] { 1 } else { 0 }) == category)
                .collect();
            let x: Vec<_> = ids.iter().map(|&i| xx[i][0]).collect();
            let y: Vec<_> = ids.iter().map(|&i| yy[i][0]).collect();
            let xb: Vec<_> = ids
                .iter()
                .map(|&i| [xx[i][1].min(xx[i][0]), xx[i][2].max(xx[i][0])])
                .collect();
            let yb: Vec<_> = ids
                .iter()
                .map(|&i| [yy[i][1].min(yy[i][0]), yy[i][2].max(yy[i][0])])
                .collect();
            p.errorbars(
                &points(&x, &y),
                Some(&xb),
                Some(&yb),
                "#666666",
                0.45,
                0.,
                0.22,
            )?;
            if category == 2 {
                p.markers(&mut f, &points(&x, &y), &MarkerStyle {
                    shape: MarkerShape::Triangle, size_pt: 5., color: [0.13, 0.13, 0.13],
                })?;
            } else {
                dots(&mut p, &x, &y, color, 3., 0.82, false)?;
            }
        }
        if index == 0 {
            let mut l = Legend::new(vec![
                LegendEntry::circle("Cometary", COLORS[1]),
                LegendEntry::circle("Asteroidal", COLORS[2]),
                LegendEntry::marker("Hyperbolic", MarkerShape::Triangle, "#222222"),
            ]);
            l.allow_reused_marker_shapes = true;
            p.legend = Some(l);
        }
        panel_label(&mut f, &p, &format!("({})", (b'a' + index as u8) as char));
        f.panels.push(p);
    }
    article_figure(f, 1.)
}
