use super::common::*;
use anyhow::Result;
use rustplotlib::*;
fn attr(g: &hdf5::Group, name: &str) -> f64 {
    g.attr(name)
        .and_then(|a| a.read_scalar::<f64>())
        .unwrap_or(f64::NAN)
}
fn mask(g: &hdf5::Group, name: &str) -> Result<Vec<bool>> {
    Ok(g.dataset(name)?.read_raw::<bool>()?)
}
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let wavelength = 299792458.0 / 430.0e6;
    let j = h.group("joint_fit")?;
    let d = h.group("derived")?;
    let t = array(&j, "t_rel_s")?;
    let tr = limits(&t, 0.02);
    let display_t = array(&j, "all_t_rel_s").unwrap_or(t.clone());
    let snr =
        array(&j, "all_snr_db").or_else(|_| array(&h.group("fft_observations")?, "fft_snr_db"))?;
    let east = array(&d, "beam_east_deg")?;
    let north = array(&d, "beam_north_deg")?;
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 222.,
        font_pt: 9.,
    });
    f.height_mm = 252.;
    let left = 21.;
    let right = 135.;
    let rows = [12., 92., 172.];
    let width = 65.;
    let height = 65.;
    let speed_scale = Scale::new(
        limits(&snr.iter().step_by(3).copied().collect::<Vec<_>>(), 0.01),
        Normalization::Linear,
        ColorMap::turbo(),
    )?;
    let cid = f.scale(speed_scale.clone());
    let mut a = Axes::new(
        [-2.6, 2.6],
        [-2.6, 2.6],
        "East offset (deg)",
        "North offset (deg)",
    );
    a.xticks = vec![-2., 0., 2.];
    a.yticks = vec![-2., 0., 2.];
    let mut p = Panel::new(Rect::new(left, rows[0], height, height), a);
    let grid = array(&d, "gain_grid_deg")?;
    let gain = array(&d, "gain_db")?;
    for level in [-30., -20., -13., -10., -3.] {
        let c = Contour::new(
            grid.clone(),
            grid.clone(),
            gain.clone(),
            vec![level, level + 1e-9],
            cid,
            false,
        )?;
        let mut pts = Vec::new();
        for s in c.segments(level) {
            pts.extend([s[0], s[1], [f64::NAN; 2]]);
        }
        let mut l = Line::new(pts, "#999999");
        l.width_pt = 0.5;
        p.push(Layer::Line(l));
    }
    curve(&mut p, &east, &north, "#3f7657", 1.2);
    let keep = mask(&j, "all_main_lobe_fit_mask").unwrap_or(vec![true; east.len()]);
    let pathmask = mask(&j, "all_path_keep").unwrap_or(vec![true; east.len() * 3]);
    for used in [false, true] {
        let ids: Vec<_> = (0..east.len())
            .filter(|&i| keep[i] == used && pathmask[i * 3])
            .collect();
        let x: Vec<_> = ids.iter().map(|&i| east[i]).collect();
        let y: Vec<_> = ids.iter().map(|&i| north[i]).collect();
        if used {
            let vals: Vec<_> = ids.iter().map(|&i| snr[i * 3]).collect();
            p.scatter_colored(&points(&x, &y), &vals, &speed_scale, 4., 0.9, "none")?;
        } else {
            dots(&mut p, &x, &y, "#888888", 3.5, 0.35, true)?;
        }
    }
    if let Ok(b) = d.group("full_beam_band") {
        let elo = array(&b, "east_lo_deg")?;
        let ehi = array(&b, "east_hi_deg")?;
        let nlo = array(&b, "north_lo_deg")?;
        let nhi = array(&b, "north_hi_deg")?;
        let ids: Vec<_> = (0..east.len()).step_by((east.len() / 12).max(1)).collect();
        let pts: Vec<_> = ids.iter().map(|&i| [east[i], north[i]]).collect();
        let xb: Vec<_> = ids
            .iter()
            .map(|&i| [elo[i].min(east[i]), ehi[i].max(east[i])])
            .collect();
        let yb: Vec<_> = ids
            .iter()
            .map(|&i| [nlo[i].min(north[i]), nhi[i].max(north[i])])
            .collect();
        p.errorbars(&pts, Some(&xb), Some(&yb), "#888888", 0.6, 2., 0.4)?;
    }
    panel_label(&mut f, &p, "(a)");
    f.panels.push(p);
    let ticks = (speed_scale.limits[0].div_euclid(10.) as i32 + 1
        ..=speed_scale.limits[1].div_euclid(10.) as i32)
        .map(|i| i as f64 * 10.)
        .collect();
    f.colorbars.push(Colorbar::new(
        Rect::new(left + height + 4., rows[0], 2., height),
        cid,
        "SNR (dB)",
        ticks,
    ));
    let velocity = array(&d, "along_velocity_km_s")?;
    let bands = d.group("bands")?;
    let vlo = array(&bands, "along_velocity_lo_km_s")?;
    let vhi = array(&bands, "along_velocity_hi_km_s")?;
    let mut vv = vlo.clone();
    vv.extend(&vhi);
    let a = Axes::new(
        tr,
        limits(&vv, 0.08),
        "",
        r"Along-track velocity ($\mathrm{km\,s^{-1}}$)",
    );
    let mut p = Panel::new(Rect::new(right, rows[0], width, height), a);
    p.major_grid("#e5e5e5", 0.4);
    p.uncertainty_band(&t, &vlo, &vhi, "#b8b8b8", 0.42)?;
    curve(&mut p, &t, &velocity, "#222222", 1.3);
    panel_label(&mut f, &p, "(b)");
    f.panels.push(p);
    let params = array(&j, "params")?;
    let covariance = array(&j, "parameter_covariance")?;
    let dim = j.dataset("parameter_covariance")?.shape()[0];
    let norm = params[3..6].iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut variance = 0.;
    for a in 0..3 {
        for b in 0..3 {
            variance +=
                params[3 + a] / norm * params[3 + b] / norm * covariance[(3 + a) * dim + 3 + b];
        }
    }
    f.labels.push(Label::new(
        format!(
            "$|\\mathbf{{v}}_0|={:.2}\\pm{:.3}\\,\\mathrm{{km\\,s^{{-1}}}}$",
            norm / 1000.,
            variance.max(0.).sqrt() / 1000.
        ),
        right + width / 2.,
        rows[0] + 6.,
    ));
    let radius = array(&j, "radius_profile_radius_m")?
        .iter()
        .map(|r| r * 1e6)
        .collect::<Vec<_>>();
    let mut prob = array(&j, "radius_profile_probability_density")?;
    let max = prob.iter().copied().fold(0_f64, f64::max);
    for v in &mut prob {
        *v /= max;
    }
    let mut a = Axes::new(
        [radius[0], *radius.last().unwrap()],
        [0., 2.7],
        r"Initial radius $r_0$ ($\mu\mathrm{m}$)",
        "Relative probability",
    );
    a.set_xscale(AxisScale::Log10)?;
    a.yticks = vec![0., 0.5, 1.];
    let mut p = Panel::new(Rect::new(left, rows[1], width, height), a);
    let r = attr(&j, "radius_profile_best_radius_m");
    let rl = attr(&j, "radius_profile_lo95_radius_m");
    let rh = attr(&j, "radius_profile_hi95_radius_m");
    rect(&mut p, [rl * 1e6, rh * 1e6], [0., 1.05], "#aaaaaa", 0.18);
    p.uncertainty_band(&radius, &vec![0.; radius.len()], &prob, "#aaaaaa", 0.42)?;
    curve(&mut p, &radius, &prob, "#222222", 1.2);
    panel_label(&mut f, &p, "(c)");
    f.panels.push(p);
    let mass = |r: f64| 4. / 3. * std::f64::consts::PI * 3000. * r.powi(3);
    let note = scientific_text(&format!(
        "$\\begin{{gathered}}r_0={:.1}\\,\\mu\\mathrm{{m}}\\\\95\\%:[{:.1},{:.1}]\\,\\mu\\mathrm{{m}}\\\\m_0={:.2e}\\,\\mathrm{{kg}}\\\\95\\%:[{:.2e},\\\\{:.2e}]\\,\\mathrm{{kg}}\\end{{gathered}}$",
        r * 1e6,
        rl * 1e6,
        rh * 1e6,
        mass(r),
        mass(rl),
        mass(rh)
    ));
    f.labels.push(Label::new(
        note,
        left + width / 2.,
        rows[1] + 22.,
    ));
    let alt = array(&j, "alt_km")?;
    let alo = array(&bands, "alt_lo_km")?;
    let ahi = array(&bands, "alt_hi_km")?;
    let mut av = alo.clone();
    av.extend(&ahi);
    let mut p = Panel::new(
        Rect::new(right, rows[1], width, height),
        Axes::new(tr, limits(&av, 0.08), "", "Altitude (km)"),
    );
    p.major_grid("#e5e5e5", 0.4);
    p.uncertainty_band(&t, &alo, &ahi, "#b8b8b8", 0.42)?;
    curve(&mut p, &t, &alt, "#222222", 1.3);
    panel_label(&mut f, &p, "(d)");
    f.panels.push(p);
    for frequency in [false, true] {
        let residual = array(
            &j,
            if frequency {
                "fft_residuals_hz"
            } else {
                "path_residuals_m"
            },
        )?;
        let sigma = array(
            &j,
            if frequency {
                "fft_sigma_hz"
            } else {
                "path_sigma_m"
            },
        )?;
        let keep = mask(&j, if frequency { "fft_keep" } else { "path_keep" })?;
        let factor = if frequency { 0.001 } else { 1. };
        let bound = if frequency {
            1000. / wavelength / 1000.
        } else {
            100.
        };
        let mut a = Axes::new(
            tr,
            [-bound, bound],
            "Time (s)",
            if frequency {
                "Beat-frequency residual (kHz)"
            } else {
                "Delay residual (m)"
            },
        );
        a.yticks = if frequency {
            vec![-0.5, 0., 0.5]
        } else {
            vec![-100., -50., 0., 50., 100.]
        };
        if frequency {
            a.secondary_y = Some(SecondaryAxis::new(
                r"Path-rate residual ($\mathrm{m\,s^{-1}}$)",
                vec![-1000., -500., 0., 500., 1000.],
                |v| -v / wavelength / 1000.,
            )?);
        } else {
            let sr = limits(&snr, 0.06);
            a.secondary_y = Some(SecondaryAxis::new("SNR (dB)", vec![20., 40., 60.], |v| {
                -100. + (v - sr[0]) / (sr[1] - sr[0]) * 200.
            })?);
        }
        let x = if frequency { right } else { left };
        let mut p = Panel::new(Rect::new(x, rows[2], width, height), a);
        p.major_grid("#e5e5e5", 0.4);
        curve(&mut p, &[tr[0], tr[1]], &[0., 0.], "#555555", 0.7);
        for col in 0..3 {
            if !frequency {
                let sr = limits(&snr, 0.06);
                let yy: Vec<_> = (0..display_t.len())
                    .map(|i| -100. + (snr[i * 3 + col] - sr[0]) / (sr[1] - sr[0]) * 200.)
                    .collect();
                let mut l = Line::new(points(&display_t, &yy), COLORS[col]);
                l.opacity = 0.7;
                l.width_pt = 0.8;
                p.push(Layer::Line(l));
            }
            for used in [false, true] {
                let ids: Vec<_> = (0..t.len())
                    .filter(|&i| keep[i * 3 + col] == used)
                    .collect();
                let xx: Vec<_> = ids.iter().map(|&i| t[i]).collect();
                let yy: Vec<_> = ids
                    .iter()
                    .map(|&i| residual[i * 3 + col] * factor)
                    .collect();
                if used {
                    let bounds: Vec<_> = ids
                        .iter()
                        .map(|&i| {
                            [
                                (residual[i * 3 + col] - sigma[i * 3 + col]) * factor,
                                (residual[i * 3 + col] + sigma[i * 3 + col]) * factor,
                            ]
                        })
                        .collect();
                    p.errorbars(
                        &points(&xx, &yy),
                        None,
                        Some(&bounds),
                        COLORS[col],
                        0.65,
                        2.,
                        0.65,
                    )?;
                }
                dots(
                    &mut p,
                    &xx,
                    &yy,
                    COLORS[col],
                    3.,
                    if used { 1.0 } else { 0.35 },
                    !used,
                )?;
            }
        }
        let mean = if frequency {
            let values: Vec<_> = residual
                .iter()
                .zip(&keep)
                .filter_map(|(&v, &k)| (k && v.is_finite()).then_some(v.abs() * wavelength))
                .collect();
            values.iter().sum::<f64>() / values.len() as f64
        } else {
            attr(&j, "mean_abs_total_path_residual_m")
        };
        let text = if frequency {
            format!("$\\langle|\\delta\\dot{{L}}|\\rangle={mean:.0}\\,\\mathrm{{m\\,s^{{-1}}}}$")
        } else {
            format!("$\\langle|\\delta R|\\rangle={mean:.1}\\,\\mathrm{{m}}$")
        };
        f.labels.push(Label::new(
            text,
            x + width / 2.,
            rows[2] + if frequency { 6. } else { 14. },
        ));
        panel_label(&mut f, &p, if frequency { "(f)" } else { "(e)" });
        if frequency {
            let mut l = Legend::new(
                (0..3)
                    .map(|i| LegendEntry::circle(SITES[i], COLORS[i]))
                    .collect(),
            );
            l.position = LegendPosition::LowerLeft;
            l.columns = 1;
            l.padding_mm = 0.8;
            l.avoid_data = false;
            p.legend = Some(l);
        }
        f.panels.push(p);
    }
    article_figure(f, 1.)
}
