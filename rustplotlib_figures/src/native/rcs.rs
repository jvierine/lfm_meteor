use super::common::*;
use anyhow::Result;
use rustplotlib::*;
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let pulse = h.group("pulses")?;
    let event = h.group("mass_comparison")?;
    let rcs = array(&pulse, "rcs_dbsm")?;
    let alt = array(&pulse, "alt_km")?;
    let speed = array(&pulse, "speed_km_s")?;
    let power = array(&event, "dynamic_mass_kg")?;
    let sigma = array(&event, "rcs_mass_kg")?;
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 250.,
        font_pt: 9.,
    });
    f.height_mm = 82.;
    let color = Scale::new([15., 71.], Normalization::Linear, ColorMap::turbo())?;
    let cid = f.scale(color.clone());
    let low = quantile(&rcs, 0.01);
    let high = quantile(&rcs, 0.99);
    let edges: Vec<_> = (0..34)
        .map(|i| low + (high - low) * i as f64 / 33.)
        .collect();
    let counts = histogram(&rcs, &edges);
    let mut a = Axes::new(
        [low, high],
        [0., counts.iter().copied().fold(0_f64, f64::max) * 1.1],
        "RCS (dBsm)",
        "Pulse count",
    );
    a.xticks = (-6..=1)
        .map(|i| i as f64 * 10.)
        .filter(|v| *v >= low && *v <= high)
        .collect();
    let mut p = Panel::new(Rect::new(22., 10., 43., 53.), a);
    p.major_grid("#e4e4e4", 0.4);
    for i in 0..counts.len() {
        rect(
            &mut p,
            [edges[i], edges[i + 1]],
            [0., counts[i]],
            COLORS[0],
            0.85,
        );
    }
    panel_label(&mut f, &p, "(a)");
    f.panels.push(p);
    let mut p = Panel::new(
        Rect::new(87., 10., 47., 53.),
        Axes::new(
            limits(&rcs, 0.03),
            limits(&alt, 0.04),
            "RCS (dBsm)",
            "Altitude (km)",
        ),
    );
    p.major_grid("#e4e4e4", 0.4);
    p.scatter_colored(&points(&rcs, &alt), &speed, &color, 2.5, 0.62, "none")?;
    panel_label(&mut f, &p, "(b)");
    f.panels.push(p);
    let good: Vec<_> = (0..power.len())
        .filter(|&i| power[i] > 0. && sigma[i] > 0. && power[i].is_finite() && sigma[i].is_finite())
        .collect();
    let x: Vec<_> = good.iter().map(|&i| power[i]).collect();
    let y: Vec<_> = good.iter().map(|&i| sigma[i]).collect();
    let xlo = array(&event, "dynamic_lo95_kg")?;
    let xhi = array(&event, "dynamic_hi95_kg")?;
    let ylo = array(&event, "delta_mass_lo95_noise_only_kg")?;
    let yhi = array(&event, "delta_mass_hi95_noise_only_kg")?;
    let xb: Vec<_> = good
        .iter()
        .map(|&i| [xlo[i].min(power[i]), xhi[i].max(power[i])])
        .collect();
    let yb: Vec<_> = good
        .iter()
        .map(|&i| [ylo[i].min(sigma[i]), yhi[i].max(sigma[i])])
        .collect();
    let xr = [10_f64, 1000_f64].map(|r| 4. * std::f64::consts::PI / 3. * 3000. * (r * 1e-6).powi(3));
    let yr = xr;
    let mut a = Axes::new(
        xr,
        yr,
        r"$m_{\mathrm{dyn}}$ (kg)",
        r"$m_{\mathrm{RCS}}$ (kg)",
    );
    a.set_xscale(AxisScale::Log10)?;
    a.set_yscale(AxisScale::Log10)?;
    a.xticks = vec![1e-10, 1e-8, 1e-6];
    a.yticks = a.xticks.clone();
    a.xticklabels = Some(vec![r"$10^{-10}$".into(), r"$10^{-8}$".into(), r"$10^{-6}$".into()]);
    a.yticklabels = a.xticklabels.clone();
    let mut p = Panel::new(Rect::new(188., 10., 47., 53.), a);
    p.major_grid("#e4e4e4", 0.4);
    p.errorbars(
        &points(&x, &y),
        Some(&xb),
        Some(&yb),
        "#000000",
        0.4,
        0.,
        0.5,
    )?;
    dots(&mut p, &x, &y, "#000000", 1.8, 1., false)?;
    let mut l = Line::new(vec![[xr[0], xr[0]], [xr[1], xr[1]]], "#666666");
    l.dash = Some("2 1".into());
    p.push(Layer::Line(l));
    panel_label(&mut f, &p, "(c)");
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(138., 10., 2., 53.),
        cid,
        r"Geocentric speed ($\mathrm{km\,s^{-1}}$)",
        vec![20., 40., 60.],
    ));
    article_figure(f, 1.)
}
