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
    let color = Scale::new([15., 71.], Normalization::Linear, ColorMap::storm())?;
    let cid = f.scale(color.clone());
    let low = rcs.iter().copied().fold(f64::INFINITY, f64::min);
    let high = rcs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut edges: Vec<_> = (0..34)
        .map(|i| low + (high - low) * i as f64 / 33.)
        .collect();
    edges[0] = low;
    edges[33] = high;
    let counts = histogram(&rcs, &edges);
    anyhow::ensure!(counts.iter().sum::<f64>() == rcs.len() as f64,
        "RCS histogram must include every retained pulse");
    let mut a = Axes::new(
        [low, high],
        [0., counts.iter().copied().fold(0_f64, f64::max) * 1.1],
        "RCS (dBsm)",
        "Pulse count",
    );
    a.xticks = (-5..=1)
        .map(|i| i as f64 * 20.)
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
    let xlim = limits(&rcs, 0.03);
    let ylim = limits(&alt, 0.04);
    let bin_edges = |lim: [f64; 2], step: f64| -> Vec<f64> {
        let first = (lim[0] / step).floor() as i32;
        let last = (lim[1] / step).ceil() as i32;
        (first..=last).map(|i| i as f64 * step).collect()
    };
    let xe = bin_edges(xlim, 4.);
    let ye = bin_edges(ylim, 2.);
    let nx = xe.len() - 1;
    let mut count = vec![0.; nx * (ye.len() - 1)];
    let mut sum = count.clone();
    for ((&r, &h), &v) in rcs.iter().zip(&alt).zip(&speed) {
        let ix = xe.partition_point(|&e| e <= r) - 1;
        let iy = ye.partition_point(|&e| e <= h) - 1;
        count[iy * nx + ix] += 1.;
        sum[iy * nx + ix] += v;
    }
    anyhow::ensure!(count.iter().sum::<f64>() == rcs.len() as f64);
    let mean = sum.iter().zip(&count)
        .map(|(&s, &n)| if n >= 3. { s / n } else { f64::NAN }).collect();
    let centers = |edges: &[f64]| edges.windows(2).map(|w| (w[0]+w[1])/2.).collect();
    let levels = vec![8., 24., 60.];
    let contours = Contour::new(centers(&xe), centers(&ye), count, levels.clone(), cid, false)?;
    let mut p = Panel::new(
        Rect::new(87., 10., 47., 53.),
        Axes::new(
            xlim,
            ylim,
            "RCS (dBsm)",
            "Altitude (km)",
        ),
    );
    p.axes.as_mut().unwrap().xticks = (-5..=1).map(|i|i as f64*20.).filter(|v|*v>=low&&*v<=high).collect();
    p.major_grid("#e4e4e4", 0.4);
    p.push(Layer::Pcolormesh(Field::new(xe, ye, mean, cid)?));
    for (i, level) in levels.into_iter().enumerate() {
        let points = contours.paths(level).into_iter()
            .flat_map(|path| path.into_iter().chain([[f64::NAN; 2]])).collect();
        let mut line = Line::new(points, "#202020");
        line.width_pt = 0.55;
        line.opacity = 0.75;
        line.dash = [Some("1 1".into()), Some("2 1".into()), None][i].clone();
        p.push(Layer::Line(line));
    }
    panel_label(&mut f, &p, "(b)");
    f.panels.push(p);
    let compatible = if event.link_exists("compatible") {
        event.dataset("compatible")?.read_raw::<bool>()?
    } else {
        vec![true; power.len()]
    };
    anyhow::ensure!(compatible.len() == power.len());
    let good: Vec<_> = (0..power.len())
        .filter(|&i| compatible[i] && power[i] > 0. && sigma[i] > 0. && power[i].is_finite() && sigma[i].is_finite())
        .collect();
    let x: Vec<_> = good.iter().map(|&i| power[i]).collect();
    let y: Vec<_> = good.iter().map(|&i| sigma[i]).collect();
    let xlo = array(&event, "dynamic_lo95_kg")?;
    let xhi = array(&event, "dynamic_hi95_kg")?;
    let ylo = array(&event, "delta_mass_lo95_noise_only_kg")?;
    let yhi = array(&event, "delta_mass_hi95_noise_only_kg")?;
    let valid = |lo: f64, hi: f64| lo.is_finite() && hi.is_finite() && lo > 0. && hi > lo;
    let extent = x.iter().chain(&y).copied().fold([f64::INFINITY, 0_f64], |[lo, hi], v| [lo.min(v), hi.max(v)]);
    let xr = [10_f64.powf(extent[0].log10().floor()), 10_f64.powf(extent[1].log10().ceil())];
    let yr = xr;
    let mut a = Axes::new(
        xr,
        yr,
        r"$m_{\mathrm{dyn}}$ (kg)",
        r"$m_{\mathrm{RCS}}$ (kg)",
    );
    a.set_xscale(AxisScale::Log10)?;
    a.set_yscale(AxisScale::Log10)?;
    a.xticks = (xr[0].log10().round() as i32..=xr[1].log10().round() as i32).step_by(2).map(|p| 10_f64.powi(p)).collect();
    a.yticks = a.xticks.clone();
    a.xticklabels = Some(a.xticks.iter().map(|v| format!("$10^{{{:.0}}}$", v.log10())).collect());
    a.yticklabels = a.xticklabels.clone();
    let mut p = Panel::new(Rect::new(188., 10., 47., 53.), a);
    p.major_grid("#e4e4e4", 0.4);
    if h.link_exists("mass_powerlaw") {
        let fit = h.group("mass_powerlaw")?;
        p.uncertainty_band(
            &array(&fit, "dynamic_mass_kg")?,
            &array(&fit, "rcs_lower95_kg")?,
            &array(&fit, "rcs_upper95_kg")?,
            "#0072b2",
            0.25,
        )?;
    }
    for (lo,hi) in [(&xlo,&xhi),(&ylo,&yhi)] {
        anyhow::ensure!(lo.len() == power.len() && hi.len() == power.len());
    }
    for (lo, hi, horizontal) in [(&xlo, &xhi, true), (&ylo, &yhi, false)] {
        let indices: Vec<_> = good.iter().copied().filter(|&i| valid(lo[i], hi[i])).collect();
        let pts: Vec<_> = indices.iter().map(|&i| [power[i], sigma[i]]).collect();
        let bounds: Vec<_> = indices.iter().map(|&i| [lo[i], hi[i]]).collect();
        if !pts.is_empty() {
            p.errorbars(&pts, horizontal.then_some(&bounds), (!horizontal).then_some(&bounds),
                "#333333", 0.55, 1.3, 0.75)?;
        }
    }
    for complete in [false, true] {
        let (x,y): (Vec<_>,Vec<_>) = good.iter().copied().filter(|&i|
            (valid(xlo[i], xhi[i]) && valid(ylo[i], yhi[i])) == complete)
            .map(|i| (power[i], sigma[i])).unzip();
        if !x.is_empty() {
            dots(&mut p, &x, &y, if complete { "#111111" } else { "#888888" }, 1.25, 0.7, false)?;
        }
    }
    let mut l = Line::new(vec![[xr[0], xr[0]], [xr[1], xr[1]]], "#666666");
    l.dash = Some("2 1".into());
    p.push(Layer::Line(l));
    if h.link_exists("mass_powerlaw") {
        let fit = h.group("mass_powerlaw")?;
        curve(&mut p, &array(&fit, "dynamic_mass_kg")?, &array(&fit, "rcs_mass_kg")?, "#0072b2", 1.4);
        let mut legend = Legend::new(vec![LegendEntry::line("Fit", "#0072b2")]);
        legend.position = LegendPosition::LowerRight;
        p.legend = Some(legend);
    }
    panel_label(&mut f, &p, "(c)");
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(138., 10., 2., 53.),
        cid,
        r"Mean speed ($\mathrm{km\,s^{-1}}$)",
        vec![20., 40., 60.],
    ));
    article_figure(f, 1.)
}
