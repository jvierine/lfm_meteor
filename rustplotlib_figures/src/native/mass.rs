use super::common::*;
use anyhow::Result;
use rustplotlib::*;
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let keep = h.dataset("selected")?.read_raw::<bool>()?;
    let selected = |name: &str| -> Result<Vec<f64>> {
        Ok(array(h, name)?
            .into_iter()
            .zip(&keep)
            .filter_map(|(v, &k)| k.then_some(v))
            .collect())
    };
    let m = selected("initial_mass_kg")?;
    let speed = selected("initial_speed_km_s")?;
    let lo = selected("mass_95_lo_kg")?;
    let hi = selected("mass_95_hi_kg")?;
    anyhow::ensure!(m.len() == 110, "Mass selection changed");
    let x: Vec<_> = m.iter().map(|v| v.log10()).collect();
    let range = limits(&x, 0.02);
    let edges: Vec<_> = (0..=8)
        .map(|i| range[0] + i as f64 * (range[1] - range[0]) / 8.)
        .collect();
    let counts = histogram(&x, &edges);
    let max = counts.iter().copied().fold(0_f64, f64::max);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.height_mm = 112.;
    let interval_values: Vec<_> = lo.iter().chain(&hi).map(|v| v.log10()).collect();
    let mut a = Axes::new(
        limits(&interval_values, 0.03),
        [0., max / 0.45],
        r"$\log_{10}(m_0/\mathrm{kg})$",
        "Number of fitted trajectories",
    );
    a.xticks = (a.xlim[0].ceil() as i32..=a.xlim[1].floor() as i32)
        .map(|i| i as f64)
        .collect();
    a.yticks = (0..=max as usize).step_by(5).map(|i| i as f64).collect();
    a.secondary_x = Some(SecondaryAxis::new(
        r"Equivalent diameter ($\mu\mathrm{m}$)",
        vec![5., 10., 20., 30., 50., 100., 200., 500.],
        |d| (3000. * std::f64::consts::PI / 6. * (d * 1e-6).powi(3)).log10(),
    )?);
    let smin = speed.iter().copied().fold(f64::INFINITY, f64::min);
    let smax = speed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (smax - smin).max(1.) / 0.34;
    let low = smin - 0.62 * span;
    let to_count = |v: f64| (v - low) / span * (max / 0.45);
    let speed_ticks: Vec<_> = (smin.div_euclid(10.) as i32 + 1..=smax.div_euclid(10.) as i32)
        .map(|i| i as f64 * 10.)
        .collect();
    a.secondary_y = Some(SecondaryAxis::new(
        r"Initial speed ($\mathrm{km\,s^{-1}}$)",
        speed_ticks,
        to_count,
    )?);
    let mut p = Panel::new(Rect::new(24., 23., 129., 72.), a);
    for i in 0..counts.len() {
        rect(
            &mut p,
            [edges[i], edges[i + 1]],
            [0., counts[i]],
            "#2f7f6f",
            0.58,
        );
    }
    let median = quantile(&x, 0.5);
    let mut med = Line::new(vec![[median, 0.], [median, max / 0.45]], "#17463d");
    med.width_pt = 1.2;
    med.dash = Some("2 1".into());
    p.push(Layer::Line(med));
    let sy: Vec<_> = speed.iter().map(|&v| to_count(v)).collect();
    let bounds: Vec<_> = (0..m.len())
        .map(|i| [lo[i].log10().min(x[i]), hi[i].log10().max(x[i])])
        .collect();
    p.errorbars(
        &points(&x, &sy),
        Some(&bounds),
        None,
        "#17463d",
        0.65,
        2.4,
        0.6,
    )?;
    dots(&mut p, &x, &sy, "#17463d", 3.7, 0.8, true)?;
    let mut legend = Legend::new(vec![
        LegendEntry::line("Mass histogram, $n=110$", "#2f7f6f"),
        LegendEntry::circle("Individual $m_0$ estimates", "#17463d"),
    ]);
    legend.position = LegendPosition::CenterRight;
    legend.avoid_data = false;
    p.legend = Some(legend);
    f.panels.push(p);
    article_figure(f, 0.8)
}
