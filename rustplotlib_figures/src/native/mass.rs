use super::common::*;
use anyhow::Result;
use rustplotlib::*;
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    if h.link_exists("rcs_mass_kg") {
        return mirrored_figure(h);
    }
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
    anyhow::ensure!(!m.is_empty(), "No selected masses");
    let x: Vec<_> = m.iter().map(|v| v.log10()).collect();
    let rx: Vec<f64> = if h.link_exists("rcs_mass_kg") {array(h,"rcs_mass_kg")?.iter().map(|v|v.log10()).collect()} else {vec![]};
    let mut all_x = x.clone(); all_x.extend(&rx);
    let range = limits(&all_x, 0.02);
    let edges: Vec<_> = if let Ok(attr) = h.attr("bin_width_dex") {
        let width = attr.read_scalar::<f64>()?;
        let start = (range[0] / width).floor() * width;
        let n = ((range[1] - start) / width).ceil() as usize;
        (0..=n).map(|i| start + i as f64 * width).collect()
    } else { (0..=8)
        .map(|i| range[0] + i as f64 * (range[1] - range[0]) / 8.)
        .collect() };
    let counts = histogram(&x, &edges);
    let rcs_counts = histogram(&rx, &edges);
    let count_max = (counts.iter().chain(&rcs_counts).copied().fold(30.0_f64, f64::max) / 10.).ceil() * 10.;
    let bin_width = edges[1] - edges[0];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.height_mm = 112.;
    let mut interval_values: Vec<_> = lo.iter().chain(&hi).map(|v| v.log10()).collect();
    if !rx.is_empty() {
        interval_values.extend(array(h,"rcs_lo95_kg")?.iter().chain(array(h,"rcs_hi95_kg")?.iter()).map(|v|v.log10()));
    }
    let mut a = Axes::new(
        limits(&interval_values, 0.03),
        [0., 75.],
        r"$\log_{10}(m_0/\mathrm{kg})$",
        "",
    );
    a.xticks = (a.xlim[0].ceil() as i32..=a.xlim[1].floor() as i32)
        .map(|i| i as f64)
        .collect();
    a.show_y_axis = false;
    let diameter_axis = SecondaryAxis::new(
        r"Equivalent diameter ($\mu\mathrm{m}$)",
        vec![5., 10., 20., 30., 50., 100., 200., 500.],
        |d| (3000. * std::f64::consts::PI / 6. * (d * 1e-6).powi(3)).log10(),
    )?;
    a.secondary_x = Some(diameter_axis);
    let to_count = |v: f64| 42. + v * 30. / 72.;
    let xlim = a.xlim;
    let mut p = Panel::new(Rect::new(24., 23., 129., 72.), a);
    for i in 0..counts.len() {
        rect(
            &mut p,
            [edges[i], edges[i + 1]],
            [0., counts[i] * 30. / count_max],
            "#2f7f6f",
            0.58,
        );
    }
    if !rx.is_empty() {
        let mut steps = vec![[edges[0], 0.]];
        for i in 0..rcs_counts.len() {
            let height = rcs_counts[i] * 30. / count_max;
            steps.push([edges[i], height]); steps.push([edges[i+1], height]);
        }
        steps.push([*edges.last().unwrap(), 0.]);
        let mut line = Line::new(steps,"#0072B2"); line.width_pt=1.4;
        p.push(Layer::Line(line));
    }
    let median = quantile(&x, 0.5);
    let mut med = Line::new(vec![[median, 0.], [median, 75.]], "#17463d");
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
    if !rx.is_empty() {
        let ry:Vec<_>=array(h,"rcs_speed_km_s")?.iter().map(|v|to_count(*v)).collect();
        let rl=array(h,"rcs_lo95_kg")?; let rh=array(h,"rcs_hi95_kg")?;
        let rb:Vec<_>=(0..rx.len()).map(|i|[rl[i].log10().min(rx[i]),rh[i].log10().max(rx[i])]).collect();
        p.errorbars(&points(&rx,&ry),Some(&rb),None,"#0072B2",0.65,2.4,0.5)?;
        dots(&mut p,&rx,&ry,"#0072B2",3.0,0.9,true)?;
    }
    let mut legend = Legend::new(vec![
        LegendEntry::line(format!("Dynamic initial mass, $n={}$", m.len()), "#2f7f6f"),
        LegendEntry {
            label: "Mass estimates; 95\\% intervals".into(),
            symbol: LegendSymbol::AsymmetricInterval { color: "#17463d".into(), size_pt: 3.7 },
        },
    ]);
    if !rx.is_empty() {
        legend.entries.insert(1,LegendEntry::line(format!("RCS in-beam mass loss, $n={}$",rx.len()),"#0072B2"));
    }
    legend.position = LegendPosition::CenterRight;
    legend.avoid_data = false;
    p.legend = Some(legend);
    f.panels.push(p);
    // Floating axes cover only their respective histogram and speed regions.
    let mut count_axes = Axes::new(
        xlim,
        [0., count_max],
        "",
        format!("Trajectories / ({bin_width:.2} dex)"),
    );
    count_axes.show_x_axis = false;
    count_axes.yticks = (0..=(count_max / 10.) as usize).map(|i| i as f64 * 10.).collect();
    f.panels.push(Panel::new(Rect::new(24., 66.2, 129., 28.8), count_axes));
    let mut speed_axes = Axes::new(xlim, [0., 72.], "", "");
    speed_axes.show_x_axis = false;
    speed_axes.show_y_axis = false;
    speed_axes.secondary_y = Some(SecondaryAxis::new(
        r"Initial speed ($\mathrm{km\,s^{-1}}$)",
        vec![0., 20., 40., 60., 72.],
        |v| v,
    )?);
    f.panels.push(Panel::new(Rect::new(24., 25.88, 129., 28.8), speed_axes));
    article_figure(f, 0.8)
}

fn mirrored_figure(h: &hdf5::File) -> Result<Figure> {
    let keep = h.dataset("selected")?.read_raw::<bool>()?;
    let selected = |name: &str| -> Result<Vec<f64>> {
        Ok(array(h, name)?.into_iter().zip(&keep).filter_map(|(v, k)| k.then_some(v)).collect())
    };
    let masses = [selected("initial_mass_kg")?, array(h, "rcs_mass_kg")?];
    let speeds = [selected("initial_speed_km_s")?, array(h, "rcs_speed_km_s")?];
    let lows = [selected("mass_95_lo_kg")?, array(h, "rcs_lo95_kg")?];
    let highs = [selected("mass_95_hi_kg")?, array(h, "rcs_hi95_kg")?];
    let xs: Vec<Vec<f64>> = masses.iter().map(|m| m.iter().map(|v| v.log10()).collect()).collect();
    let values: Vec<f64> = xs.iter().flatten().copied().collect();
    let range = limits(&values, 0.02);
    let width = h.attr("bin_width_dex")?.read_scalar::<f64>()?;
    let start = (range[0] / width).floor() * width;
    let bins = ((range[1] - start) / width).ceil() as usize;
    let edges: Vec<f64> = (0..=bins).map(|i| start + i as f64 * width).collect();
    let counts: Vec<Vec<f64>> = xs.iter().map(|x| histogram(x, &edges)).collect();
    let count_max = (counts.iter().flatten().copied().fold(30_f64, f64::max) / 20.).ceil() * 20.;
    let interval_values: Vec<f64> = lows.iter().chain(&highs).flatten().map(|v| v.log10()).collect();
    let mut xlim = limits(&interval_values, 0.03);
    xlim[0] = xlim[0].min(edges[0]);
    xlim[1] = xlim[1].max(*edges.last().unwrap());
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.height_mm = 148.;
    let unit_mm = 64. / 65.;
    let panel_height = 55. * unit_mm;
    let center = 23. + panel_height;
    // The lower panel reverses every vertical coordinate, including the axes.
    for j in 0..2 {
        let sign = if j == 0 { 1. } else { -1. };
        let top = if j == 0 { 23. } else { center };
        let color = if j == 0 { "#2f7f6f" } else { "#0072B2" };
        let ylim = if j == 0 { [0., 55.] } else { [-55., 0.] };
        let mut a = Axes::new(xlim, ylim, if j == 1 { r"$\log_{10}(m/\mathrm{kg})$" } else { "" }, "");
        a.show_y_axis = false;
        a.show_x_axis = j == 1;
        a.xticks = (xlim[0].ceil() as i32..=xlim[1].floor() as i32).map(|v| v as f64).collect();
        if j == 0 {
            a.secondary_x = Some(SecondaryAxis::new(r"Equivalent radius ($\mu\mathrm{m}$)",
                vec![20., 30., 50., 100., 200., 500., 1000.],
                |r| (3000. * 4. * std::f64::consts::PI / 3. * (r * 1e-6).powi(3)).log10())?);
        }
        let mut p = Panel::new(Rect::new(24., top, 129., panel_height), a);
        for decade in xlim[0].ceil() as i32..=xlim[1].floor() as i32 {
            let mut grid = Line::new(vec![[decade as f64, ylim[0]], [decade as f64, ylim[1]]], "#000000");
            grid.width_pt = 0.4;
            grid.opacity = 0.12;
            grid.protect_from_legend = false;
            p.push(Layer::Line(grid));
        }
        curve(&mut p, &xlim, &[0., 0.], "#777777", 0.6);
        for i in 0..bins {
            let height = counts[j][i] * 30. / count_max;
            rect(&mut p, [edges[i], edges[i+1]], if j == 0 { [0., height] } else { [-height, 0.] }, color, 0.5);
        }
        let median = quantile(&xs[j], 0.5);
        let histogram_height = counts[j].iter().copied().fold(0_f64, f64::max) * 30. / count_max;
        let mut line = Line::new(vec![[median, 0.], [median, sign * histogram_height]], color);
        line.width_pt = 1.; line.dash = Some("2 1".into());
        p.push(Layer::Line(line));
        let ys: Vec<f64> = speeds[j].iter().map(|v| sign * (33. + v * 20. / 72.)).collect();
        let bounds: Vec<[f64;2]> = (0..xs[j].len()).map(|i|
            [lows[j][i].log10().min(xs[j][i]), highs[j][i].log10().max(xs[j][i])]).collect();
        p.errorbars(&points(&xs[j], &ys), Some(&bounds), None, color, 0.65, 2.4, 0.5)?;
        dots(&mut p, &xs[j], &ys, color, 3.2, 0.9, true)?;
        let name = if j == 0 { "Dynamic mass" } else { "RCS mass loss" };
        let mut legend = Legend::new(vec![
            LegendEntry { label: format!("{name} ($n={}$, 95\\%)", xs[j].len()), symbol: LegendSymbol::AsymmetricInterval { color: color.into(), size_pt: 3.2 } },
        ]);
        legend.position = LegendPosition::CenterRight; legend.avoid_data = false;
        legend.padding_mm = 0.7;
        legend.background_opacity = 0.;
        p.legend = Some(legend);
        f.panels.push(p);
        let mut sa = Axes::new(xlim, if j == 0 { [0., 72.] } else { [-72., 0.] }, "", "");
        sa.show_x_axis = false; sa.show_y_axis = false;
        sa.secondary_y = Some(SecondaryAxis::new(r"Initial speed ($\mathrm{km\,s^{-1}}$)", vec![0.,20.,40.,60.,72.], move |v| sign*v)?);
        f.panels.push(Panel::new(Rect::new(24., if j == 0 {top+2.*unit_mm} else {top+33.*unit_mm}, 129., 20.*unit_mm), sa));
    }
    let mut ca = Axes::new(xlim, [-count_max, count_max], "", format!("Count / ({width:.2} dex)"));
    ca.show_x_axis = false;
    ca.yticks = (-4..=4).map(|i| count_max * i as f64 / 4.).collect();
    ca.yticklabels = Some((-4_i32..=4).map(|i| format!("{:.0}", count_max * i.abs() as f64 / 4.)).collect());
    f.panels.push(Panel::new(Rect::new(24., center-30.*unit_mm, 129., 60.*unit_mm), ca));
    article_figure(f, 0.8)
}
