use anyhow::Result;
use rustplotlib::*;

#[derive(hdf5::H5Type, Clone, Copy)]
#[repr(C)]
struct Complex {
    r: f64,
    i: f64,
}

fn attr(h: &hdf5::File, name: &str) -> Result<f64> {
    Ok(h.attr(name)?.read_scalar::<f64>()?)
}

fn line(
    panel: &mut Panel,
    points: Vec<[f64; 2]>,
    color: &str,
    width: f64,
    dash: Option<&str>,
    opacity: f64,
) {
    let mut curve = Line::new(points, color);
    curve.width_pt = width;
    curve.dash = dash.map(String::from);
    curve.opacity = opacity;
    panel.push(Layer::Line(curve));
}

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let time = h.dataset("t_us")?.read_raw::<f64>()?;
    let voltage = h.dataset("deramped")?.read_raw::<Complex>()?;
    let corrected = h.dataset("fit_corrected_deramped")?.read_raw::<Complex>()?;
    let frequency = h.dataset("freq_hz")?.read_raw::<f64>()?;
    let power = h.dataset("power_db_rel")?.read_raw::<f64>()?;
    anyhow::ensure!(time.len() == voltage.len() && voltage.len() == corrected.len());
    anyhow::ensure!(frequency.len() == power.len());
    let norm = voltage.iter().map(|z| z.r.hypot(z.i)).fold(0_f64, f64::max);
    anyhow::ensure!(norm.is_finite() && norm > 0.);
    let peak = attr(h, "peak_frequency_hz")? / 1000.;
    let start = attr(h, "fit_echo_window_shift_us")?;
    let duration = attr(h, "pulse_length_us")?;
    let colors: Vec<String> = (0..6).map(|i| LineStyle::for_series(i).color).collect();
    let mut f = ArticleLayout::new(90., 10.)?.figure(1., 145.)?;
    let mut upper_axes = Axes::new(
        [-50., 250.],
        [-2., 2.],
        r"Time from gate start ($\mu\mathrm{s}$)",
        "Normalized voltage",
    );
    upper_axes.xticks = vec![0., 100., 200.];
    upper_axes.yticks = vec![-2., -1., 0., 1., 2.];
    let mut upper = Panel::new(Rect::new(17., 25., 69., 43.), upper_axes);
    upper.major_grid("#dddddd", 0.45);
    for x in [0., duration] {
        line(
            &mut upper,
            vec![[x, -2.], [x, 2.]],
            &colors[4],
            0.9,
            Some("1.2 0.8"),
            1.,
        );
    }
    for x in [start, start + duration] {
        line(
            &mut upper,
            vec![[x, -2.], [x, 2.]],
            &colors[5],
            1.1,
            Some("2 1"),
            1.,
        );
    }
    for (imaginary, index) in [(false, 0), (true, 1)] {
        let points = |data: &[Complex]| {
            time.iter()
                .zip(data)
                .map(|(&t, z)| [t, if imaginary { z.i / norm } else { z.r / norm }])
                .collect()
        };
        line(
            &mut upper,
            points(&corrected),
            &colors[index + 2],
            1.,
            Some("0.35 0.8"),
            1.,
        );
        line(&mut upper, points(&voltage), &colors[index], 0.85, None, 1.);
    }
    // A compact external key keeps the echo itself unobstructed.
    let key = [
        ("Zero-Doppler gate", colors[4].as_str(), Some("1.2 0.8")),
        ("Corrected real", colors[2].as_str(), Some("0.35 0.8")),
        ("Real", colors[0].as_str(), None),
        ("Fitted envelope", colors[5].as_str(), Some("2 1")),
        ("Corrected imag.", colors[3].as_str(), Some("0.35 0.8")),
        ("Imaginary", colors[1].as_str(), None),
    ];
    let mut key_panel = Panel::new(Rect::new(0., 0., 90., 20.), Axes::new([0.,90.], [0.,20.], "", ""));
    key_panel.axes_visible = false;
    for (i, (text, color, dash)) in key.iter().enumerate() {
        let x = 3. + (i % 2) as f64 * 44.;
        let y = 4. + (i / 2) as f64 * 6.;
        line(&mut key_panel, vec![[x,20.-y],[x+5.,20.-y]], color, 1., *dash, 1.);
        f.labels.push(Label::new(*text, x+23., y));
    }
    f.panels.push(key_panel);
    f.panels.push(upper);
    let mut lower_axes = Axes::new(
        [peak - 80., peak + 80.],
        [-55., 3.],
        r"Beat frequency, $f_b$ (kHz)",
        "Relative power (dB)",
    );
    lower_axes.xticks = vec![-80., -40., 0., 40.];
    lower_axes.yticks = vec![-50., -40., -30., -20., -10., 0.];
    let mut lower = Panel::new(Rect::new(17., 88., 69., 43.), lower_axes);
    lower.major_grid("#dddddd", 0.45);
    let points = frequency
        .iter()
        .zip(&power)
        .filter_map(|(&x, &y)| ((x / 1000. - peak).abs() <= 80.).then_some([x / 1000., y]))
        .collect();
    line(&mut lower, points, &colors[0], 1., None, 1.);
    line(
        &mut lower,
        vec![[peak, -55.], [peak, 3.]],
        &colors[1],
        1.1,
        None,
        1.,
    );
    let p = lower.data_to_figure([peak + 45., -7.]);
    f.labels.push(Label::new(
        format!("$f_b={peak:.2}\\,\\mathrm{{kHz}}$"),
        p[0],
        p[1],
    ));
    f.panels.push(lower);
    Ok(f)
}
