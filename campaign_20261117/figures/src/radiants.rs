use anyhow::Result;
use rustplotlib::annotation::GraticuleLabel;
use rustplotlib::*;

fn array(h: &hdf5::File, name: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(name)?.read_raw::<f64>()?)
}

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let lon = array(h, "lambda_minus_sun_deg")?;
    let lat = array(h, "beta_ecliptic_deg")?;
    let speed = array(h, "speed_km_s")?;
    let x = array(h, "signed_westward_apex_longitude_deg")?;
    let y = array(h, "grid_latitude_deg")?;
    let visible = array(h, "visible_hours")?;
    let effective = array(h, "effective_hours")?;
    anyhow::ensure!(lon.len() == lat.len() && lon.len() == speed.len());
    anyhow::ensure!(visible.len() == x.len() * y.len() && visible.len() == effective.len());
    let article = ArticleLayout::new(184., 10.)?;
    let mut f = article.figure(1., 88.)?;
    let projection = Projection::Hammer {
        central_longitude: 270.,
        west_positive: true,
    };
    let root2 = 2_f64.sqrt();
    let mut panel = Panel::new(
        Rect::new(22., 7., 136., 68.),
        Axes::new([-2. * root2, 2. * root2], [-root2, root2], "", ""),
    );
    panel.axes_visible = false;
    let gray = f.scale(Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::new(vec![[0.86; 3], [0.86; 3]])?,
    )?);
    let mut polygons = Vec::new();
    for j in 0..y.len() - 1 {
        for i in 0..x.len() - 1 {
            let indices = [
                j * x.len() + i,
                j * x.len() + i + 1,
                (j + 1) * x.len() + i + 1,
                (j + 1) * x.len() + i,
            ];
            if indices.iter().all(|&k| visible[k] == 0.) {
                let points = [
                    [270. - x[i], y[j]],
                    [270. - x[i + 1], y[j]],
                    [270. - x[i + 1], y[j + 1]],
                    [270. - x[i], y[j + 1]],
                ];
                polygons.push(
                    points
                        .iter()
                        .map(|&p| projection.project(p).unwrap())
                        .collect(),
                );
            }
        }
    }
    let n = polygons.len();
    panel.push(Layer::PolygonMesh(PolygonMesh::new(
        polygons,
        vec![0.5; n],
        gray,
    )?));
    let mut frame = projection.boundary("#111111");
    frame.width_pt = 0.8;
    let mut obstacles = panel.stroke_obstacles(&frame, 0.4);
    for latitude in (-75..=75).step_by(15) {
        let mut parallel = projection.parallel(latitude as f64, "#d6d6d6");
        parallel.width_pt = 0.6;
        obstacles.extend(panel.stroke_obstacles(&parallel, 0.5));
        panel.push(Layer::Line(parallel));
        let q = projection.project([450., latitude as f64]).unwrap();
        let p = panel.data_to_figure(q);
        f.labels.push(f.boundary_label(
            format!("${latitude}^\\circ$"),
            p,
            [q[0] / 8., -q[1] / 2.],
            0.8,
        )?);
    }
    for longitude in (0..360).step_by(30) {
        let mut meridian = projection.meridian(longitude as f64, "#d6d6d6");
        meridian.width_pt = 0.6;
        panel.push(Layer::Line(meridian));
    }
    let max = effective.iter().copied().fold(0_f64, f64::max).floor() as usize;
    let contour = Contour::new(
        x,
        y,
        effective,
        (1..=max).map(|i| i as f64).collect(),
        gray,
        false,
    )?;
    let mut contour_paths = Vec::new();
    for level in 1..=max {
        let mut candidates = Vec::new();
        for path in contour.paths(level as f64) {
            let geographic: Vec<_> = path.iter().map(|p| [270. - p[0], p[1]]).collect();
            let mut line = projection.line(&geographic, "#929292");
            line.width_pt = 0.6;
            candidates.extend(line.points.iter().map(|&p| panel.data_to_figure(p)));
            candidates.push([f64::NAN; 2]);
            panel.push(Layer::Line(line));
        }
        contour_paths.push((level, candidates));
    }
    panel.push(Layer::Line(frame));
    let speed_scale = Scale::new([15., 71.], Normalization::Linear, ColorMap::turbo())?;
    let speed_id = f.scale(speed_scale.clone());
    for (longitude, value) in [(0., 0.), (180., 1.)] {
        let scale = Scale::new(
            [0., 1.],
            Normalization::Linear,
            ColorMap::new(vec![[1., 0.83, 0.12], [0., 0., 0.]])?,
        )?;
        panel.scatter_colored(
            &[projection.project([longitude, 0.]).unwrap()],
            &[value],
            &scale,
            8.,
            1.,
            "#222222",
        )?;
    }
    let points: Vec<_> = lon
        .iter()
        .zip(&lat)
        .map(|(&a, &b)| projection.project([a, b]).unwrap())
        .collect();
    let mut text_obstacles = obstacles.clone();
    for &p in &points {
        let p = panel.data_to_figure(p);
        text_obstacles.push(Rect::new(p[0] - 1., p[1] - 1., 2., 2.));
    }
    panel.scatter_colored(&points, &speed, &speed_scale, 4., 0.76, "white")?;
    let p = panel.data_to_figure(projection.project([270., 0.]).unwrap());
    f.labels.push(Label::new(
        r"{\fontsize{16}{19}\selectfont$\otimes$}",
        p[0],
        p[1],
    ));
    // Choose one common latitude band, rather than independently scattering
    // longitude labels across different parallels as the font size changes.
    let mut tick_labels = None;
    for latitude in [-52.5, -37.5, -22.5].into_iter()
        .chain((-64..=-20).rev().map(|v| v as f64)) {
        let ticks: Vec<_> = [0, 270, 180, 60, 300, 240, 120].iter()
            .map(|&longitude| GraticuleLabel {
                latex: format!("${longitude}^\\circ$"),
                required: [0, 270, 180].contains(&longitude),
                candidates: vec![panel.data_to_figure(projection.project(
                    [longitude as f64, latitude]).unwrap())],
            }).collect();
        if let Ok(labels) = f.graticule_labels(&ticks, &text_obstacles) {
            tick_labels = Some(labels);
            break;
        }
    }
    let tick_labels = tick_labels.ok_or_else(|| anyhow::anyhow!(
        "No common clear longitude-label band at article font size"))?;
    f.labels.extend(tick_labels);
    // Contours may be interrupted, but the outer frame must never be erased.
    let bounds = Rect::new(22.5, 7.5, 135., 67.);
    let mut contour_obstacles = Vec::new();
    contour_obstacles.extend(panel.stroke_obstacles(&projection.boundary("black"), 0.4));
    for &p in &points {
        let p = panel.data_to_figure(p);
        contour_obstacles.push(Rect::new(p[0] - 1., p[1] - 1., 2., 2.));
    }
    for (level, path) in contour_paths {
        match f.curved_contour_label(&format!("{level} h"), 10., &path, bounds, &contour_obstacles) {
            Ok(mut labels) => {
                for label in &mut labels {
                    label.color = "#666666".into();
                }
                f.labels.extend(labels);
            }
            Err(error) => eprintln!("Contour retained without crowded label: {error}"),
        }
    }
    f.panels.push(panel);
    f.labels.push(Label::new(
        r"Sun-centered ecliptic longitude, $\lambda-\lambda_\odot$",
        90.,
        83.,
    ));
    let mut ylabel = Label::new(r"Ecliptic latitude, $\beta$", 5., 41.);
    ylabel.rotation = -90.;
    f.labels.push(ylabel);
    f.colorbars.push(Colorbar::new(
        Rect::new(166., 13., 2.5, 56.),
        speed_id,
        r"Geocentric speed, $v_g$ ($\mathrm{km\,s^{-1}}$)",
        vec![20., 30., 40., 50., 60., 70.],
    ));
    // Curved text uses separately rotated glyphs whose bounding rectangles can
    // intersect even though their ink does not. Placement above checks obstacles.
    f.check_label_overlap = false;
    article.check_inclusion(&f, 184.)?;
    Ok(f)
}
