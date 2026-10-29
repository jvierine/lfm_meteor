use anyhow::Result;
use rustplotlib::*;
use std::path::Path;

fn edges(x: &[f64]) -> Vec<f64> {
    let mut e = vec![x[0] - (x[1] - x[0]) / 2.];
    e.extend(x.windows(2).map(|p| (p[0] + p[1]) / 2.));
    e.push(x[x.len()-1] + (x[x.len()-1] - x[x.len()-2]) / 2.);
    e
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: syisr-lookup snapshot.h5 output-base");
    let h = hdf5::File::open(&args[1])?;
    let x = h.dataset("velocity_km_s")?.read_raw::<f64>()?;
    let y: Vec<_> = h.dataset("source_n0_rM2_m_inv")?.read_raw::<f64>()?
        .into_iter().map(f64::log10).collect();
    let alt = h.dataset("altitude_km")?.read_raw::<f64>()?;
    let data = h.dataset("rcs_dbsm")?.read_raw::<f64>()?;
    let overlay=if let Ok(g)=h.group("measurements") {Some((
        g.dataset("altitude_panel")?.read_raw::<f64>()?,
        g.dataset("speed_km_s")?.read_raw::<f64>()?,
        g.dataset("log10_source")?.read_raw::<f64>()?))}else{None};
    let xe = edges(&x);
    let ye = edges(&y);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 240.; f.height_mm = 205.; f.font_pt = 10.;
    let scale = f.scale(Scale::new([-150., -20.], Normalization::Linear, ColorMap::inferno())?);
    for panel in 0..9 {
        let i = 8-panel;
        let row = panel/3;
        let col = panel%3;
        let rect = Rect::new(23.+col as f64*64., 12.+row as f64*64., 52., 46.);
        let mut axes = Axes::new([xe[0], *xe.last().unwrap()], [ye[0], *ye.last().unwrap()],
            if row == 2 {"Speed (km s$^{-1}$)"} else {""},
            if col == 0 {"$\\log_{10}[n_0r_M^2/(\\mathrm{m}^{-1})]$"} else {""});
        axes.xticks = vec![10.,30.,50.,72.];
        axes.yticks = vec![12.4,13.1,13.7,14.4];
        let mut p = Panel::new(rect, axes);
        // Skip missing cells entirely, without interpolating across holes.
        for s in 0..y.len() {
            for v in 0..x.len() {
                let z = data[(i*x.len()+v)*y.len()+s];
                if z.is_finite() {
                    p.pcolormesh(vec![xe[v],xe[v+1]], vec![ye[s],ye[s+1]], vec![z], scale)?;
                }
            }
        }
        if let Some((panel_index,speed,source))=&overlay {
            let pts:Vec<_>=(0..speed.len()).filter(|&k|panel_index[k] as usize==i).map(|k|[speed[k],source[k]]).collect();
            let white=Scale::new([0.,1.],Normalization::Linear,ColorMap::new(vec![[1.,1.,1.],[1.,1.,1.]])?)?;
            p.scatter_colored(&pts,&vec![0.5;pts.len()],&white,2.6,0.65,"#111111")?;
        }
        f.panels.push(p);
        f.labels.push(Label::new(format!("{} km",alt[i]), rect.x+rect.width/2., rect.y-4.));
    }
    f.colorbars.push(Colorbar::new(Rect::new(216., 18., 3., 164.), scale,
        "RCS (dBsm)", vec![-140.,-120.,-100.,-80.,-60.,-40.,-20.]));
    let out = Path::new(&args[2]);
    f.latex.cache = out.parent().unwrap().join(".native-latex-cache");
    for ext in ["pdf", "png", "svg"] { f.save(out.with_extension(ext))?; }
    Ok(())
}
