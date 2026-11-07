use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    density_figure(h, false)
}

pub fn comparison_figure(h: &hdf5::File) -> Result<Figure> {
    density_figure(h, true)
}

fn density_figure(h: &hdf5::File, comparison: bool) -> Result<Figure> {
    let speeds = array(h, "velocity_km_s")?;
    let heights = array(h, "altitude_km")?;
    let nstar = array(h, "n_star_min_m3")?;
    let values = array(h, "log10_density_over_critical")?;
    let x: Vec<_> = array(h, "z_edges_m")?.iter().map(|v| v*100.).collect();
    let y: Vec<_> = array(h, "r_edges_m")?.iter().map(|v| v*100.).collect();
    let xc: Vec<_> = x.windows(2).map(|v| (v[0]+v[1])/2.).collect();
    let yc: Vec<_> = y.windows(2).map(|v| (v[0]+v[1])/2.).collect();
    let cells = (x.len()-1)*(y.len()-1);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 175.;
    f.height_mm = if comparison {184.} else {175.};
    f.font_pt = 10.;
    let scale = f.scale(Scale::new([-3.,3.], Normalization::Linear, ColorMap::inferno())?);
    for row in 0..heights.len() {
        let altitude_index = heights.len() - 1 - row;
        let bottom = row == heights.len() - 1;
        for col in 0..speeds.len() {
            let rect = Rect::new(20.+col as f64*38., 19.+row as f64*31., 34., 26.);
            let mut axes=Axes::new(if comparison {[-20.,120.]} else {[-6.,24.]},
                if comparison {[-50.,50.]} else {[-9.,9.]}, if bottom {"$z$ (cm)"} else {""},
                if col==0 {format!("{:.0} km; $r$ (cm)",heights[altitude_index])} else {String::new()});
            axes.xticks=if comparison {vec![0.,50.,100.]} else {vec![0.,10.,20.]};
            axes.yticks=if comparison {vec![-40.,0.,40.]} else {vec![-5.,0.,5.]};
            axes.show_x_axis=bottom;
            axes.show_y_axis=col==0;
            let mut panel=Panel::new(rect,axes);
            let index = altitude_index*speeds.len()+col;
            let data=values[index*cells..(index+1)*cells].to_vec();
            panel.pcolormesh(x.clone(),y.clone(),data.clone(),scale)?;
            let maarsy_level=2.*(53.5_f64/430.).log10();
            let contour=Contour::new(xc.clone(),yc.clone(),data,vec![maarsy_level,0.],scale,false)?;
            for (level,color) in [(maarsy_level,"#ffffff"),(0.,"#5de4ce")] {
                if !comparison && level != 0. {continue;}
                let mut path=Vec::new();
                for segment in contour.segments(level) {path.extend([segment[0],segment[1],[f64::NAN;2]]);}
                let mut line=Line::new(path,color);
                line.width_pt=0.9;
                panel.push(Layer::Line(line));
            }
            f.panels.push(panel);
            let mut label=Label::new(format!("$\\log_{{10}}n_\\star={:.2}$",nstar[index].log10()),
                rect.x+rect.width/2.,rect.y-2.2);
            label.background=Some("white".into());
            label.padding_mm=0.2;
            f.labels.push(label);
            if row==0 {f.labels.push(Label::new(format!("${:.1}$ km s$^{{-1}}$",speeds[col]),rect.x+rect.width/2.,8.));}
        }
    }
    let mut bar=Colorbar::new(Rect::new(28.,156.,135.,2.5),scale,
        "$\\log_{10}(n_e/n_{\\mathrm{crit}})$",vec![-3.,-2.,-1.,0.,1.,2.,3.]);
    bar.horizontal=true;
    f.colorbars.push(bar);
    if comparison {
    for (x0,text,color) in [(22.,"SYISR: 430 MHz","#5de4ce"),
                          (95.,"MAARSY: 53.5 MHz","#ffffff")] {
        let mut axes=Axes::new([0.,1.],[0.,1.],"","");
        axes.show_x_axis=false; axes.show_y_axis=false;
        let mut key=Panel::new(Rect::new(x0,175.,8.,1.),axes);
        rect(&mut key,[0.,1.],[0.,1.],"#555555",1.);
        let mut line=Line::new(vec![[0.,0.5],[1.,0.5]],color);
        line.width_pt=1.;
        key.push(Layer::Line(line));
        f.panels.push(key);
        f.labels.push(Label::new(text,x0+35.,175.5));
    }
    }
    Ok(f)
}
