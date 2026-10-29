import {SeriesView,LinePlot,PlotViewport} from './pkg/rustplotlib_web.js';
import {bindPlotNavigation} from './pkg/navigation.js';

const finite=a=>a.filter(Number.isFinite);
const limits=(a,min=1)=>{a=finite(a);if(!a.length)return [0,1];const lo=Math.min(...a),hi=Math.max(...a),s=Math.max(hi-lo,min);return [(lo+hi-s)/2-.08*s,(lo+hi+s)/2+.08*s];};
const stops=[[.267,.005,.329],[.23,.322,.546],[.128,.567,.551],[.369,.789,.383],[.993,.906,.144]];
function colour(t){const z=Math.max(0,Math.min(1,t))*4,i=Math.min(3,Math.floor(z)),a=z-i;return stops[i].map((v,k)=>v*(1-a)+stops[i+1][k]*a);}
const css=c=>`rgb(${c.map(v=>Math.round(255*v)).join(',')})`;

export function beamPanels(container,onSelect){
 const panels=[];let current=null,selection=new Set();
 for(let j=0;j<2;j++){
  const box=document.createElement('section');box.className='plot beam-plot';
  const axes=document.createElement('canvas'),canvas=document.createElement('canvas');canvas.className='gpu';canvas.tabIndex=0;
  canvas.setAttribute('aria-label',j?'Sanya transverse beam and SNR':'Sanya gain and radar cross section');
  canvas.addEventListener('pointerdown',()=>canvas.focus({preventScroll:true}));
  const extra=document.createElement('div');extra.className='beam-scale';
  const rightCanvas=document.createElement('canvas');rightCanvas.style.cssText='left:auto;right:0;width:84px;pointer-events:none';rightCanvas.hidden=!!j;
  box.append(axes,canvas,extra,rightCanvas);container.append(box);
  const panel={j,box,axesCanvas:axes,canvas,axes:new SeriesView(axes),rightAxes:new SeriesView(rightCanvas),gpu:new LinePlot(canvas),extra,bounds:null,points:[]};panels.push(panel);
  const navigate=(method,args)=>{if(!panel.bounds)return;const v=new PlotViewport(),b=panel.bounds,f=panel.full;
   v.zoom_box((b[0]-f[0])/(f[1]-f[0]),(b[2]-f[2])/(f[3]-f[2]),(b[1]-f[0])/(f[1]-f[0]),(b[3]-f[2])/(f[3]-f[2]));v[method](...args);const z=v.bounds();v.free();panel.bounds=[f[0]+z[0]*(f[1]-f[0]),f[0]+z[2]*(f[1]-f[0]),f[2]+z[1]*(f[3]-f[2]),f[2]+z[3]*(f[3]-f[2])];};
  bindPlotNavigation(canvas,{zoom:(...a)=>navigate('zoom',a),pan:(...a)=>navigate('pan',a),zoom_box:(...a)=>navigate('zoom_box',a),reset:()=>{panel.bounds=null;}},()=>draw(),()=>({x:70,y:30,width:box.clientWidth-153,height:box.clientHeight-78}));
  canvas.addEventListener('click',e=>{if(e.ctrlKey||e.metaKey||!current?.beam)return;const b=panel.bounds,w=box.clientWidth,h=box.clientHeight;let best=null,d=12;
   for(const p of panel.points){const x=70+(p.x-b[0])/(b[1]-b[0])*(w-153),y=h-48-(p.y-b[2])/(b[3]-b[2])*(h-78);if(x<70||x>w-83||y<30||y>h-48)continue;const r=Math.hypot(e.offsetX-x,e.offsetY-y);if(r<d){d=r;best=p.id;}}
   if(best!==null)onSelect(best);
  });
  new ResizeObserver(()=>draw()).observe(box);
 }
 function draw(){
  for(const v of panels){const d=current?.beam;v.box.hidden=!d;if(!d)continue;
   const w=v.box.clientWidth,h=v.box.clientHeight,excluded=new Set(current.annotation.excluded);if(w<200)continue;
   const xl=v.j?[-d.extent_m,d.extent_m]:limits(d.time_s,.01),yl=v.j?[-d.extent_m,d.extent_m]:limits([...d.relative_gain_db.map(g=>2*g),0,-20],3);
   if(v.j){const ratio=(w-153)/(h-78);if(ratio>1){xl[0]*=ratio;xl[1]*=ratio;}else{yl[0]/=ratio;yl[1]/=ratio;}}
   v.full=[...xl,...yl];v.bounds??=v.full.slice();const b=v.bounds;
   v.axesCanvas.style.width=`${w-65}px`;
   v.axes.begin_axes(w-65,h,devicePixelRatio,...b,v.j?'East–west offset (m)':'Time from reference epoch (s)',v.j?'North–south offset (m)':'Two-way gain relative to peak (dB)');v.axes.x_ticks(false,false);
   v.gpu.set_margins(70,83,30,48);v.gpu.begin(w,h,devicePixelRatio,...b);v.points=[];v.extra.replaceChildren();
   const line=(xy,c,size=1.4)=>v.gpu.series(new Float64Array(xy),...c,false,size,false);
   const marker=(i,x,y,c)=>{const id='0:'+d.time_ns[i];if(!Number.isFinite(x)||!Number.isFinite(y))return;v.points.push({id,x,y});
    if(selection.has(id))v.gpu.series(new Float64Array([x,y]),.05,.05,.05,true,13,false);
    v.gpu.series(new Float64Array([x,y]),...c,true,7,false);
    if(excluded.has(id)){const dx=(b[1]-b[0])*6/(w-153),dy=(b[3]-b[2])*6/(h-78);line([x-dx,y-dy,x+dx,y+dy],[.74,.15,.21],2);line([x-dx,y+dy,x+dx,y-dy],[.74,.15,.21],2);}
   };
   if(v.j){
    for(const contour of d.contours){const shade=contour.level===-3?.35:contour.level===-10?.62:.8;
     for(const path of contour.lines)line(path.flat(),[shade,shade,shade],contour.level===-3?1.8:1);
     const closed=contour.lines.filter(p=>p.length>15&&Math.hypot(p[0][0]-p.at(-1)[0],p[0][1]-p.at(-1)[1])<1);
     const centre=p=>Math.hypot(p.reduce((s,v)=>s+v[0],0)/p.length,p.reduce((s,v)=>s+v[1],0)/p.length);
     closed.sort((a,b)=>centre(a)-centre(b));
     if(closed.length){const top=closed[0].reduce((a,b)=>a[1]>b[1]?a:b);v.axes.node_label(...top,`${2*contour.level} dB`,'#687780');}
    }
    line(d.east_m.flatMap((x,i)=>[x,d.north_m[i]]),[.65,.68,.7],1);
    const s=limits(d.snr_db,1),col=x=>colour((x-s[0])/(s[1]-s[0]));
    d.east_m.forEach((x,i)=>marker(i,x,d.north_m[i],col(d.snr_db[i])));
    const bar=document.createElement('div');bar.className='snr-bar';bar.style.background=`linear-gradient(to top,${stops.map(css).join(',')})`;v.extra.append(bar);
    scale(v,s,'SNR (dB)');
   }else{
    line(d.time_s.flatMap((t,i)=>[t,2*d.relative_gain_db[i]]),[.15,.43,.68],2);
    line([b[0],-20,b[1],-20],[.65,.68,.7],1);
    const r=limits(d.rcs_dbsm,3);const map=y=>b[2]+(y-r[0])/(r[1]-r[0])*(b[3]-b[2]);
    for(const curve of current.rcs_mass?.curves||[])line(curve.time_s.flatMap((t,i)=>[t,map(curve.rcs_dbsm[i])]),[.72,.25,.07],1);
    d.rcs_dbsm.forEach((y,i)=>{if(Number.isFinite(y))marker(i,d.time_s[i],map(y),[.72,.25,.07]);});
    v.rightAxes.right_axis(84,h,devicePixelRatio,...r,'RCS (dBsm)');
   }
  }
 }
 function scale(v,values,label,orange=false){
  v.extra.classList.toggle('rcs-scale',orange);const h=v.box.clientHeight;
  for(let k=0;k<=4;k++){const tick=document.createElement('span');tick.className='beam-tick';tick.style.top=`${30+(h-78)*(1-k/4)-7}px`;tick.textContent=(values[0]+(values[1]-values[0])*k/4).toFixed(1);v.extra.append(tick);}
  const text=document.createElement('span');text.className='beam-axis-label';text.textContent=label;v.extra.append(text);
 }
 return {update(data,selected){if(current?.event!==data.event||current?.fingerprint!==data.fingerprint)panels.forEach(v=>v.bounds=null);current=data;selection=selected;draw();},reset(){panels.forEach(v=>v.bounds=null);draw();}};
}
