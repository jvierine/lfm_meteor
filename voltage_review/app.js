import init, {SeriesView, LinePlot, PlotViewport} from '/pkg/rustplotlib_web.js';
import {bindPlotNavigation} from '/pkg/navigation.js';
import {beamPanels} from './beam.js';
const $=id=>document.getElementById(id);
const names=['Sanya','Danzhou','Wenchang'],colors=[[0,.447,.698],[.835,.369,0],[0,.62,.451]];
let token=decodeURIComponent(location.hash.slice(1))||sessionStorage.getItem('review-token')||'';
if(location.hash)history.replaceState(null,'',location.pathname);
let catalog=[],data=null,selected=new Set(),undo=[],busy=false,request=0,views=[],xview=null;
let beamView=null;
const status=(text,error=false)=>{$('connection').textContent=text;$('connection').classList.toggle('error',error);};
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{Authorization:'Bearer '+token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});const value=await r.json();if(!r.ok)throw Error(value.error||r.statusText);return value;}
function filtered(){const q=$('search').value.toLowerCase(),mode=$('filter').value;return catalog.filter(e=>e.event.includes(q)&&(mode==='all'||mode==='flagged'&&(e.flagged||e.bad)));}
function sidebar(){const root=$('events');root.replaceChildren();for(const e of filtered()){const b=document.createElement('button');b.textContent=e.event;b.className=data?.event===e.event?'active':'';const small=document.createElement('small');small.textContent=`${e.n} points${e.flagged?' · '+e.flagged+' flagged':''}${e.bad?' · manually marked bad':''}`;b.append(small);b.onclick=()=>load(e.event);root.append(b);} $('counts').textContent=`${catalog.length} fitted events`;}
function controls(){for(const id of ['flag','restore','clear'])$(id).disabled=busy||!selected.size;$('undo').disabled=busy||!undo.length;$('bad').disabled=busy||!data;$('savenote').disabled=busy||!data;$('selection').textContent=`${selected.size} selected`;$('bad').checked=!!data?.annotation.bad;}
async function refresh(){try{catalog=await api('/api/events');sidebar();status('Connected');$('login').hidden=true;if(data&&!busy)await load(data.event);else if(!data&&catalog.length)await load(catalog[0].event);}catch(e){status(e.message,true);$('login').hidden=false;}}
async function load(id){if(busy)return;const version=++request;status('Loading');try{const next=await api('/api/event/'+id);if(version!==request)return;data=next;selected.clear();undo=[];xview=null;views.forEach(v=>v.bounds=null);$('eventname').textContent=id;const rms=Math.sqrt(next.residual_beat_hz.reduce((a,x)=>a+x*x,0)/next.ids.length);$('quality').textContent=`${next.ids.length} measurements · residual beat RMS ${rms.toFixed(1)} Hz · ${next.uncertainty_status}`;$('note').value=next.annotation.note;$('saved').textContent=`Saved revision ${next.annotation.version}`;sidebar();table();draw();controls();status('Connected');}catch(e){status(e.message,true);}}
async function save(change,isUndo=false){if(!data||busy)return;const previous=structuredClone(data.annotation);const updated={...previous,...change};busy=true;controls();$('saved').textContent='Saving';try{data.annotation=await api('/api/flags/'+data.event,{...updated,fingerprint:data.fingerprint});if(!isUndo)undo.push(previous);else undo.pop();const entry=catalog.find(e=>e.event===data.event);if(entry){entry.flagged=data.annotation.excluded.length;entry.bad=data.annotation.bad;entry.version=data.annotation.version;}$('note').value=data.annotation.note;$('saved').textContent=`Saved revision ${data.annotation.version}`;sidebar();table();draw();}catch(e){$('saved').textContent='Not saved';status(e.message,true);}finally{busy=false;controls();}}
function table(){const root=$('points');root.replaceChildren();if(!data)return;const excluded=new Set(data.annotation.excluded);data.ids.forEach((id,i)=>{const row=document.createElement('tr');row.classList.toggle('flagged',excluded.has(id));const cell=document.createElement('td'),check=document.createElement('input');check.type='checkbox';check.checked=selected.has(id);check.setAttribute('aria-label','Select '+names[data.station[i]]+' pulse '+data.time_ns[i]);check.onchange=()=>{check.checked?selected.add(id):selected.delete(id);draw();controls();};cell.append(check);row.append(cell);for(const text of [names[data.station[i]],data.time_ns[i],data.time_s[i].toFixed(6),data.residual_beat_hz[i].toFixed(1),excluded.has(id)?'Excluded':'Included']){const c=document.createElement('td');c.textContent=text;row.append(c);}root.append(row);});omittedTable();}
function point(i){return i<0?data.omitted_measurements[-i-1]:{id:data.ids[i],station:data.station[i],time_s:data.time_s[i],time_ns:data.time_ns[i],residual_beat_hz:data.residual_beat_hz[i]};}
function eventTimes(){return [...data.time_s,...(data.omitted_measurements||[]).map(p=>p.time_s)];}
function range(values,minspan=1){let lo=Math.min(...values),hi=Math.max(...values),span=Math.max(hi-lo,minspan);return [(lo+hi-span)/2-.08*span,(lo+hi+span)/2+.08*span];}
function xy(indices,ys){return new Float64Array(indices.flatMap(i=>[data.time_s[i],ys[i]]));}
function omittedTable(){
 const excluded=new Set(data.annotation.excluded);
 for(const p of data.omitted_measurements||[]){
  const row=document.createElement('tr');row.classList.toggle('flagged',excluded.has(p.id));
  const cell=document.createElement('td'),check=document.createElement('input');check.type='checkbox';check.checked=selected.has(p.id);check.setAttribute('aria-label','Select excluded '+names[p.station]+' pulse '+p.time_ns);
  check.onchange=()=>{check.checked?selected.add(p.id):selected.delete(p.id);draw();controls();};cell.append(check);row.append(cell);
  for(const value of [names[p.station],p.time_ns,p.time_s.toFixed(6),'—',excluded.has(p.id)?'Excluded from fit':'Restored; refit pending']){const td=document.createElement('td');td.textContent=value;row.append(td);}
  $('points').append(row);
 }
}
function uncertaintyMath(symbol,lo,best,hi,unit,scientific){
 if(!Number.isFinite(best)||best<=0)return '<span>Estimate unavailable</span>';
 const exponent=scientific?Number(best.toExponential().split('e')[1]):0,factor=10**exponent;
 const number=x=>Number(x.toPrecision(3)).toLocaleString('en-US',{useGrouping:false,maximumSignificantDigits:3});
 const power=exponent===0?'':`<mo>⋅</mo><msup><mn>10</mn><mn>${exponent}</mn></msup>`;
 const valid=Number.isFinite(lo)&&Number.isFinite(hi)&&lo>=0&&hi>=lo;
 const interval=valid?`<mspace width="0.3em"/><mo>(</mo><mn>${number(lo/factor)}</mn><mo>,</mo><mn>${number(hi/factor)}</mn><mo>)</mo>`:'';
 return `<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>${symbol}</mi><mn>0</mn></msub><mo>=</mo><mn>${number(best/factor)}</mn>${interval}${power}<mspace width="0.3em"/><mtext>${unit}</mtext><mspace width="0.6em"/><mtext>${valid?'(95%)':'(interval unavailable)'}</mtext></mrow></math>`;
}
function fitSummary(){
 $('fitstamp').textContent=data.fit_unix_time?`Fit: ${new Date(data.fit_unix_time*1000).toISOString()} · flag revision ${data.fit_flag_revision||0} · ${data.fit_excluded_count||0} excluded · ${data.fit_timestamp_source}${data.annotation.version>(data.fit_flag_revision||0)?' · newer review revision exists':''}`:'Fit timestamp unavailable';
 const root=$('masssummary');root.replaceChildren();
 for(const [symbol,key,bestKey,scale,unit,scientific] of [['m','initial_mass_kg_quantiles','best_initial_mass_kg',1,'kg',true],['d','initial_radius_m_quantiles','best_initial_radius_m',2e6,'µm',false]]){
  const q=data[key];if(!q)continue;
  const span=document.createElement('span');const [lo,mid,hi]=q.map(v=>v*scale);
  const best=data[bestKey];
  span.className='mass-estimate';span.innerHTML=uncertaintyMath(symbol,lo,best===undefined?mid:best*scale,hi,unit,scientific);if(best===undefined)span.append(document.createTextNode(' (posterior median)'));root.append(span);
 }
 const note=document.createElement('span');note.className='mass-status';note.textContent=data.mass_status||'Initial mass and radius posterior unavailable';root.append(note);
 const r=data.rcs_mass;
 if(r?.mass_kg>0){
  const span=document.createElement('span');span.className='mass-estimate';
  span.innerHTML=uncertaintyMath('m',r.lo95_scatter_kg,r.mass_kg,r.hi95_scatter_kg,'kg',true).replace('<mn>0</mn>','<mtext>RCS</mtext>');
  root.append(span);
  const diameter=m=>1e6*Math.cbrt(6*m/(Math.PI*3000));
  const size=document.createElement('span');size.className='mass-estimate';
  size.innerHTML=uncertaintyMath('d',Number.isFinite(r.lo95_scatter_kg)?diameter(r.lo95_scatter_kg):null,diameter(r.mass_kg),Number.isFinite(r.hi95_scatter_kg)?diameter(r.hi95_scatter_kg):null,'µm',false).replace('<mn>0</mn>','<mtext>RCS</mtext>');
  root.append(size);const detail=document.createElement('span');detail.className='mass-status';
  detail.textContent=`RCS RMS scatter: ${r.rcs_rms_db?.toFixed(2)} dB. 95% empirical scatter interval, using a shared Gaussian RCS offset; not total physical uncertainty. Observed mass loss over ${r.covered_duration_s.toFixed(3)} s; equivalent diameter at 3000 kg/m³, not initial diameter; gain ≥ −10 dB one-way.${r.uncertainty_extrapolated_fraction>0?' Uncertainty tails use extrapolated lookup edge slopes.':''}${r.lo95_scatter_kg==null?' Interval unavailable: ambiguous inversion.':''}${r.covered_duration_s<.999*r.requested_duration_s?' Partial coverage.':''}`;root.append(detail);
 }
}
function navigate(v,method,args){
 if(!data||!v.bounds)return;
 const bx=range(eventTimes(),.01),by=v.fullBounds,sx=bx[1]-bx[0],sy=by[1]-by[0];
 const viewport=new PlotViewport();
 viewport.zoom_box((xview[0]-bx[0])/sx,(v.bounds[0]-by[0])/sy,(xview[1]-bx[0])/sx,(v.bounds[1]-by[0])/sy);
 viewport[method](...args);const b=viewport.bounds();viewport.free();
 xview=[bx[0]+b[0]*sx,bx[0]+b[2]*sx];v.bounds=[by[0]+b[1]*sy,by[0]+b[3]*sy];
}
function setup(){for(let j=0;j<4;j++){
 const box=document.createElement('section');box.className='plot';
 const axes=document.createElement('canvas'),gpu=document.createElement('canvas');gpu.className='gpu';gpu.tabIndex=0;
 gpu.setAttribute('aria-label',['Apparent path measurements','Fitted speed','Path residual measurements','Beat residual measurements'][j]);
 box.append(axes,gpu);$('plots').append(box);
 const v={box,canvas:gpu,axes:new SeriesView(axes),gpu:new LinePlot(gpu),bounds:null,points:[],j};views.push(v);v.gpu.set_margins(70,18,30,48);
 gpu.addEventListener('pointerdown',()=>gpu.focus());
 bindPlotNavigation(gpu,{zoom:(...a)=>navigate(v,'zoom',a),pan:(...a)=>navigate(v,'pan',a),zoom_box:(...a)=>navigate(v,'zoom_box',a),reset},draw,
   ()=>({x:70,y:30,width:box.clientWidth-88,height:box.clientHeight-78}));
 gpu.addEventListener('pointermove',e=>{const pick=nearest(v,e.offsetX,e.offsetY);const p=pick===null?null:point(pick);$('hover').textContent=p===null?' ':`${names[p.station]} · ${p.time_ns} ns · t=${p.time_s.toFixed(6)} s${pick<0?' · excluded from fit':''}`;});
 gpu.addEventListener('click',e=>{if(e.ctrlKey||e.metaKey)return;const i=nearest(v,e.offsetX,e.offsetY);if(i!==null){const id=point(i).id;selected.has(id)?selected.delete(id):selected.add(id);table();draw();controls();}});
 new ResizeObserver(()=>draw()).observe(box);
}}
function project(v,x,y){return [70+(x-xview[0])/(xview[1]-xview[0])*(v.box.clientWidth-88),v.box.clientHeight-48-(y-v.bounds[0])/(v.bounds[1]-v.bounds[0])*(v.box.clientHeight-78)];}
function nearest(v,x,y){if(!data||v.j===1)return null;let best=null,distance=12;for(const [i,yy] of v.points){const [px,py]=project(v,point(i).time_s,yy);if(px<70||px>v.box.clientWidth-18||py<30||py>v.box.clientHeight-48)continue;const d=Math.hypot(px-x,py-y);if(d<distance){distance=d;best=i;}}return best;}
function draw(){if(!data||!views.length)return;const n=data.ids.length,offset=Array(n),model=Array(n),res=data.modeled_path_m.map((m,i)=>m-data.seed_apparent_path_m[i]);const groups=names.map((_,s)=>data.station.flatMap((v,i)=>v===s?[i]:[]).sort((a,b)=>data.time_s[a]-data.time_s[b]));groups.forEach(ids=>{const b=data.seed_apparent_path_m[ids[0]];ids.forEach(i=>{offset[i]=(data.seed_apparent_path_m[i]-b)/1000;model[i]=(data.modeled_path_m[i]-b)/1000;});});const ys=[offset,data.speed_km_s,res,data.residual_beat_hz],bounds=[range(offset,.1),range(data.speed_km_s,.1),range(res,1),range(data.residual_beat_hz.flatMap((b,i)=>[b-1.96*data.conditional_beat_std_hz[i],b+1.96*data.conditional_beat_std_hz[i]]),100)];xview??=range(data.time_s,.01);const excluded=new Set(data.annotation.excluded);
fitSummary();
beamView?.update(data,selected);
if(views.every(v=>v.bounds===null))xview=range(eventTimes(),.01);
const ghostY=(p,j)=>j===0?(p.path_m-data.seed_apparent_path_m[groups[p.station][0]])/1000:j===2?(Number.isFinite(p.modeled_path_m)?p.modeled_path_m-p.path_m:NaN):j===3?p.residual_beat_hz:NaN;
for(const j of [0,2,3]){const ghost=(data.omitted_measurements||[]).map(p=>ghostY(p,j)).filter(Number.isFinite);if(ghost.length)bounds[j]=range([...bounds[j],...ghost],j===0?.1:1);}
if(data.path_measurement_lower95_m){const edges=[],resEdges=[];groups.forEach(ids=>{const b=data.seed_apparent_path_m[ids[0]];ids.forEach(i=>{edges.push((data.path_measurement_lower95_m[i]-b)/1000,(data.path_measurement_upper95_m[i]-b)/1000);resEdges.push(data.modeled_path_m[i]-data.path_measurement_upper95_m[i],data.modeled_path_m[i]-data.path_measurement_lower95_m[i]);});});bounds[0]=range([...offset,...edges],.1);bounds[2]=range([...res,...resEdges],1);}
if(data.path_lower95_m){const edges=[];groups.forEach(ids=>{const b=data.seed_apparent_path_m[ids[0]];ids.forEach(i=>edges.push((data.path_lower95_m[i]-b)/1000,(data.path_upper95_m[i]-b)/1000));});bounds[0]=range([...offset,...model,...edges],.1);bounds[1]=range([...data.speed_km_s,...data.speed_lower95_km_s,...data.speed_upper95_km_s],.1);}
for(const j of [0,2,3]){const ghost=(data.omitted_measurements||[]).map(p=>ghostY(p,j)).filter(Number.isFinite);if(ghost.length)bounds[j]=range([...bounds[j],...ghost],j===0?.1:1);}
for(const v of views){const j=v.j;v.fullBounds=bounds[j];v.bounds??=bounds[j];v.canvas.dataset.xlim=JSON.stringify(xview);v.canvas.dataset.ylim=JSON.stringify(v.bounds);const w=v.box.clientWidth,h=v.box.clientHeight,dpr=devicePixelRatio;v.axes.begin_axes(w,h,dpr,...xview,...v.bounds,'Time from reference epoch (s)',['Apparent path offset (km)','Speed (km/s)','Path residual (m)','Residual beat (Hz)'][j]);v.axes.x_ticks(false,false);v.gpu.begin(w,h,dpr,...xview,...v.bounds);v.points=[];const line=(points,c,size=1.5)=>v.gpu.series(new Float64Array(points),...c,false,size,false);if(j>=2)line([xview[0],0,xview[1],0],[.65,.68,.7],1);
if(data.path_lower95_m&&j===0)groups.forEach((ids,s)=>{const b=data.seed_apparent_path_m[ids[0]];v.gpu.band(new Float64Array(ids.flatMap(i=>[data.time_s[i],(data.path_lower95_m[i]-b)/1000,(data.path_upper95_m[i]-b)/1000])),...colors[s],.5);});
if(data.speed_lower95_km_s&&j===1)v.gpu.band(new Float64Array(data.speed_time_s.flatMap((t,i)=>[t,data.speed_lower95_km_s[i],data.speed_upper95_km_s[i]])),.4,.4,.4,.5);
if(j!==1)(data.omitted_measurements||[]).forEach((p,i)=>{const y=ghostY(p,j);if(!Number.isFinite(y))return;v.points.push([-i-1,y]);const dx=(xview[1]-xview[0])*6/(w-88),dy=(v.bounds[1]-v.bounds[0])*6/(h-78),c=excluded.has(p.id)?[.74,.15,.21]:[.5,.5,.5];if(selected.has(p.id))v.gpu.series(new Float64Array([p.time_s,y]),.1,.1,.1,true,15,false);line([p.time_s-dx,y-dy,p.time_s+dx,y+dy],c,2);line([p.time_s-dx,y+dy,p.time_s+dx,y-dy],c,2);});
if(data.path_measurement_lower95_m&&(j===0||j===2))groups.forEach((ids,s)=>{const b=data.seed_apparent_path_m[ids[0]];for(const i of ids){const lo=j===0?(data.path_measurement_lower95_m[i]-b)/1000:data.modeled_path_m[i]-data.path_measurement_upper95_m[i],hi=j===0?(data.path_measurement_upper95_m[i]-b)/1000:data.modeled_path_m[i]-data.path_measurement_lower95_m[i];line([data.time_s[i],lo,data.time_s[i],hi],colors[s],1);}});
if(j===1){line(data.speed_time_s.flatMap((t,i)=>[t,data.speed_km_s[i]]),[.15,.18,.2],2);continue;}
groups.forEach((ids,s)=>{if(j===0)line([...xy(ids,model)],colors[s]);if(j===3)for(const i of ids){const lo=ys[j][i]-1.96*data.conditional_beat_std_hz[i],hi=ys[j][i]+1.96*data.conditional_beat_std_hz[i];line([data.time_s[i],lo,data.time_s[i],hi],colors[s],.8);}const active=ids.filter(i=>!excluded.has(data.ids[i]));v.gpu.series(xy(active,ys[j]),...colors[s],true,6,false);for(const i of ids){v.points.push([i,ys[j][i]]);if(selected.has(data.ids[i])){v.gpu.series(xy([i],ys[j]),.05,.08,.12,true,12,false);v.gpu.series(xy([i],ys[j]),...colors[s],true,7,false);}if(excluded.has(data.ids[i])){const dx=(xview[1]-xview[0])*5/(w-88),dy=(v.bounds[1]-v.bounds[0])*5/(h-78),t=data.time_s[i],y=ys[j][i];line([t-dx,y-dy,t+dx,y+dy],[.74,.15,.21],2);line([t-dx,y+dy,t+dx,y-dy],[.74,.15,.21],2);}}});}}
function reset(){xview=null;views.forEach(v=>v.bounds=null);beamView?.reset();draw();}
function move(step){if(!data)return;const list=filtered(),i=list.findIndex(e=>e.event===data.event),next=list[i+step];if(next)load(next.event);}
$('search').oninput=sidebar;$('filter').onchange=sidebar;$('refresh').onclick=refresh;$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('reset').onclick=reset;
$('flag').onclick=()=>save({excluded:[...new Set([...data.annotation.excluded,...selected])]});$('restore').onclick=()=>save({excluded:data.annotation.excluded.filter(id=>!selected.has(id))});$('clear').onclick=()=>{selected.clear();table();draw();controls();};$('undo').onclick=()=>undo.length&&save({...undo.at(-1),version:data.annotation.version},true);$('bad').onchange=e=>save({bad:e.target.checked});$('savenote').onclick=()=>save({note:$('note').value});$('connect').onclick=()=>{token=$('token').value.trim();sessionStorage.setItem('review-token',token);refresh();};document.addEventListener('keydown',e=>{const typing=['TEXTAREA','SELECT'].includes(e.target.tagName)||(e.target.tagName==='INPUT'&&!['checkbox','radio','button'].includes(e.target.type))||e.target.isContentEditable;if(typing||e.ctrlKey||e.metaKey||e.altKey||e.repeat)return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);if(e.key.toLowerCase()==='c')$('clear').click();if(e.key.toLowerCase()==='f'){$('flag').click();e.preventDefault();}if(e.key.toLowerCase()==='b'&&data)save({bad:!data.annotation.bad});});
try{await init();setup();beamView=beamPanels($('plots'),id=>{selected.has(id)?selected.delete(id):selected.add(id);table();draw();controls();});if(token)sessionStorage.setItem('review-token',token);await refresh();}catch(e){status('Viewer failed: '+e.message,true);}
setInterval(async()=>{
 if(!data||busy||selected.size||document.hidden||$('note').value!==data.annotation.note)return;
 const id=data.event;
 try{
  const next=await api('/api/event/'+id);
  if(busy||selected.size||data.event!==id||$('note').value!==data.annotation.note)return;
  const changed=next.fingerprint!==data.fingerprint||next.uncertainty_status!==data.uncertainty_status||!!next.path_measurement_lower95_m!==!!data.path_measurement_lower95_m||JSON.stringify(next.omitted_measurements)!==JSON.stringify(data.omitted_measurements)||JSON.stringify(next.beam)!==JSON.stringify(data.beam)||JSON.stringify(next.rcs_mass)!==JSON.stringify(data.rcs_mass);
  if(changed)await load(id);
 }catch(e){status(e.message,true);}
},30000);
