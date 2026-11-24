"""Live mockup-style dashboard with a continuous, animated field-coverage view."""
from __future__ import annotations

import json

import streamlit.components.v1 as components

from coverageai.field_anim import field_animation_payload

_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#0e1117;color:#c9d1d9;
font:12px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}
#app{height:100%;display:flex;flex-direction:column;gap:7px;padding:3px}
#dash{display:grid;grid-template-columns:minmax(170px,1.05fr) minmax(260px,2fr) minmax(190px,1.05fr);
gap:8px;flex:1;min-height:180px}
.col{display:flex;flex-direction:column;gap:7px;min-width:0}
.card{background:#161b22;border:1px solid #2b3340;border-radius:8px;overflow:hidden}
.title{background:#203528;color:#e6edf3;font-weight:650;padding:7px 9px;border-bottom:1px solid #2b3340}
table{width:100%;border-collapse:collapse;padding:6px 9px}
td{padding:4px 8px;vertical-align:top}td:first-child{color:#8a94a6}td:last-child{text-align:right;color:#e6edf3}
#visual{position:relative;min-width:0;min-height:180px;background:#10151c;border:1px solid #2b3340;border-radius:9px;overflow:hidden}
canvas{display:block;width:100%;height:100%}
#error{display:none;position:absolute;inset:8px;color:#ffb4a9;background:#291b1b;padding:8px;border-radius:6px}
.readings-header{display:flex;justify-content:space-between;align-items:center;color:#8a94a6;font-size:10px}
.readings{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}
.reading{aspect-ratio:1;max-width:112px;width:100%;justify-self:center;border:2px solid #303846;
border-radius:50%;background:#1b222c;display:flex;flex-direction:column;align-items:center;
justify-content:center;text-align:center;padding:7px}
.reading .label{font-size:10px;color:#8a94a6}
.reading .value{font-size:13px;font-weight:650;color:#e6edf3;margin-top:3px}
.reading .adjustment{font-size:9px;color:#b8e986;margin-top:2px;min-height:12px}
.reading .value.changed,.reading .adjustment.changed{animation:reading-change .5s ease-out}
@keyframes reading-change{45%{color:#3ddc97;transform:scale(1.15)}}
.live{display:inline-block;color:#3ddc97;font-size:9px;font-weight:700;letter-spacing:.04em}
#coverage{color:#b8e986;font-weight:700;font-size:15px}
#feedback{margin:7px;padding:8px;border-radius:7px;background:#202733;line-height:1.45}
#feedback.good{border-left:3px solid #3ddc97}#feedback.needs{border-left:3px solid #f5b942}
#controls{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
button{background:#1f2630;color:#e6edf3;border:1px solid #2b3340;border-radius:6px;
padding:3px 8px;font:inherit;cursor:pointer}
button.on{background:#2f6fed;border-color:#2f6fed}
#speeds{display:flex;gap:3px}#scrub{flex:1;min-width:70px;accent-color:#4da3ff}
#status{color:#8a94a6;min-width:80px;text-align:right}
@media(max-width:760px){#dash{grid-template-columns:minmax(130px,1fr) minmax(190px,1.5fr);
grid-template-rows:minmax(150px,1fr) auto}.left{grid-column:1}.right{grid-column:2;grid-row:1 / span 2}
#visual{grid-column:1;grid-row:2;min-height:140px}.readings{grid-template-columns:repeat(2,minmax(0,1fr))}
.reading{max-width:84px}}
</style></head><body><div id="app">
<div id="dash">
 <div class="col left">
  <div class="card"><div class="title">Environmental Conditions</div><table>
   <tr><td>Weather</td><td id="weather">—</td></tr>
   <tr><td>Wind speed</td><td id="wind">—</td></tr>
   <tr><td>Wind direction</td><td id="winddir">—</td></tr>
   <tr><td>Temperature</td><td id="temp">—</td></tr>
   <tr><td>Humidity</td><td id="humidity">—</td></tr></table></div>
  <div class="card"><div class="title">Machine Target Setup</div><table>
   <tr><td>Travel speed</td><td id="targetspeed">—</td></tr>
   <tr><td>Boom height</td><td id="targetboom">—</td></tr></table></div>
  <div class="card"><div class="title">Spray System Setup</div><table>
   <tr><td>Nozzle</td><td id="nozzle">—</td></tr>
   <tr><td>Application rate</td><td id="targetgpa">—</td></tr>
   <tr><td>Pressure</td><td id="targetpressure">—</td></tr>
   <tr><td>Target coverage</td><td id="targetcoverage">—</td></tr></table></div>
 </div>
 <div id="visual"><canvas id="field"></canvas><div id="error" role="alert"></div></div>
 <div class="col right">
  <div class="card"><div class="title">CoverageVision Correction</div><table>
   <tr><td>Travel speed</td><td id="speed">—</td></tr>
   <tr><td>Boom height</td><td id="boom">—</td></tr>
   <tr><td>Pressure</td><td id="pressure">—</td></tr>
   <tr><td>Live coverage reading <span class="live">LIVE</span></td><td id="coverage">—</td></tr>
   <tr><td>Coverage vs goal</td><td id="improvement">—</td></tr>
   <tr><td>Predicted drift loss</td><td id="drift">—</td></tr></table>
   <div id="feedback"></div></div>
 </div>
</div>
<div class="readings-header"><span class="live">LIVE FEEDBACK ADJUSTMENTS</span><span id="reading-section"></span></div>
<div class="readings">
 <div class="reading"><span class="label">Pressure</span><span class="value" id="rpressure">—</span>
  <span class="adjustment" id="apressure">—</span></div>
 <div class="reading"><span class="label">Travel speed</span><span class="value" id="rspeed">—</span>
  <span class="adjustment" id="aspeed">—</span></div>
 <div class="reading"><span class="label">Application rate</span><span class="value" id="rgpa">—</span>
  <span class="adjustment" id="agpa">—</span></div>
 <div class="reading"><span class="label">Boom height</span><span class="value" id="rboom">—</span>
  <span class="adjustment" id="aboom">—</span></div>
 <div class="reading"><span class="label">Coverage</span><span class="value" id="rcoverage">—</span>
  <span class="adjustment" id="acoverage">—</span></div>
</div>
<div id="controls"><button id="play">Pause</button><button id="reset">Reset</button>
 <span id="speeds"><button data-speed="1" class="on">1×</button><button data-speed="2">2×</button>
 <button data-speed="4">4×</button></span>
 <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Field position">
 <span id="status"></span></div></div>
<script>
window.onerror=(message)=>{const e=document.getElementById('error');e.textContent='Dashboard display error: '+message;e.style.display='block'};
const D=__PAYLOAD__,S=D.sections,N=S.length,rows=D.rows,cols=D.cols;
const canvas=document.getElementById('field'),ctx=canvas.getContext('2d');
const VIR=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
 [31,158,137],[53,183,121],[110,206,88],[253,231,37]];
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function viridis(v){const u=clamp(v,0,1)*8,i=Math.min(7,Math.floor(u)),f=u-i,a=VIR[i],b=VIR[i+1];
 return[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f]}
const coverageMax=Math.max(D.target*1.5,...S.map(s=>s.actual),1);
let W=0,H=0,dpr=1,t=0,speed=1,playing=true,last=null,field=null;
function buildSurface(){
 const w=480,h=Math.max(100,Math.min(220,rows*32)),c=document.createElement('canvas');
 c.width=w;c.height=h;const cx=c.getContext('2d'),img=cx.createImageData(w,h);
 for(let py=0;py<h;py++)for(let px=0;px<w;px++){
  const x=(px+.5)/w,y=(py+.5)/h;let sum=0,weights=0;
  for(const s of S){const dx=(x-(s.col+.5)/cols)*cols,dy=(y-(s.row+.5)/rows)*rows;
   const weight=1/(dx*dx+dy*dy+.08);sum+=weight*s.actual;weights+=weight}
  const color=viridis(sum/weights/coverageMax),i=(py*w+px)*4;
  img.data[i]=color[0];img.data[i+1]=color[1];img.data[i+2]=color[2];img.data[i+3]=255}
 cx.putImageData(img,0,0);field=c}
function resize(){
 const r=canvas.getBoundingClientRect();dpr=window.devicePixelRatio||1;W=r.width;H=r.height;
 canvas.width=Math.max(1,Math.round(W*dpr));canvas.height=Math.max(1,Math.round(H*dpr));
 buildSurface()}
function roundRect(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);
 ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath()}
function drawField(b){
 ctx.save();roundRect(b.x,b.y,b.w,b.h,7);ctx.clip();
 ctx.fillStyle='#153721';ctx.fillRect(b.x,b.y,b.w,b.h);
 for(let i=0;i<rows*7;i++){const y=b.y+(i+.5)/(rows*7)*b.h;
  ctx.strokeStyle=i%2?'rgba(147,190,103,.12)':'rgba(8,24,16,.22)';
  ctx.beginPath();ctx.moveTo(b.x,y);ctx.lineTo(b.x+b.w,y);ctx.stroke()}
 const complete=t>=N?N:Math.floor(t),s=complete<N?S[complete]:null,cw=b.w/cols,ch=b.h/rows;
 ctx.beginPath();
 for(let k=0;k<complete;k++){const a=S[k];ctx.rect(b.x+a.col*cw,b.y+a.row*ch,cw,ch)}
 if(s){const f=t>0?t-complete:.025,left=b.x+s.col*cw,x=s.dir>0?left:left+cw*(1-f);
  ctx.rect(x,b.y+s.row*ch,cw*f,ch)}
 ctx.clip();ctx.drawImage(field,b.x,b.y,b.w,b.h);ctx.restore()}
function drawSprayer(b){
 if(t>=N)return;const k=Math.min(Math.floor(t),N-1),f=t-k,s=S[k];
 const x=b.x+(s.dir>0?s.col+f:s.col+1-f)/cols*b.w,y=b.y+(s.row+.5)/rows*b.h;
 const boom=Math.max(32,b.h/rows*.84),body=Math.max(14,Math.min(23,b.h/rows*.44));
 ctx.save();ctx.translate(x,y);if(s.dir<0)ctx.scale(-1,1);
 ctx.fillStyle='rgba(185,230,255,.18)';ctx.beginPath();ctx.moveTo(-body*.48,-boom*.46);
 ctx.lineTo(-body*2.8,-boom*.53);ctx.lineTo(-body*2.8,boom*.53);
 ctx.lineTo(-body*.48,boom*.46);ctx.closePath();ctx.fill();
 ctx.strokeStyle='#d9e8ef';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(0,-boom/2);ctx.lineTo(0,boom/2);ctx.stroke();
 ctx.fillStyle='#101820';ctx.fillRect(-body*.4,-body*.78,body*.8,body*.2);
 ctx.fillRect(-body*.4,body*.58,body*.8,body*.2);
 ctx.fillStyle='#e65d50';roundRect(-body*.42,-body*.56,body*.84,body*1.12,4);ctx.fill();
 ctx.fillStyle='#c9d1d9';roundRect(-body*.28,-body*.35,body*.56,body*.3,2);ctx.fill();
 ctx.restore()}
const put=(id,value)=>{document.getElementById(id).textContent=value};
function putLive(id,value){
 const el=document.getElementById(id);
 if(el.textContent===value)return;
 el.textContent=value;el.classList.remove('changed');void el.offsetWidth;el.classList.add('changed')}
function adjustment(value,target,unit,precision){
 if(!Number.isFinite(target))return 'target unavailable';
 const delta=value-target;if(Math.abs(delta)<.05)return 'at target';
 return (delta>0?'+':'')+delta.toFixed(precision)+' '+unit+' vs target'}
function updatePanel(){
 const s=S[Math.min(Math.floor(t),N-1)];if(!s)return;
 put('weather',D.setup.weather);put('wind',s.wind_speed_ms.toFixed(1)+' m/s');
 put('winddir',s.wind_rel_deg.toFixed(0)+'° relative');
 put('temp',s.temp_c.toFixed(1)+' °C');put('humidity',s.rh_pct.toFixed(0)+' %');
 put('targetspeed',D.setup.speed+' mph');put('targetboom',D.setup.boom+' in');
 put('nozzle',D.setup.nozzle);put('targetgpa',D.setup.gpa+' gal/acre');
 put('targetpressure',D.setup.pressure+' psi');put('targetcoverage',D.target+' %');
 put('speed',s.speed_mph.toFixed(1)+' mph');put('boom',s.boom_height_in.toFixed(0)+' in');
 put('pressure',s.pressure_psi.toFixed(0)+' psi');put('coverage',s.card.toFixed(1)+' %');
 const delta=s.card-D.target;
 put('improvement',(delta>=0?'+':'')+delta.toFixed(1)+' pp vs target');
 put('drift',s.predicted_drift_pct.toFixed(1)+' %');
 put('reading-section','Section '+s.number+' / '+N);
 putLive('rpressure',s.pressure_psi.toFixed(0)+' psi');putLive('rspeed',s.speed_mph.toFixed(1)+' mph');
 putLive('rgpa',s.rate_gpa.toFixed(1)+' GPA');putLive('rboom',s.boom_height_in.toFixed(1)+' in');
 putLive('rcoverage',s.card.toFixed(1)+' %');
 putLive('apressure',adjustment(s.pressure_psi,Number(D.setup.pressure),'psi',1));
 putLive('aspeed',adjustment(s.speed_mph,Number(D.setup.speed),'mph',1));
 putLive('agpa',adjustment(s.rate_gpa,Number(D.setup.gpa),'GPA',1));
 putLive('aboom',adjustment(s.boom_height_in,Number(D.setup.boom),'in',1));
 putLive('acoverage',adjustment(s.card,D.target,'pp',1));
 const feedback=document.getElementById('feedback'),lines=s.advice;
 feedback.className=lines.length?'needs':'good';
 feedback.textContent=lines.length?'Feedback: '+lines.join(' · '):
  (s.card>=D.target?'Card reading meets target. No change recommended.':'Card reading is below target; review the model recommendation below.');
}
function draw(){
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 const pad=Math.max(8,W*.025),b={x:pad,y:8,w:W-2*pad,h:H-16};
 drawField(b);drawSprayer(b);ctx.strokeStyle='#78927a';ctx.lineWidth=1;
 roundRect(b.x+.5,b.y+.5,b.w-1,b.h-1,7);ctx.stroke();
 const progress=clamp(t/N,0,1),y=b.y+b.h-3;
 ctx.fillStyle='rgba(14,17,23,.7)';ctx.fillRect(b.x,y,b.w,3);
 ctx.fillStyle='#b8e986';ctx.fillRect(b.x,y,b.w*progress,3);
 put('status',t>=N?'Pass complete':`Section ${Math.min(Math.floor(t)+1,N)} / ${N}`);
 const scrub=document.getElementById('scrub');
 if(document.activeElement!==scrub)scrub.value=Math.round(progress*1000);
 updatePanel()}
function animate(now){
 if(last===null)last=now;const dt=Math.min(.1,(now-last)/1000);last=now;
 if(playing&&t<N){t=Math.min(N,t+dt*speed/.9);if(t>=N)playing=false}
 document.getElementById('play').textContent=playing?'Pause':(t>=N?'Replay':'Play');
 draw();requestAnimationFrame(animate)}
document.getElementById('play').onclick=()=>{if(t>=N)t=0;playing=!playing};
document.getElementById('reset').onclick=()=>{t=0;playing=false};
document.getElementById('scrub').oninput=e=>{t=Number(e.target.value)/1000*N;playing=false};
document.getElementById('speeds').onclick=e=>{const value=e.target.dataset.speed;if(!value)return;
 speed=Number(value);for(const b of e.currentTarget.children)b.classList.toggle('on',b.dataset.speed===value)};
new ResizeObserver(resize).observe(canvas.parentElement);resize();requestAnimationFrame(animate);
</script></body></html>"""


def render_smooth_field(res, c_req: float, height: int = 480) -> None:
    """Render the live dashboard and continuous coverage animation on one timeline."""
    payload = field_animation_payload(res, c_req, cards=False, advice=True)
    inputs = res.meta.get("inputs", {})
    payload["setup"] = {
        "weather": str(inputs.get("Weather scenario", "—")),
        "speed": str(inputs.get("Target travel speed (mph)", "—")),
        "boom": str(round(float(inputs.get("Starting boom height (m)", 0.5)) / 0.0254, 1)),
        "nozzle": str(inputs.get("Nozzle", "—")),
        "gpa": str(inputs.get("Target spray rate (gal/acre)", "—")),
        "pressure": str(inputs.get("Target pressure (psi)", "—")),
    }
    data = json.dumps(payload).replace("</", "<\\/")
    components.html(_HTML.replace("__PAYLOAD__", data), height=int(height), scrolling=False)
