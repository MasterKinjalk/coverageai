"""Animated "Field run" view: publication-style figures that play over time with a moving
cursor. Self-contained canvas component (no Streamlit reruns, no network dependencies)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coverageai.field_anim import field_animation_payload  # noqa: E402

_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#0e1117;color:#c9d1d9;font:13px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}
#app{display:flex;flex-direction:column;height:100%;gap:6px}
#top{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:#8a94a6}#top b{color:#e6edf3;font-weight:600}
#stage{flex:1;min-height:0;position:relative;background:#161b22;border:1px solid #2b3340;border-radius:10px;overflow:hidden}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
#ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{background:#1f2630;color:#e6edf3;border:1px solid #2b3340;border-radius:6px;padding:4px 12px;font:inherit;cursor:pointer}
button:hover{background:#2b3340}button.on{background:#2f6fed;border-color:#2f6fed}
#scrub{flex:1;min-width:140px;accent-color:#4da3ff}
#read{font-size:12px;color:#8a94a6;min-height:34px;background:#161b22;border:1px solid #2b3340;border-radius:8px;padding:4px 10px}
#read b{color:#e6edf3;font-weight:600}
#tip{position:absolute;display:none;pointer-events:none;background:#0e1117ee;border:1px solid #3a4454;border-radius:8px;padding:6px 9px;font-size:11.5px;line-height:1.4;white-space:nowrap;z-index:5}
#tip b{color:#e6edf3}#tip .m{color:#8a94a6}
#sum{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(14,17,23,.72);z-index:6}
#sum>div{background:#161b22;border:1px solid #2b3340;border-radius:12px;padding:16px 24px;min-width:300px}
#sum h2{margin:0 0 8px;font-size:16px;color:#e6edf3}
#sum table{border-collapse:collapse;width:100%;margin-bottom:12px}#sum td{padding:3px 0;border-bottom:1px solid #2b3340}#sum td:last-child{text-align:right;color:#e6edf3;font-weight:600}
</style></head><body><div id="app">
<div id="top"></div>
<div id="stage"><canvas id="cv"></canvas><div id="tip"></div><div id="sum"></div></div>
<div id="ctl"><button id="play">Pause</button><button id="restart">Restart</button><span id="spd"></span>
<input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Position"><span id="pos" style="color:#8a94a6;min-width:90px;text-align:right"></span></div>
<div id="read"></div>
</div>
<script>
const D=__PAYLOAD__;
const S=D.sections,N=S.length,$=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d');
const TXT='#c9d1d9',MUT='#8a94a6',GRID='#2b3340',PLOT='#10151c';
const CP='#6ea8fe',CC='#c58bff',CM='#ffb84d',CT='#e6edf3',CB='#4dd0c8',CW='#4da3ff',CD='#ef9a5b';
let W=0,H=0,dpr=1,t=0,playing=true,speed=1,last=null,hover=-1,scrubbing=false;
const BASE=0.9;
const FONT='system-ui,-apple-system,Segoe UI,sans-serif';
$('top').innerHTML='<span>Strategy: <b>'+D.strategy+'</b></span><span>Target coverage: <b>'+D.target+'%</b></span><span>Hidden surprises: <b>'+D.surprises+'</b></span><span>Sections: <b>'+N+'</b></span>';
$('spd').innerHTML=[0.5,1,2,4].map(s=>'<button data-s="'+s+'" class="'+(s===1?'on':'')+'">'+s+'x</button>').join(' ');
$('spd').onclick=e=>{const s=e.target.dataset&&e.target.dataset.s;if(!s)return;speed=+s;[...$('spd').children].forEach(b=>b.classList.toggle('on',b.dataset.s===s))};
function sync(){$('play').textContent=(t>=N)?'Replay':(playing?'Pause':'Play')}
$('play').onclick=()=>{if(t>=N){t=0;playing=true}else playing=!playing;sync()};
$('restart').onclick=()=>{t=0;playing=true;sync()};
$('scrub').oninput=e=>{t=e.target.value/1000*N;sync()};
$('scrub').addEventListener('pointerdown',()=>scrubbing=true);
window.addEventListener('pointerup',()=>scrubbing=false);

// ---------- helpers
function axisRange(lo,hi,n){ // outer-rounded range + ticks
 if(!(hi>lo)){hi=lo+1}
 const raw=(hi-lo)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/mag;
 const step=(f<1.5?1:f<3?2:f<7?5:10)*mag;
 const a=Math.floor(lo/step+1e-9)*step,b=Math.ceil(hi/step-1e-9)*step,ticks=[];
 for(let v=a;v<=b+step*1e-6;v+=step)ticks.push(Math.abs(v)<1e-9?0:+v.toFixed(6));
 return{min:a,max:b,ticks:ticks,step:step}}
function fmt(v,step){return step>=1?String(Math.round(v)):v.toFixed(step>=.1?1:2)}
function hex(c,a){const n=parseInt(c.slice(1),16);return 'rgba('+(n>>16)+','+((n>>8)&255)+','+(n&255)+','+a+')'}
const VIR=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[253,231,37]];
function viridis(u){u=Math.max(0,Math.min(1,u))*8;const i=Math.min(7,Math.floor(u)),f=u-i,a=VIR[i],b=VIR[i+1];
 return[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f]}
function rgb(c,a){return 'rgba('+Math.round(c[0])+','+Math.round(c[1])+','+Math.round(c[2])+','+(a===undefined?1:a)+')'}
function lum(c){return (0.299*c[0]+0.587*c[1]+0.114*c[2])/255}
function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath()}
function text(s,x,y,o){o=o||{};ctx.font=(o.w||400)+' '+(o.sz||11)+'px '+FONT;ctx.fillStyle=o.c||TXT;ctx.textAlign=o.al||'left';ctx.textBaseline=o.bl||'alphabetic';ctx.fillText(s,x,y)}
function tw(s,sz,w){ctx.font=(w||400)+' '+sz+'px '+FONT;return ctx.measureText(s).width}

// ---------- data ranges
const vmax=axisRange(0,Math.max(D.target*1.5,...S.map(s=>s.actual)),5);
const RB=axisRange(0,Math.max(D.bands.good_min,D.target,...S.map(s=>Math.max(s.predicted,s.after_feedback,s.card,s.actual)))*1.08,5);
const RCl=axisRange(0,Math.max(...S.map(s=>Math.max(s.wind_speed_ms,s.crosswind_ms)))*1.08,4);
const vm=S.map(s=>s.vmd_um),RCr=axisRange(Math.min(...vm)*0.92,Math.max(...vm)*1.06,4);
const RDr=axisRange(0,Math.max(0.5,...S.map(s=>s.drift_pct))*1.08,3);
const lim=Math.max(1,...S.map(s=>Math.max(Math.abs(s.feedback_pp),Math.abs(s.error_pp))))*1.08,RDf=axisRange(-lim,lim,4);
const maxWind=Math.max(...S.map(s=>s.wind_speed_ms));

// ---------- layout
let LA=null,PL=[];
function layout(){
 const r=cv.parentElement.getBoundingClientRect();dpr=window.devicePixelRatio||1;W=r.width;H=r.height;
 cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);
 const m=10,wide=W>=760;
 let A,R;
 if(wide){const lw=Math.max(300,Math.min(W*0.4,520));A={x:m,y:m,w:lw-m,h:H-2*m};R={x:lw+m,y:m,w:W-lw-2*m,h:H-2*m}}
 else{const ah=Math.min(H*0.4,360);A={x:m,y:m,w:W-2*m,h:ah};R={x:m,y:ah+2*m,w:W-2*m,h:H-ah-3*m}}
 // heatmap panel
 const lapW=54,top=26,cbH=62,pad=6;
 const gx=A.x+lapW,gw=A.w-lapW-pad,gh=A.h-top-cbH-18;
 const cols=D.cols,rows=D.rows,g=4;
 let cw=(gw-(cols-1)*g)/cols,ch=(gh-(rows-1)*g)/rows;
 ch=Math.min(ch,cw*0.95,92);cw=Math.min(cw,ch*1.6+20);
 const gridW=cols*cw+(cols-1)*g,gridH=rows*ch+(rows-1)*g;
 LA={A:A,gx:gx,gy:A.y+top,cw:cw,ch:ch,g:g,gridW:gridW,gridH:gridH,cbY:A.y+top+gridH+38,cbH:11};
 // right-hand stacked panels (shared x)
 const fr=[.31,.27,.15,.27],ids=['B','C','E','D'];PL=[];let y=R.y;
 const hs=fr.map(f=>(R.h-3*8)*f);
 ids.forEach((id,i)=>{
  const box={x:R.x,y:y,w:R.w,h:hs[i]},last=i===ids.length-1;
  const p={id:id,x:box.x+50,y:box.y+22,w:box.w-50-58,h:box.h-22-(last?36:15),box:box,last:last};
  p.X=v=>p.x+(v-0.5)/N*p.w;PL.push(p);y+=hs[i]+8});
}
function setY(p,r){p.r=r;p.Y=v=>p.y+p.h-(v-r.min)/(r.max-r.min)*p.h}

// ---------- panel frame
function frame(p,title,legend,ylabel,r,col){
 setY(p,r);
 ctx.fillStyle=PLOT;ctx.fillRect(p.x,p.y,p.w,p.h);
 ctx.strokeStyle=GRID;ctx.lineWidth=1;
 r.ticks.forEach(v=>{const y=Math.round(p.Y(v))+.5;ctx.beginPath();ctx.moveTo(p.x,y);ctx.lineTo(p.x+p.w,y);ctx.strokeStyle='rgba(43,51,64,.7)';ctx.stroke();
  text(fmt(v,r.step),p.x-5,y+3.5,{al:'right',sz:10.5,c:col||MUT})});
 ctx.strokeStyle='#3a4454';ctx.strokeRect(p.x+.5,p.y+.5,p.w-1,p.h-1);
 const step=Math.max(1,Math.ceil(N/10/(p.w>500?1:1.6)));
 for(let k=1;k<=N;k++){if(k===1||k===N||(k%step===0&&N-k>=step*0.6)){const x=Math.round(p.X(k))+.5;ctx.beginPath();ctx.moveTo(x,p.y+p.h);ctx.lineTo(x,p.y+p.h+3);ctx.strokeStyle='#5b6575';ctx.stroke();
  text(String(k),x,p.y+p.h+13,{al:'center',sz:10.5,c:MUT})}}
 if(p.last)text('Field section (driving order)',p.x+p.w/2,p.y+p.h+31,{al:'center',sz:11.5,c:TXT});
 text(title,p.box.x+2,p.box.y+13,{sz:12.5,w:600,c:CT});
 ctx.save();ctx.translate(p.box.x+11,p.y+p.h/2);ctx.rotate(-Math.PI/2);text(ylabel,0,0,{al:'center',sz:10.5,c:col||MUT});ctx.restore();
 // legend, right aligned in title row
 let lx=p.box.x+p.box.w-2;
 for(let i=legend.length-1;i>=0;i--){const L=legend[i],w=tw(L.n,10.5);lx-=w;text(L.n,lx,p.box.y+13,{sz:10.5,c:TXT});lx-=22;
  ctx.strokeStyle=L.c;ctx.fillStyle=L.c;ctx.lineWidth=L.bar?5:2;ctx.setLineDash(L.d||[]);ctx.beginPath();ctx.moveTo(lx+2,p.box.y+9.5);ctx.lineTo(lx+16,p.box.y+9.5);ctx.stroke();ctx.setLineDash([]);
  if(L.m){ctx.beginPath();ctx.arc(lx+9,p.box.y+9.5,2.6,0,6.283);ctx.fill()}
  lx-=12}
}
function cur(){return Math.min(Math.floor(t),N-1)}
function series(p,key,color,o){
 o=o||{};const c=cur();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=o.lw||1.8;ctx.setLineDash(o.dash||[]);
 const Y=o.Y||p.Y;ctx.beginPath();
 for(let k=0;k<=c;k++){const x=p.X(k+1),y=Y(S[k][key]);if(k===0)ctx.moveTo(x,y);else if(o.step){ctx.lineTo(x,Y(S[k-1][key]));ctx.lineTo(x,y)}else ctx.lineTo(x,y)}
 ctx.stroke();ctx.setLineDash([]);
 if(o.mk)for(let k=0;k<=c;k++){ctx.beginPath();ctx.arc(p.X(k+1),Y(S[k][key]),o.mk,0,6.283);ctx.fill()}
}
function cursors(p){
 const cx=p.X(Math.min(t,N-1)+1);
 ctx.save();ctx.beginPath();ctx.rect(p.x,p.y,p.w,p.h);ctx.clip();
 ctx.strokeStyle=CW;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(cx,p.y);ctx.lineTo(cx,p.y+p.h);ctx.stroke();
 if(hover>=0){const hx=p.X(hover+1);ctx.strokeStyle='rgba(230,237,243,.55)';ctx.setLineDash([3,3]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(hx,p.y);ctx.lineTo(hx,p.y+p.h);ctx.stroke();ctx.setLineDash([])}
 ctx.restore()}

// ---------- panels
function drawB(p){
 frame(p,'B. Coverage by field section',[{n:'Model prediction',c:CP,d:[5,3]},{n:'After feedback',c:CC},{n:'Card reading',c:CM,m:1},{n:'Actual',c:CT,m:1}],'Coverage (% of card area)',RB);
 const bd=D.bands,b0=p.Y(0),b1=p.Y(bd.border_min),b2=p.Y(bd.good_min),b3=p.Y(RB.max);
 const cl=D.legend;
 ctx.fillStyle=hex(cl[2].color,.10);ctx.fillRect(p.x,b1,p.w,b0-b1);
 ctx.fillStyle=hex(cl[1].color,.10);ctx.fillRect(p.x,b2,p.w,b1-b2);
 ctx.fillStyle=hex(cl[0].color,.10);ctx.fillRect(p.x,b3,p.w,b2-b3);
 [['Poor',cl[2].color,(b0+b1)/2],['Borderline',cl[1].color,(b1+b2)/2],['Good',cl[0].color,(b2+b3)/2]].forEach(a=>{if(Math.abs(a[2]-0)>0)text(a[0],p.x+p.w-4,a[2]+3,{al:'right',sz:10,c:hex(a[1],.85)})});
 const ty=p.Y(D.target);ctx.strokeStyle='#e6edf3';ctx.setLineDash([2,3]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p.x,ty);ctx.lineTo(p.x+p.w,ty);ctx.stroke();ctx.setLineDash([]);
 text('Target '+D.target+'%',p.x+4,ty-3,{sz:10,c:CT});
 series(p,'predicted',CP,{dash:[5,3],lw:1.5});series(p,'after_feedback',CC);series(p,'card',CM,{mk:2.4,lw:1.2});series(p,'actual',CT,{mk:2.4,lw:2});
 cursors(p)}
function drawC(p){
 frame(p,'C. Controller response to weather',[{n:'Wind speed',c:CW},{n:'Crosswind',c:CB,d:[5,3]},{n:'Droplet size (right axis)',c:CM,m:1}],'Wind (m/s)',RCl);
 const Yr=v=>p.y+p.h-(v-RCr.min)/(RCr.max-RCr.min)*p.h;
 RCr.ticks.forEach(v=>text(fmt(v,RCr.step),p.x+p.w+5,Yr(v)+3.5,{sz:10.5,c:CM}));
 ctx.save();ctx.translate(p.box.x+p.box.w-7,p.y+p.h/2);ctx.rotate(Math.PI/2);text('Median droplet size VMD (µm)',0,0,{al:'center',sz:10.5,c:CM});ctx.restore();
 series(p,'wind_speed_ms',CW);series(p,'crosswind_ms',CB,{dash:[5,3],lw:1.5});series(p,'vmd_um',CM,{Y:Yr,step:true,mk:2.2,lw:1.5});
 cursors(p)}
function drawE(p){
 frame(p,'C2. Drift loss',[{n:'Drift (% of sprayed volume)',c:CD,bar:1}],'Drift (%)',RDr);
 const bw=Math.max(2,p.w/N*0.62);ctx.fillStyle=CD;
 for(let k=0;k<=cur();k++){const y=p.Y(S[k].drift_pct);ctx.fillRect(p.X(k+1)-bw/2,y,bw,p.Y(0)-y)}
 cursors(p)}
function drawD(p){
 frame(p,'D. Feedback loop',[{n:'Feedback correction',c:CB},{n:'Prediction error (actual − after feedback)',c:CT,m:1}],'Percentage points (pp)',RDf);
 const z=p.Y(0);ctx.strokeStyle='#8a94a6';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p.x,z);ctx.lineTo(p.x+p.w,z);ctx.stroke();
 ctx.strokeStyle='rgba(230,237,243,.55)';ctx.fillStyle=CT;ctx.lineWidth=1.2;
 for(let k=0;k<=cur();k++){const x=p.X(k+1),y=p.Y(S[k].error_pp);ctx.beginPath();ctx.moveTo(x,z);ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.arc(x,y,2.4,0,6.283);ctx.fill()}
 series(p,'feedback_pp',CB,{step:true,lw:1.9});
 cursors(p)}

// ---------- heatmap
function tileXY(k){const s=S[k];return{x:LA.gx+s.col*(LA.cw+LA.g),y:LA.gy+s.row*(LA.ch+LA.g)}}
function drawA(){
 const A=LA.A,cw=LA.cw,ch=LA.ch,c0=cur();
 text('A. Field coverage map (actual coverage per section)',A.x+2,A.y+13,{sz:12.5,w:600,c:CT});
 // lap labels
 for(let r=0;r<D.rows;r++){const y=LA.gy+r*(ch+LA.g)+ch/2;const dir=(S.find(s=>s.row===r)||{dir:1}).dir;
  text('Lap '+(r+1),A.x+2,y-1,{sz:10.5,c:MUT});text(dir>0?'drive →':'← drive',A.x+2,y+11,{sz:10.5,c:CW})}
 for(let k=0;k<N;k++){
  const s=S[k],c=tileXY(k),reached=k<=c0&&(t>0||k===0)&&t>=k;
  ctx.fillStyle=PLOT;rr(c.x,c.y,cw,ch,4);ctx.fill();
  let col=null;
  if(reached){col=viridis(s.actual/vmax.max);ctx.fillStyle=rgb(col);rr(c.x,c.y,cw,ch,4);ctx.fill()}
  else{ctx.fillStyle='#232b36';rr(c.x,c.y,cw,ch,4);ctx.fill()}
  const dark=col?lum(col)>0.5:false,fg=col?(dark?'rgba(10,12,16,.92)':'rgba(255,255,255,.95)'):MUT;
  if(cw>=22)text(String(s.number),c.x+4,c.y+11,{sz:Math.min(10.5,cw/3.2),c:col?(dark?'rgba(10,12,16,.6)':'rgba(255,255,255,.65)'):'#5b6575'});
  if(reached){
   const big=ch>=34&&cw>=34;
   text(s.actual.toFixed(0),c.x+cw/2,c.y+ch*(big?.46:.58),{al:'center',sz:Math.max(10,Math.min(18,cw/2.6,ch/2.6)),w:700,c:fg});
   if(big){const L=Math.min(cw*.7,30)*(0.35+0.65*s.wind_speed_ms/Math.max(maxWind,1)),a=s.wind_screen_deg*Math.PI/180,mx=c.x+cw/2,my=c.y+ch*.82;
    ctx.save();ctx.translate(mx,my);ctx.rotate(a);ctx.strokeStyle=fg;ctx.fillStyle=fg;ctx.globalAlpha=.8;ctx.lineWidth=1.3;
    ctx.beginPath();ctx.moveTo(-L/2,0);ctx.lineTo(L/2,0);ctx.stroke();ctx.beginPath();ctx.moveTo(L/2+3,0);ctx.lineTo(L/2-3,-2.6);ctx.lineTo(L/2-3,2.6);ctx.closePath();ctx.fill();ctx.restore()}
   if(s.respray){ctx.strokeStyle='#ff5a5a';ctx.lineWidth=2.2;const m=Math.min(cw,ch)*.11,x=c.x+cw-m-4,y=c.y+m+4;ctx.beginPath();ctx.moveTo(x-m,y-m);ctx.lineTo(x+m,y+m);ctx.moveTo(x+m,y-m);ctx.lineTo(x-m,y+m);ctx.stroke()}
  }
  const cur_=(k===c0&&t<N),hov=k===hover;
  ctx.lineWidth=cur_||hov?2.2:1;ctx.strokeStyle=cur_?CW:hov?'#e6edf3':'rgba(58,68,84,.9)';rr(c.x,c.y,cw,ch,4);ctx.stroke();
 }
 text('Position along lap (driving order)',LA.gx+LA.gridW/2,LA.gy+LA.gridH+14,{al:'center',sz:10.5,c:MUT});
 // colorbar
 const bx=LA.gx,bw=Math.max(160,LA.gridW),by=LA.cbY,bh=LA.cbH;
 const gr=ctx.createLinearGradient(bx,0,bx+bw,0);for(let i=0;i<=16;i++)gr.addColorStop(i/16,rgb(viridis(i/16*(vmax.max/vmax.max))));
 ctx.fillStyle=gr;ctx.fillRect(bx,by,bw,bh);ctx.strokeStyle='#3a4454';ctx.lineWidth=1;ctx.strokeRect(bx+.5,by+.5,bw-1,bh-1);
 vmax.ticks.forEach(v=>{const x=bx+v/vmax.max*bw;ctx.strokeStyle='#8a94a6';ctx.beginPath();ctx.moveTo(x,by+bh);ctx.lineTo(x,by+bh+3);ctx.stroke();text(fmt(v,vmax.step),x,by+bh+14,{al:'center',sz:10.5,c:MUT})});
 const tx=bx+D.target/vmax.max*bw;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(tx,by-4);ctx.lineTo(tx,by+bh);ctx.stroke();
 ctx.fillStyle='#fff';ctx.beginPath();ctx.moveTo(tx,by-1);ctx.lineTo(tx-4,by-7);ctx.lineTo(tx+4,by-7);ctx.closePath();ctx.fill();
 text('target '+D.target+'%',tx,by-10,{al:'center',sz:10.5,c:CT});
 text('Actual coverage (% of card area)',bx+bw/2,by+bh+28,{al:'center',sz:11.5,c:TXT});
 // key
 const ky=by+bh+46;ctx.strokeStyle=CW;ctx.fillStyle=CW;ctx.lineWidth=1.3;
 const L5=30*(5/Math.max(maxWind,1))*.65+10;
 ctx.beginPath();ctx.moveTo(bx,ky-3);ctx.lineTo(bx+L5,ky-3);ctx.stroke();ctx.beginPath();ctx.moveTo(bx+L5+3,ky-3);ctx.lineTo(bx+L5-3,ky-6);ctx.lineTo(bx+L5-3,ky);ctx.closePath();ctx.fill();
 text('wind direction (length ∝ speed, max '+maxWind.toFixed(1)+' m/s)',bx+L5+10,ky,{sz:10,c:MUT});
 text('✖',bx,ky+14,{sz:11,c:'#ff5a5a'});text('needs re-spraying',bx+14,ky+14,{sz:10,c:MUT});
 text('Outlined = current section',bx+bw/2,ky+14,{sz:10,c:MUT});
}

function draw(){
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 drawA();drawB(PL[0]);drawC(PL[1]);drawE(PL[2]);drawD(PL[3]);
}

// ---------- readout / tooltip / summary
function pct(v){return v.toFixed(1)+'%'}
function rowsFor(k){
 const s=S[k],reached=k<=cur()&&t>=k;
 let h='<b>Section '+s.number+'</b> <span class="m">(lap '+(s.row+1)+', driving '+(s.dir>0?'right':'left')+')</span><br>';
 h+='<span class="m">Wind</span> '+s.wind_speed_ms.toFixed(1)+' m/s (cross '+s.crosswind_ms.toFixed(1)+') &middot; '+s.temp_c.toFixed(0)+' &deg;C &middot; '+s.rh_pct.toFixed(0)+'% RH<br>';
 h+='<span class="m">Nozzle</span> '+s.nozzle+' &middot; '+s.droplet+' ('+s.vmd_um.toFixed(0)+' &micro;m) &middot; '+s.rate_gpa.toFixed(1)+' gal/acre<br>';
 if(!reached)return h+'<span class="m">Not sprayed yet.</span>';
 h+='<span class="m">Coverage</span> predicted '+pct(s.predicted)+' &rarr; after feedback '+pct(s.after_feedback)+' &rarr; card '+pct(s.card)+' &rarr; actual <b>'+pct(s.actual)+'</b> ('+s.label+')<br>';
 h+='<span class="m">Drift</span> '+s.drift_pct.toFixed(1)+'% &middot; <span class="m">feedback</span> '+(s.feedback_pp>=0?'+':'')+s.feedback_pp.toFixed(1)+' pp &middot; <span class="m">error</span> '+(s.error_pp>=0?'+':'')+s.error_pp.toFixed(1)+' pp';
 if(s.respray)h+=' &middot; <b style="color:#ff5a5a">&#10006; needs re-spraying</b>';
 if(s.no_nozzle)h+=' &middot; <b style="color:#f5b942">no nozzle meets target</b>';
 return h}
function readout(){
 const k=hover>=0?hover:cur();const h=rowsFor(k);const el=$('read');
 if(el._h!==h){el.innerHTML=h.replace('<br>',' &nbsp;|&nbsp; ').replace(/<br>/g,' &nbsp;|&nbsp; ');el._h=h}
 $('pos').textContent='Section '+(cur()+1)+' / '+N;
 if(!scrubbing)$('scrub').value=Math.round(Math.min(t,N)/N*1000);
}
let sumShown=false,sumClosed=false;
function summary(){
 if(t<N)sumClosed=false;
 const show=t>=N&&!sumClosed;if(show===sumShown)return;sumShown=show;const el=$('sum');el.style.display=show?'flex':'none';if(!show)return;
 const a=S.map(s=>s.actual),m=a.reduce((x,y)=>x+y,0)/N,sd=N>1?Math.sqrt(a.reduce((x,y)=>x+(y-m)*(y-m),0)/(N-1)):0;
 const on=S.filter(s=>s.on_target).length,dr=S.reduce((x,s)=>x+s.drift_pct,0)/N,rs=S.filter(s=>s.respray).length;
 el.innerHTML='<div><h2>Pass complete</h2><table><tr><td>Sections on target</td><td>'+on+' / '+N+'</td></tr><tr><td>Coverage, mean &plusmn; sd</td><td>'+m.toFixed(1)+' &plusmn; '+sd.toFixed(1)+' %</td></tr><tr><td>Mean drift</td><td>'+dr.toFixed(1)+' %</td></tr><tr><td>Sections to re-spray</td><td>'+rs+'</td></tr></table><button id="rp" class="on">Replay</button> <button id="rv">Review run</button></div>';
 $('rp').onclick=()=>{t=0;playing=true;sync()};
 $('rv').onclick=()=>{sumClosed=true}}

function frameLoop(ts){
 if(last===null)last=ts;const dt=Math.min(1,(ts-last)/1000);last=ts;
 if(playing&&t<N){t=Math.min(N,t+dt*speed/BASE);if(t>=N)playing=false;sync()}
 draw();readout();summary();requestAnimationFrame(frameLoop)}

// ---------- interaction
function pick(e){
 const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
 for(let k=0;k<N;k++){const c=tileXY(k);if(x>=c.x&&x<=c.x+LA.cw&&y>=c.y&&y<=c.y+LA.ch)return k}
 for(const p of PL){if(x>=p.x&&x<=p.x+p.w&&y>=p.box.y&&y<=p.y+p.h+20){return Math.max(0,Math.min(N-1,Math.round((x-p.x)/p.w*N+0.5)-1))}}
 return -1}
cv.addEventListener('mousemove',e=>{hover=pick(e);cv.style.cursor=hover>=0?'pointer':'default';
 const tip=$('tip');if(hover<0){tip.style.display='none';return}
 tip.innerHTML=rowsFor(hover);tip.style.display='block';
 const r=cv.getBoundingClientRect();let x=e.clientX-r.left+14,y=e.clientY-r.top+14;
 if(x+tip.offsetWidth>W-4)x=e.clientX-r.left-tip.offsetWidth-14;if(y+tip.offsetHeight>H-4)y=H-tip.offsetHeight-4;
 tip.style.left=Math.max(4,x)+'px';tip.style.top=Math.max(4,y)+'px'});
cv.addEventListener('mouseleave',()=>{hover=-1;$('tip').style.display='none'});
cv.addEventListener('click',e=>{const k=pick(e);if(k>=0){t=k;playing=false;sync()}});
new ResizeObserver(layout).observe(cv.parentElement);layout();
requestAnimationFrame(frameLoop);
</script></body></html>"""


def embed_html(html: str, height: int) -> None:
    """Show a self-contained HTML page in an iframe. ``components.html`` is deprecated (Streamlit
    announced its removal after 2026-06-01), so use ``st.iframe`` when this Streamlit has it and
    fall back to ``components.html`` for older versions (pyproject allows streamlit>=1.38)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=int(height))
    else:
        components.html(html, height=int(height), scrolling=False)


def render_field_animation(res, c_req: float, height: int = 900) -> None:
    """Render the animated field run (runs entirely in the browser, no Streamlit reruns)."""
    payload = field_animation_payload(res, c_req)
    data = json.dumps(payload).replace("</", "<\\/")
    embed_html(_HTML.replace("__PAYLOAD__", data), height)
