/* Recorded inputs and saved predictions only; no model is fitted or run here. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const models = ['transformer', 'mlp', 'ridge'];
  const colors = { observed: '#e6edf3', transformer: '#78b7ff', mlp: '#edb576', ridge: '#b6a3ee' };
  const state = { recording: null, seed: 401, head: 0, playing: false, rate: 12,
    yaw: -.57, pitch: .55, zoom: 1, hover: -1, lastFrame: -1, dirty: true };
  const cache = new Map();
  const canvasCache = new Map();
  let projections = [], cloudWidth = 0, cloudHeight = 0, lastTick = 0, pointer = null;
  let source;

  function unpack(text, Type) {
    const raw = atob(text), bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new Type(bytes.buffer);
  }

  function context(id) {
    const canvas = $(id), rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, rect.width), h = Math.max(1, rect.height);
    const pw = Math.round(w * ratio), ph = Math.round(h * ratio);
    const resized = canvas.width !== pw || canvas.height !== ph;
    if (resized) { canvas.width = pw; canvas.height = ph; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    canvasCache.set(id, { w, h });
    return { ctx, w, h, resized };
  }

  function series(name) {
    return state.recording.speed[name === 'observed' || name === 'ridge' ? name : `${name}_${state.seed}`];
  }

  function prefix(name) {
    return state.recording.prefix[name === 'observed' || name === 'ridge' ? name : `${name}_${state.seed}`];
  }

  function metric(name) {
    return state.recording.metrics[name === 'ridge' ? name : `${name}_${state.seed}`];
  }

  function frame() { return Math.min(state.recording.frames - 1, Math.floor(state.head)); }

  function decode(record) {
    if (cache.has(record.id)) return cache.get(record.id);
    const result = { ...record, positions: unpack(record.positions, Float32Array),
      activity: unpack(record.activity, Uint8Array), speed: {}, prefix: {}, points: [] };
    if (result.positions.length !== source.neurons * 3 || result.activity.length !== record.frames * source.neurons) {
      throw new Error('The recorded data do not match the expected frame and cell dimensions.');
    }
    for (const [key, value] of Object.entries(record.speed)) {
      const values = unpack(value, Float32Array);
      if (values.length !== record.frames) throw new Error('Prediction timeline length mismatch.');
      result.speed[key] = values;
      const cumulative = new Float64Array(values.length + 1);
      for (let i = 0; i < values.length; i++) cumulative[i + 1] = cumulative[i] + values[i];
      result.prefix[key] = cumulative;
    }
    const ranges = [0, 1, 2].map(axis => {
      const vals = Array.from({ length: source.neurons }, (_, i) => result.positions[i * 3 + axis]);
      return [Math.min(...vals), Math.max(...vals)];
    });
    const scaleXY = Math.max(ranges[0][1] - ranges[0][0], ranges[1][1] - ranges[1][0]);
    for (let i = 0; i < source.neurons; i++) {
      const [x, y, plane] = result.positions.slice(i * 3, i * 3 + 3);
      result.points.push({
        x: (x - (ranges[0][0] + ranges[0][1]) / 2) / scaleXY * 2.2,
        z: (y - (ranges[1][0] + ranges[1][1]) / 2) / scaleXY * 2.2,
        y: (.5 - (plane - ranges[2][0]) / Math.max(1, ranges[2][1] - ranges[2][0])) * 1.35,
        index: i,
      });
    }
    cache.set(record.id, result);
    return result;
  }

  function changeRecording(id) {
    const record = source.mice.find(r => r.id === id);
    if (!record) throw new Error('Unknown recording.');
    state.recording = decode(record);
    state.head = 0; state.lastFrame = -1; state.hover = -1; state.dirty = true;
    $('scrubber').max = String(record.frames - 1);
    $('last-frame').textContent = `TEST FRAME ${(record.frames - 1).toLocaleString()}`;
    $('neuron-detail').textContent = `512 cells · ${record.plane_count} imaging planes · same input for every model`;
    $('neuron-tooltip').hidden = true;
    updateMetrics(); drawTrace(); render();
  }

  function updateMetrics() {
    for (const name of models) {
      const m = metric(name);
      $(`r2-${name}`).textContent = m.r2.toFixed(3);
      $(`r2-${name}`).classList.toggle('bad', m.r2 < 0);
      $(`zero-${name}`).textContent = `${m.mse_over_zero.toFixed(2)}×`;
      $(`zero-${name}`).classList.toggle('bad', m.mse_over_zero > 1);
    }
  }

  function setPlaying(value) {
    state.playing = value;
    $('play').innerHTML = `${value ? 'Ⅱ' : '▶'} <span>${value ? 'Pause' : 'Play'}</span>`;
    $('play').setAttribute('aria-label', value ? 'Pause timeline' : 'Play timeline');
    $('play').setAttribute('aria-pressed', String(value));
    lastTick = 0;
  }

  function seek(value) {
    state.head = Math.max(0, Math.min(state.recording.frames - 1, value));
    state.lastFrame = -1;
    render();
  }

  function cameraPoint(point) {
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const x = point.x * cy + point.z * sy;
    const z = -point.x * sy + point.z * cy;
    const y = point.y * cp - z * sp;
    const depth = point.y * sp + z * cp;
    const perspective = 5.2 / (5.2 + depth);
    return { x: x * perspective, y: y * perspective, depth, size: perspective, index: point.index };
  }

  function project(point, w, h) {
    const p = cameraPoint(point), fit = state.cameraFit;
    const scale = fit.scale * state.zoom;
    return { ...p, x: w * .51 + (p.x-fit.x) * scale, y: h * .48 - (p.y-fit.y) * scale };
  }

  function drawCloud() {
    const { ctx, w, h, resized } = context('neurons');
    if (resized || cloudWidth !== w || cloudHeight !== h) state.dirty = true;
    cloudWidth = w; cloudHeight = h;
    if (state.dirty) {
      const corners = [];
      for (const x of [-1.15,1.15]) for (const y of [-.72,.72]) for (const z of [-1.15,1.15]) corners.push(cameraPoint({x,y,z}));
      const xs=corners.map(p=>p.x),ys=corners.map(p=>p.y);
      const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
      state.cameraFit={scale:Math.min((w-65)/(xmax-xmin),(h-38)/(ymax-ymin)),x:(xmin+xmax)/2,y:(ymin+ymax)/2};
    }
    ctx.clearRect(0, 0, w, h);
    const glow = ctx.createRadialGradient(w*.51, h*.48, 0, w*.51, h*.48, Math.min(w,h)*.8);
    glow.addColorStop(0, '#23483a35'); glow.addColorStop(1, '#0d172200');
    ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = .7;
    for (const plane of [0, 5, 10, 15, 19]) {
      const y = (.5 - plane / 19) * 1.35;
      const corners = [[-1.1,-1.1],[1.1,-1.1],[1.1,1.1],[-1.1,1.1]].map(([x,z]) => project({x,y,z},w,h));
      ctx.strokeStyle = plane === 0 || plane === 19 ? '#527b7650' : '#486c6928';
      ctx.beginPath(); corners.forEach((p,i) => i ? ctx.lineTo(p.x,p.y) : ctx.moveTo(p.x,p.y)); ctx.closePath(); ctx.stroke();
    }
    if (state.dirty) {
      projections = state.recording.points.map(p => project(p, w, h)).sort((a,b) => b.depth - a.depth);
      state.dirty = false;
    }
    const offset = frame() * source.neurons;
    for (const p of projections) {
      const level = state.recording.activity[offset + p.index] / 255;
      const r = (1.6 + level * 2.8) * p.size;
      if (level > .08) {
        const aura = ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r*3.5);
        aura.addColorStop(0, `rgba(119,241,210,${.08 + level*.22})`);
        aura.addColorStop(1, 'rgba(119,241,210,0)');
        ctx.fillStyle = aura; ctx.fillRect(p.x-r*3.5,p.y-r*3.5,r*7,r*7);
      }
      const c = Math.round(85 + level * 150);
      ctx.fillStyle = level === 0 ? '#41626bb5' : `rgb(${Math.round(68+level*157)},${c},${Math.round(113+level*119)})`;
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill();
      if (p.index === state.hover) {
        ctx.strokeStyle = '#e3fff6'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p.x,p.y,r+5,0,Math.PI*2); ctx.stroke();
      }
    }
    ctx.font = '9px ui-sans-serif, system-ui'; ctx.fillStyle = '#8faeb6';
    for (const [label, position] of [['x', {x:1.27,y:-.69,z:-1.1}], ['y', {x:-1.14,y:-.69,z:1.3}], ['plane 0', {x:-1.19,y:.77,z:-1.15}], ['plane 19', {x:-1.22,y:-.88,z:-1.15}]]) {
      const p=project(position,w,h); ctx.fillText(label,Math.max(8,Math.min(w-60,p.x)),Math.max(12,Math.min(h-12,p.y)));
    }
    updateTooltip();
  }

  function updateTooltip() {
    if (state.hover < 0) { $('neuron-tooltip').hidden = true; return; }
    const i=state.hover, p=state.recording.positions, level=state.recording.activity[frame()*512+i];
    const activity=level===0?'at or below training mean':level===255?'≥ +5 SD':`≈ +${(level/51).toFixed(2)} SD`;
    $('neuron-tooltip').innerHTML = `<strong>Cell ${state.recording.neuron_ids[i]}</strong><br>x ${p[i*3].toFixed(1)} · y ${p[i*3+1].toFixed(1)} · plane ${p[i*3+2]}<br>${activity}`;
    $('neuron-tooltip').hidden=false;
  }

  function ellipse(ctx, x, y, rx, ry, angle, fill) {
    ctx.fillStyle = fill; ctx.beginPath(); ctx.ellipse(x,y,rx,ry,angle,0,2*Math.PI); ctx.fill();
  }

  function drawMouse(name) {
    const { ctx, w, h } = context(`mouse-${name}`);
    const f=frame(), speed=series(name)[f];
    const distance=prefix(name)[f] + speed*(state.head-f);
    const phase=distance*.38, motion=Math.min(1,speed/.16), ink=colors[name];
    ctx.clearRect(0,0,w,h);
    const scale=Math.min(w/330,h/126);
    ctx.save(); ctx.translate(w*.53,h*.58); ctx.scale(scale,scale);
    const ground=32;
    ctx.strokeStyle='#304454'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(-145,ground+2); ctx.lineTo(140,ground+2); ctx.stroke();
    ctx.strokeStyle=ink+'45';
    const drift=(distance*7)%25;
    for(let x=-140-drift;x<138;x+=25){ctx.beginPath();ctx.moveTo(x,ground+8);ctx.lineTo(x+9,ground+8);ctx.stroke();}
    ellipse(ctx,0,ground,57,4,0,'#00000035');
    const bounce=motion*Math.sin(2*phase)*1.4;
    ctx.translate(0,bounce);
    //animate a speed illustration with a shared gait mapping, not a reconstructed pose
    function leg(hip, shift, far) {
      const cycle=phase+shift, stride=Math.sin(cycle)*14*motion;
      const lift=Math.max(0,Math.cos(cycle))*10*motion;
      ctx.strokeStyle=far?'#46545f':ink; ctx.lineWidth=far?3.2:3.8; ctx.lineCap='round';
      ctx.beginPath();ctx.moveTo(hip,-2);ctx.quadraticCurveTo(hip-stride*.5,16,hip+stride,ground-lift-bounce-3);ctx.stroke();
      ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(hip+stride-2,ground-lift-bounce-3);ctx.lineTo(hip+stride+7,ground-lift-bounce-2);ctx.stroke();
    }
    leg(-23,Math.PI,true);leg(24,0,true);
    ctx.strokeStyle='#a7838d';ctx.lineWidth=3;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(-40,1);
    ctx.bezierCurveTo(-65,10,-76,0,-91,-4+Math.sin(phase)*motion*4);
    ctx.bezierCurveTo(-111,-11,-111,-24,-123,-22+Math.cos(phase)*motion*2);ctx.stroke();
    const body=ctx.createLinearGradient(0,-35,0,20);body.addColorStop(0,'#8795a0');body.addColorStop(.5,'#657582');body.addColorStop(1,'#394b59');
    ellipse(ctx,-6,-8,47,25,-.08,body);
    ellipse(ctx,-27,-2,24,24,0,body);
    ellipse(ctx,24,-10,24,19,-.25,body);
    ellipse(ctx,40,-12,19,13,.12,body);
    ctx.fillStyle='#8b98a1';ctx.beginPath();ctx.moveTo(41,-22);ctx.quadraticCurveTo(54,-17,62,-9);ctx.quadraticCurveTo(55,-2,41,0);ctx.closePath();ctx.fill();
    ellipse(ctx,30,-29,10,13,-.3,'#8c9aa3');ellipse(ctx,31,-29,6.8,9,-.3,'#bd939b');
    ellipse(ctx,46,-16,3,3.5,0,'#09131b');ellipse(ctx,46.7,-17,1,1,0,'#ffffff');
    ellipse(ctx,62,-9,3.1,2.6,.15,'#d1a0a8');
    ctx.strokeStyle='#acbcc580';ctx.lineWidth=.7;
    for(const offset of [-5,0,5]){ctx.beginPath();ctx.moveTo(56,-7);ctx.lineTo(79,-8+offset);ctx.stroke();}
    leg(-25,0,false);leg(22,Math.PI,false);
    ctx.strokeStyle=ink+'8c';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(-7,-10,42,22,-.07,Math.PI,Math.PI*1.78);ctx.stroke();
    ctx.restore();
  }

  function drawTrace() {
    if(!state.recording)return;
    const {ctx,w,h}=context('trace');ctx.clearRect(0,0,w,h);
    const top=8,bottom=h-10, ymax=Math.max(.1,state.recording.max_speed)*1.04;
    ctx.strokeStyle='#233847';ctx.lineWidth=.6;
    for(let j=0;j<4;j++){const y=top+(bottom-top)*j/3;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    for(const name of ['observed',...models]){
      const values=series(name);ctx.strokeStyle=colors[name];ctx.lineWidth=name==='observed'?1:.9;ctx.globalAlpha=name==='observed'?.6:.8;ctx.beginPath();
      for(let i=0;i<values.length;i++){
        const x=i/(values.length-1)*w,y=bottom-values[i]/ymax*(bottom-top);
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha=1;ctx.fillStyle='#9db0c0';ctx.font='8px ui-sans-serif, system-ui';ctx.fillText(`${ymax.toFixed(1)} SD`,4,7);
  }

  function render() {
    if(!state.recording)return;
    const f=frame();
    if(f!==state.lastFrame){
      const actual=series('observed')[f];
      $('observed-speed').textContent=actual.toFixed(2);
      $('observed-state').textContent=actual < .005?'At or near zero running':'Recorded behavior';
      for(const name of models){const value=series(name)[f];$(`speed-${name}`).textContent=value.toFixed(2);$(`error-${name}`).textContent=Math.abs(value-actual).toFixed(2);}
      $('frame-counter').textContent=`${state.recording.id} · frame ${f.toLocaleString()} / ${(state.recording.frames-1).toLocaleString()}`;
      $('native-frame').textContent=`NATIVE FRAME ${(state.recording.first_native_frame+f).toLocaleString()}`;
      $('scrubber').value=String(f);
      state.lastFrame=f;
    }
    $('trace-cursor').style.left=`${state.head/(state.recording.frames-1)*100}%`;
    drawCloud();
    for(const name of ['observed',...models])drawMouse(name);
  }

  function animate(time) {
    if(state.recording && state.playing){
      if(lastTick)state.head+=Math.min((time-lastTick)/1000,.15)*state.rate;
      if(state.head>=state.recording.frames-1){state.head=state.recording.frames-1;setPlaying(false);}
      render();
    }
    lastTick=time;requestAnimationFrame(animate);
  }

  function setup() {
    source=window.NEURAL_VIEW_DATA;
    if(!source||source.schema!==1)throw new Error('Could not load data.js. Keep it next to index.html and open the page in a browser.');
    $('recording').replaceChildren(...source.mice.map(record=>{const opt=document.createElement('option');opt.value=record.id;opt.textContent=`${record.id} · ${record.frames.toLocaleString()} test frames`;return opt;}));
    $('recording').addEventListener('change',event=>changeRecording(event.target.value));
    $('seed').addEventListener('change',event=>{state.seed=Number(event.target.value);state.lastFrame=-1;updateMetrics();drawTrace();render();});
    $('play').disabled=false;
    $('play').addEventListener('click',()=>{if(state.head>=state.recording.frames-1)seek(0);setPlaying(!state.playing);});
    $('previous').addEventListener('click',()=>{setPlaying(false);seek(frame()-1);});
    $('next').addEventListener('click',()=>{setPlaying(false);seek(frame()+1);});
    $('rate').addEventListener('change',event=>{state.rate=Number(event.target.value);});
    $('scrubber').addEventListener('input',event=>{setPlaying(false);seek(Number(event.target.value));});
    $('trace').addEventListener('pointerdown',event=>{const r=event.target.getBoundingClientRect();setPlaying(false);seek(Math.round((event.clientX-r.left)/r.width*(state.recording.frames-1)));});
    $('reset-view').addEventListener('click',()=>{state.yaw=-.57;state.pitch=.55;state.zoom=1;state.dirty=true;render();});
    const cloud=$('neurons');
    cloud.addEventListener('pointerdown',event=>{pointer={x:event.clientX,y:event.clientY};cloud.setPointerCapture(event.pointerId);});
    cloud.addEventListener('pointermove',event=>{
      if(pointer){state.yaw+=(event.clientX-pointer.x)*.009;state.pitch=Math.max(-1.3,Math.min(1.3,state.pitch+(event.clientY-pointer.y)*.009));pointer={x:event.clientX,y:event.clientY};state.dirty=true;state.hover=-1;render();return;}
      const r=cloud.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;
      let nearest=-1,dist=12;
      for(const p of projections){const d=Math.hypot(x-p.x,y-p.y);if(d<dist){dist=d;nearest=p.index;}}
      if(nearest!==state.hover){state.hover=nearest;render();}
    });
    cloud.addEventListener('pointerup',()=>{pointer=null;});
    cloud.addEventListener('pointercancel',()=>{pointer=null;});
    cloud.addEventListener('pointerleave',()=>{if(!pointer){state.hover=-1;render();}});
    cloud.addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.55,Math.min(2,state.zoom*Math.exp(-event.deltaY*.001)));state.dirty=true;render();},{passive:false});
    cloud.addEventListener('keydown',event=>{
      if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
        event.preventDefault();state.yaw+=(event.key==='ArrowRight'?.1:event.key==='ArrowLeft'?-.1:0);
        state.pitch=Math.max(-1.3,Math.min(1.3,state.pitch+(event.key==='ArrowDown'?.1:event.key==='ArrowUp'?-.1:0)));state.dirty=true;render();
      }
    });
    new ResizeObserver(()=>{state.dirty=true;drawTrace();render();}).observe(document.querySelector('main'));
    document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);});
    document.addEventListener('keydown',event=>{
      if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','SUMMARY','CANVAS','A'].includes(document.activeElement.tagName)){event.preventDefault();$('play').click();}
    });
    changeRecording(source.mice[0].id);
    setPlaying(!window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    requestAnimationFrame(animate);
    //expose a read-only snapshot for reproducibility checks and inspecting the displayed values
    window.neuralViewer=Object.freeze({snapshot:()=>({recording:state.recording.id,frame:frame(),nativeFrame:state.recording.first_native_frame+frame(),
      seed:state.seed,playing:state.playing,rate:state.rate,camera:{yaw:state.yaw,pitch:state.pitch,zoom:state.zoom},
      neuronCount:state.recording.points.length,activityOffset:frame()*512,
      visibleNeuronCount:projections.filter(p=>p.x>=0&&p.x<=cloudWidth&&p.y>=0&&p.y<=cloudHeight).length,
      speeds:Object.fromEntries(['observed',...models].map(name=>[name,series(name)[frame()]])),
      metrics:Object.fromEntries(models.map(name=>[name,metric(name)]))})});
  }

  try{setup();}catch(error){$('load-error').textContent=error.message;$('load-error').hidden=false;console.error(error);}
})();
