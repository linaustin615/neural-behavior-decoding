/* Capture saved predictions in a local browser; optional Playwright dependency. */
'use strict';
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const out = process.env.NEURAL_CAPTURE_DIR || '/tmp/neural-presentation-capture';
const preview = process.argv.includes('--preview');

(async () => {
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({channel:'chrome', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1120,height:1200},deviceScaleFactor:1,reducedMotion:'reduce'});
    const errors=[], requests=[];
    page.on('pageerror', e=>errors.push(e.message));
    page.on('request', r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
    await page.goto(pathToFileURL(path.join(__dirname,'index.html')).href);
    await page.waitForFunction(()=>window.neuralViewer);
    assert.equal(await page.locator('#load-error').isVisible(), false);
    const clip = await page.evaluate(()=>{
      const raw=window.NEURAL_VIEW_DATA.mice[0];
      const text=atob(raw.speed.observed), bytes=Uint8Array.from(text,c=>c.charCodeAt(0));
      const y=new Float32Array(bytes.buffer);
      for(let s=0;s+144<=y.length;s++){
        const values=Array.from(y.slice(s,s+144));
        const quiet=values.map(v=>v<=.05),active=values.map(v=>v>=.5);
        let best=0,run=0;for(const a of active){run=a?run+1:0;best=Math.max(best,run);}
        if(quiet.slice(0,12).every(Boolean)&&quiet.slice(-12).every(Boolean)&&quiet.filter(Boolean).length>=36&&active.filter(Boolean).length>=36&&best>=24){
          return {recording:raw.id,seed:401,start:s,end:s+143,quiet:quiet.filter(Boolean).length,active:active.filter(Boolean).length};
        }
      }
      throw new Error('No interval satisfies the declared observed-only selection rule');
    });
    let cases=0,speedChecks=0,metricChecks=0;
    if(!preview){
      for(const mouse of ['D3','D4','D7','D9']){
        await page.selectOption('#recording',mouse);
        for(const seed of ['401','402','403']){
          await page.selectOption('#seed',seed);
          const frames=await page.locator('#scrubber').getAttribute('max');
          for(const f of [0,Math.floor(Number(frames)/2),Number(frames)]){
            await page.evaluate(f=>window.neuralViewer.setFrame(f),f);
            const check=await page.evaluate(()=>{
              const s=window.neuralViewer.snapshot(), record=window.NEURAL_VIEW_DATA.mice.find(r=>r.id===s.recording);
              let speeds=0,metrics=0;
              for(const model of ['observed','blend','corrected','ridge']){
                const key=['observed','ridge'].includes(model)?model:`${model}_${s.seed}`;
                const values=new Float32Array(Uint8Array.from(atob(record.speed[key]),c=>c.charCodeAt(0)).buffer);
                if(s.speeds[model]!==values[s.frame])throw Error('display alignment');
                const id=model==='observed'?'observed-speed':`speed-${model}`;
                if(document.getElementById(id).textContent!==values[s.frame].toFixed(2))throw Error('speed label');
                speeds++;
                if(model!=='observed'){
                  for(const k of ['mse','mae','r2','mse_over_zero']){
                    if(s.metrics[model][k]!==record.metrics[key][k])throw Error('metric alignment');
                    metrics++;
                  }
                }
              }
              if(s.neuronCount!==512||s.visibleNeuronCount!==512||s.nativeFrame!==record.first_native_frame+s.frame)throw Error('frame or spatial alignment');
              return {speeds,metrics};
            });
            cases++;speedChecks+=check.speeds;metricChecks+=check.metrics;
          }
        }
      }
      await page.evaluate(()=>window.neuralViewer.setFrame(0));
      await page.click('#previous');assert.equal((await page.evaluate(()=>window.neuralViewer.snapshot())).frame,0);
      await page.click('#next');assert.equal((await page.evaluate(()=>window.neuralViewer.snapshot())).frame,1);
      await page.click('#play');await page.waitForTimeout(250);await page.click('#play');
      assert.ok((await page.evaluate(()=>window.neuralViewer.snapshot())).frame>1);
      await page.locator('#neurons').focus();await page.keyboard.press('ArrowRight');
      assert.notEqual((await page.evaluate(()=>window.neuralViewer.snapshot())).camera.yaw,-.57);
      await page.click('#reset-view');assert.equal((await page.evaluate(()=>window.neuralViewer.snapshot())).camera.yaw,-.57);
      await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
      await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
      await page.setViewportSize({width:1120,height:1200});
    }
    await page.selectOption('#recording',clip.recording);await page.selectOption('#seed',String(clip.seed));
    await page.evaluate(()=>{
      document.body.classList.add('capture');
      document.querySelector('.timeline-heading p').textContent='Full test interval · cursor follows the illustrated excerpt';
      document.querySelector('.archive-status').textContent='D3 · SEED 401 · SPEED ILLUSTRATION';
    });
    await page.evaluate(f=>window.neuralViewer.setFrame(f),clip.start+72);
    await page.waitForTimeout(100);
    const height=await page.evaluate(()=>Math.ceil(document.querySelector('main').getBoundingClientRect().bottom));
    await page.setViewportSize({width:1120,height});
    await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
    if(preview){console.log(JSON.stringify({clip,height,errors,requests}));return;}
    const fps=50, duration=12, snapshots=[];
    for(let i=0;i<fps*duration;i++){
      const head=clip.start+i*12/fps;
      const s=await page.evaluate(f=>{window.neuralViewer.setFrame(f);return window.neuralViewer.snapshot();},head);
      assert.equal(s.frame,Math.floor(head));assert.equal(s.visibleNeuronCount,512);
      assert.deepEqual(s.camera,{yaw:-.57,pitch:.55,zoom:1});
      snapshots.push(s);
      await page.screenshot({path:path.join(out,`frame-${String(i).padStart(4,'0')}.png`)});
      if(i%100===0)console.log(`Captured ${i}/600 frames`);
    }
    assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
    fs.writeFileSync(path.join(out,'capture.json'),JSON.stringify({clip,fps,duration,width:1120,height,browser:browser.version(),
      runtime_errors:errors,external_requests:requests,cases,speedChecks,metricChecks,mobile_no_horizontal_overflow:true,
      fixed_camera:true,all_cells_visible:true,snapshots},null,2)+'\n');
    console.log('Capture and browser checks complete',out);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
