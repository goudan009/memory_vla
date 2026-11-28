// Run production UI script against deterministic fake DOM and API. No real jobs.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../evaluation/scripts/resource_downloads.js'),'utf8');
const elements={};
for(const name of ['panel','choice','start','refresh','preview','message','detail','progress','summary','raw','register']){
  elements['resource-download-'+name]={value:'',disabled:false,hidden:false,textContent:'',open:true,events:{},
    addEventListener(k,f){this.events[k]=f;},getClientRects(){return [1];},
    replaceChildren(...options){this.options=options;this.value=options[0]?.value||'';}};
}
const get=name=>elements['resource-download-'+name];
const document={getElementById:id=>elements[id],hidden:false,createElement:tag=>({tag,textContent:'',children:[],append(...nodes){this.children.push(...nodes);}})};
let registered=null,poll;
const window={registryStorage:{registerDownload:recipe=>{registered=recipe;}}};
function Option(label,value){this.label=label;this.value=value;}
let state='not_started', delay=null, failStart=false;
const trace=[];
async function fetch(url,options){
  const p=options?.method==='POST'?JSON.parse(options.body):Object.fromEntries(new URLSearchParams(url.split('?')[1]));
  trace.push(p);
  if(delay&&p.action===delay.action){const d=delay;delay=null;await d.promise;}
  const result=p.action==='options'?{recipes:[{id:'a',host:'A100',repo:'org/a',target:'/data/a',kind:'dataset'},{id:'b',host:'4090',repo:'org/b',target:'/data/b',kind:'weights'}]}:
    p.action==='status'?{status:state,job_id:p.recipe_id+'-hash',total_bytes:10,verified_bytes:state==='complete'?10:0}:
    p.action==='preview'?{job_id:p.recipe_id+'-hash',total_bytes:10}:
    failStart?{error:'timeout'}:{status:'starting',job_id:p.expected_job_id,total_bytes:10};
  return {ok:!result.error,json:async()=>result};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  new Function('document','Option','fetch','confirm','setInterval','URLSearchParams','window',code)(document,Option,fetch,()=>true,callback=>{poll=callback;},URLSearchParams,window);
  get('panel').events.toggle();await flush();
  assert.equal(get('choice').value,'a');assert.equal(get('preview').disabled,false);assert.equal(get('start').disabled,true);
  assert.equal(get('summary').hidden,false);assert.equal(get('summary').options[0].children[1].textContent,'A100');assert.equal(get('raw').open,false);assert.equal(get('register').hidden,true);
  get('choice').value='b';get('choice').onchange();await flush();assert.equal(trace.at(-1).recipe_id,'b');
  let resolve;delay={action:'preview',promise:new Promise(r=>resolve=r)};
  const pending=get('preview').onclick();assert.ok(get('choice').disabled&&get('refresh').disabled&&get('start').disabled);
  resolve();await pending;assert.equal(get('start').disabled,false);
  await get('start').onclick();assert.equal(trace.at(-1).expected_job_id,'b-hash');
  assert.ok(get('start').disabled&&get('preview').disabled);
  get('raw').open=true;state='complete';await poll();assert.equal(get('progress').value,1);assert.ok(get('preview').disabled);assert.equal(get('raw').open,true);assert.equal(get('register').hidden,false);
  get('register').onclick();assert.equal(registered.target,'/data/b');assert.equal(registered.host,'4090');assert.equal(registered.kind,'weights');
  state='failed';get('refresh').onclick();await flush();assert.equal(get('preview').disabled,false);
  await get('preview').onclick();failStart=true;await get('start').onclick();
  assert.ok(get('preview').disabled&&get('message').textContent.includes('查询状态'));
  console.log('Download UI: state, recipe switching, preview guard, binding, completion, retry, uncertain submission, summary cards, stable expanded details and prefilled registration passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
