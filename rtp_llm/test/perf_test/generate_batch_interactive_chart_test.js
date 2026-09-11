// Exercise chart controls against recorded data without a browser dependency.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
if(process.argv.length!==4) throw new Error('Usage: node generate_batch_interactive_chart_test.js chart.html interactive_data.json');
const html=fs.readFileSync(process.argv[2],'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const source=scripts[scripts.length-1][1];
const dataset=JSON.parse(fs.readFileSync(process.argv[3],'utf8')).rows;
const elements={};
for(const id of ['view','batch','band','reset','count','plot','trend','detail','axis-legend']){
 elements[id]={value:id==='view'?'mean':'all',textContent:'',children:[],append(o){this.children.push(o)},removeAllListeners(){this.handler=null},on(name,fn){this.handler=fn}};
}
const plots={};
const sandbox={console,document:{getElementById:id=>elements[id],createElement:()=>({})},
 Plotly:{react:(id,traces,layout)=>{plots[id]={traces,layout}}}};
vm.createContext(sandbox);
vm.runInContext(source,sandbox);
assert.equal(plots.plot.traces[0].x.length,dataset.length);
const batches=[...new Set(dataset.map(r=>r.batch_size))];
assert.equal(elements.batch.children.length,batches.length);
for(const view of ['mean','sum','latency']){
 elements.view.value=view;
 for(const batch of batches){
  elements.batch.value=String(batch);
  for(const band of ['all','low/low','low/high','high/low','high/high']){
   elements.band.value=band;sandbox.draw();
   const expected=dataset.filter(r=>r.batch_size===batch&&(band==='all'||r.band===band));
   for(const axis of ['x','y','z']) {
    const title=plots.plot.layout.scene[axis+'axis'].title;
    assert.equal(typeof title,'object');
    assert(title.text.length>0);
    assert(elements['axis-legend'].textContent.includes(title.text));
   }
   assert(plots.trend.layout.xaxis.title.text.includes('请求数'));
   assert(plots.trend.layout.yaxis.title.text.includes('ms'));
   assert(elements['axis-legend'].textContent.includes('1024'));
   const trace=plots.plot.traces[0];
   assert.equal(trace.x.length,expected.length);
   if(expected.length){
    const first=expected[0];
    assert.equal(trace.x[0],view==='latency'?first.mean_compute_tokens/1024:first.batch_size);
    assert.equal(trace.z[0],view==='latency'?first.median_batch_ttft_ms:(view==='sum'?first.compute_tokens:first.mean_compute_tokens)/1024);
    elements.plot.handler({points:[{customdata:first}]});
    assert(elements.detail.textContent.includes(String(first.case_id)));
   }
  }
 }
}
elements.reset.onclick();
assert.equal(plots.plot.traces[0].x.length,dataset.length);
assert.equal(elements.view.value,'mean');
console.log('PASS: axes, units, legend, all view/filter combinations, click details and reset.');
