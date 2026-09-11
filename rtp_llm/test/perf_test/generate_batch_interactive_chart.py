#!/usr/bin/env python3
"""Generate a self-contained batch/cache/compute HTML from interactive_data.json.

Input contains measurement_contract and rows exported by the batch analysis.
No GPU, serving libraries, network/CDN, or experiment-specific paths are required.
"""
import argparse
import json
from pathlib import Path

from plotly.offline import get_plotlyjs

CONTRACT = "batch_input_ids_wall_barrier_to_last_response_ms"


def render_html(rows, partial=False):
    page = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>DSV4 batch / cache / compute / TTFT</title>
<style>
body{font:15px system-ui;margin:0;background:#f4f7fb;color:#183049}main{max-width:1400px;margin:auto;padding:28px}
h1{font-size:30px;margin:0 0 8px}p{line-height:1.6;color:#526477}.card{background:white;border:1px solid #dce5ef;border-radius:14px;padding:18px;margin-top:18px}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,input,button{font:inherit;padding:6px}
#plot{height:670px}#trend{height:450px}.small{font-size:13px}#detail{white-space:pre-wrap;font-size:13px;max-height:260px;overflow:auto}
</style><main><h1>Batch × Cache × Compute → TTFT</h1>
<p>随机混合 batch · input IDs / gRPC · __COUNT__ 个随机混合 batch。__PREVIEW__</p>
<div class="card controls">
<label>视图 <select id="view"><option value="mean">Batch / 平均 cache / 平均 compute；颜色=TTFT</option>
<option value="sum">Batch / 总 cache / 总 compute；颜色=TTFT</option>
<option value="latency">平均 compute / 平均 cache / TTFT；颜色=batch</option></select></label>
<label>Batch <select id="batch"><option value="all">全部</option></select></label>
<label>采样层 <select id="band"><option value="all">全部</option><option>low/low</option><option>low/high</option><option>high/low</option><option>high/high</option></select></label>
<button id="reset">重置</button><span id="count"></span></div>
<div class="card"><p id="axis-legend" aria-live="polite"></p><div id="plot"></div><p class="small">拖动旋转，滚轮缩放，点击点查看批内每条请求；右上工具栏可导出图片。仅绘制实测点，不插值未测量曲面。</p>
<pre id="detail">点击一个点查看请求分布和三轮 TTFT。</pre></div>
<div class="card"><div id="trend"></div></div>
<p>整批 TTFT：barrier 放行至最后一条响应，包含 input IDs 编码、RPC、排队及执行；每条请求只生成一个 token。
seed、调度设置和 prompt 构造不计入此计时。首轮预先定义为预热，其后三轮取中位数。独立前缀；不代表线上文本分布或生产专家路由。</p>
<p>low/high 为 cache/compute 采样层。不同 batch 的请求分布不完全相同，趋势线只描述观测，不能单独解释为 cache 的因果收益。
完整数据：metrics.csv、requests.csv、rounds.csv、interactive_data.json、REPORT.md。</p></main>
<script>__PLOTLY__</script><script>
const rows=__DATA__, palette={"low/low":"#2878b5","low/high":"#24a885","high/low":"#e0a128","high/high":"#cc5476"};
for(const b of [...new Set(rows.map(r=>r.batch_size))].sort((a,b)=>a-b)){let o=document.createElement('option');o.value=b;o.textContent=b;document.getElementById('batch').append(o)}
function draw(){
 const view=document.getElementById('view').value,b=document.getElementById('batch').value,band=document.getElementById('band').value;
 const data=rows.filter(r=>(b==='all'||r.batch_size==b)&&(band==='all'||r.band===band));
 document.getElementById('count').textContent=data.length+' 个点';
 let x,y,z,axis,color,title;
 if(view==='latency'){x=data.map(r=>r.mean_compute_tokens/1024);y=data.map(r=>r.mean_cached_tokens/1024);z=data.map(r=>r.median_batch_ttft_ms);axis=['平均 compute / 请求 (Ki tokens)','平均实际 cache / 请求 (Ki tokens)','整批 TTFT 中位数 (ms)'];color=data.map(r=>r.batch_size);title='Batch 大小 (请求数)'}
 else{let sum=view==='sum';x=data.map(r=>r.batch_size);y=data.map(r=>(sum?r.cached_tokens:r.mean_cached_tokens)/1024);z=data.map(r=>(sum?r.compute_tokens:r.mean_compute_tokens)/1024);axis=['Batch 大小 (请求数)',(sum?'整批总':'每请求平均')+'实际 cache (Ki tokens)',(sum?'整批总':'每请求平均')+' compute (Ki tokens)'];color=data.map(r=>r.median_batch_ttft_ms);title='整批 TTFT 中位数 (ms)'}
 document.getElementById('axis-legend').textContent='X：'+axis[0]+' ｜ Y：'+axis[1]+' ｜ Z：'+axis[2]+' ｜ 颜色：'+title+'。1 Ki tokens = 1024 tokens；compute = input − 实际 cache。';
 const text=data.map(r=>'case '+r.case_id+' · B='+r.batch_size+' · '+r.band+'<br>TTFT median '+r.median_batch_ttft_ms.toFixed(2)+' ms ['+r.min_batch_ttft_ms.toFixed(2)+', '+r.max_batch_ttft_ms.toFixed(2)+']<br>cache/request '+r.mean_cached_tokens.toFixed(1)+' ['+r.min_cached_tokens+','+r.max_cached_tokens+']<br>compute/request '+r.mean_compute_tokens.toFixed(1)+' ['+r.min_compute_tokens+','+r.max_compute_tokens+']<br>request P95 TTFT '+r.p95_request_ttft_ms.toFixed(2)+' ms');
 Plotly.react('plot',[{type:'scatter3d',mode:'markers',x,y,z,text,customdata:data,hovertemplate:'%{text}<extra></extra>',marker:{size:5,opacity:.92,color,colorscale:'Viridis',cmin:view==='latency'?Math.min(...rows.map(r=>r.batch_size)):Math.min(...rows.map(r=>r.median_batch_ttft_ms)),cmax:view==='latency'?Math.max(...rows.map(r=>r.batch_size)):Math.max(...rows.map(r=>r.median_batch_ttft_ms)),colorbar:{title:{text:title}},line:{width:.3,color:'#ffffff'}}}],{margin:{l:45,r:100,t:35,b:55},scene:{xaxis:{title:{text:axis[0],font:{size:13}}},yaxis:{title:{text:axis[1],font:{size:13}}},zaxis:{title:{text:axis[2],font:{size:13}}},camera:{eye:{x:1.7,y:1.6,z:1.2}}},uirevision:view},{responsive:true,displaylogo:false});
 document.getElementById('plot').removeAllListeners('plotly_click');
 document.getElementById('plot').on('plotly_click',event=>{const r=event.points[0].customdata;document.getElementById('detail').textContent='case '+r.case_id+' · B='+r.batch_size+' · '+r.band+'\nFormal rounds (ms): '+r.formal_rounds_ms.map(v=>v.toFixed(3)).join(', ')+'\nrequest | input | observed cache | compute\n'+r.request_distribution.map((v,i)=>[i,...v].join(' | ')).join('\n')});
 const traces=Object.keys(palette).map(key=>{const rs=data.filter(r=>r.band===key);return {type:'scatter',mode:'lines+markers',name:key+' cache/compute',x:rs.map(r=>r.batch_size),y:rs.map(r=>r.median_batch_ttft_ms),line:{color:palette[key]},error_y:{type:'data',symmetric:false,array:rs.map(r=>r.max_batch_ttft_ms-r.median_batch_ttft_ms),arrayminus:rs.map(r=>r.median_batch_ttft_ms-r.min_batch_ttft_ms),thickness:1,width:2}}});
 Plotly.react('trend',traces,{title:{text:'固定采样层内的随机混合 batch（误差线为三轮 min–max）'},xaxis:{title:{text:'Batch 大小 (请求数)'},automargin:true},yaxis:{title:{text:'整批 TTFT 中位数 (ms)'},automargin:true},legend:{orientation:'h'},margin:{t:60,b:70,l:80,r:20}},{responsive:true,displaylogo:false});
}
['view','batch','band'].forEach(id=>document.getElementById(id).onchange=draw);
document.getElementById('reset').onclick=()=>{document.getElementById('view').value='mean';document.getElementById('batch').value='all';document.getElementById('band').value='all';draw()};
draw();
</script></html>"""
    page = page.replace("__COUNT__", str(len(rows))).replace(
        "__PREVIEW__", "部分数据预览" if partial else "全部测量完成"
    )
    page = page.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    )
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Analyzed interactive_data.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Offline HTML output"
    )
    parser.add_argument(
        "--partial", action="store_true", help="Label incomplete data as preview"
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("measurement_contract") != CONTRACT:
        parser.error("Expected batch input_ids wall-time measurement contract")
    if not data.get("rows"):
        parser.error("No chart rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(data["rows"], args.partial), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
