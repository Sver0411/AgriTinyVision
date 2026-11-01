const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="atv-token"]').content;
let state = null, initialized = false, previewUrl = null, previewBusy = false;
const fmt = (n, digits = 1) => n == null ? '—' : Number(n).toFixed(digits);
const percent = n => n == null ? '—' : `${(n * 100).toFixed(1)}%`;
const reasons = {threshold_interval:'阈值区间判断',decision_boundary:'阈值附近，建议重试',invalid_model_output:'模型输出异常',blur:'图像模糊',exposure:'亮度异常',low_contrast:'对比度不足',invalid_input:'输入无效'};
function reasonText(value) { return (value || '').split(';').filter(Boolean).map(x => reasons[x] || x).join(' · ') || '—'; }
function showError(message) { $('error-banner').textContent = message || ''; $('error-banner').hidden = !message; }
function config() { return {port:$('port').value,baudrate:Number($('baudrate').value),camera:Number($('camera').value),fps:Number($('fps').value),rotation:Number($('rotation').value),mirror:$('mirror').checked}; }
async function action(name, payload = {}) {
  try {
    const response = await fetch(`/api/action/${name}`, {method:'POST',headers:{'Content-Type':'application/json','X-ATV-Token':token},body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '操作失败');
    showError('');
  } catch (error) { showError(error.message); }
}
function metric(id, value, unit) { const el = $(id); el.replaceChildren(document.createTextNode(fmt(value))); const small=document.createElement('small');small.textContent=unit;el.append(small); }
function badge(id, connected, text) { const el=$(id);el.classList.toggle('connected',connected);el.replaceChildren();const dot=document.createElement('span');dot.className='dot';el.append(dot,document.createTextNode(text)); }
function chart(history) {
  $('chart-empty').hidden = history.length > 0;
  if (!history.length) return;
  const maximum=Math.max(100,...history.map(x=>x.end_to_end_ms || 0))*1.18;
  for (const [id,key] of [['line-e2e','end_to_end_ms'],['line-inference','inference_ms']]) {
    const points=history.map((x,i)=>x[key] == null ? null : `${history.length===1?320:i/(history.length-1)*640},${145-x[key]/maximum*135}`).filter(Boolean).join(' ');
    $(id).setAttribute('points', points);
  }
  $('chart-start').textContent=`帧号 ${history[0].sequence} → ${history.at(-1).sequence}`;
  $('chart-scale').textContent=`0 — ${Math.ceil(maximum)} ms`;
}
function render(s) {
  state=s;
  if (!initialized) {
    for (const key of ['baudrate','camera','fps','rotation']) $(key).value=String(s.config[key]);
    $('mirror').checked=s.config.mirror;
    if (s.config.port) {if (![...$('port').options].some(x=>x.value===s.config.port)) $('port').add(new Option(s.config.port,s.config.port));$('port').value=s.config.port;}
    initialized=true;
  }
  $('camera').disabled=s.camera_connected;$('port').disabled=s.device_connected || s.connecting;$('baudrate').disabled=s.device_connected || s.connecting;
  badge('camera-badge',s.camera_connected,s.camera_connected?'相机已连接':'相机未连接');
  badge('device-badge',s.device_connected,s.connecting?'连接中':s.device_connected?'模型已加载':'未连接');
  $('session-chip').classList.toggle('connected',s.device_connected);
  $('session-label').textContent=s.running?'设备推理中':s.device_connected?'设备已就绪':s.connecting?'正在连接':'等待连接';
  $('camera-toggle').textContent=s.camera_connected?'断开相机':'连接相机';
  $('device-toggle').textContent=s.connecting?'连接中…':s.device_connected?'断开 ESP32':'连接 ESP32';
  $('device-toggle').disabled=s.connecting;
  $('start-label').textContent=s.running?'暂停推理':'开始推理';$('start-icon').textContent=s.running?'Ⅱ':'▶';
  $('start').disabled=!s.running && (!s.camera_connected || !s.device_connected);
  $('single').disabled=!s.camera_connected || !s.device_connected || s.busy;
  $('save').disabled=!s.run_directory || s.busy;
  $('camera-placeholder').hidden=s.camera_connected;$('preview').hidden=!s.camera_connected;
  $('capture-label').textContent=s.camera_connected?`摄像头 ${s.config.camera}`:'相机未连接';
  $('capture-size').textContent=`采集尺寸 ${s.camera_size ? s.camera_size.join(' × ') : '—'}`;
  $('capture-count').textContent=`采集帧 ${s.camera_connected?s.capture_count:'—'}`;
  $('completed-note').textContent=`已完成 ${s.completed} 帧`;
  const r=s.result,d=s.device;
  $('result-state').className='result-state '+(r?r.state:'waiting');
  $('state').textContent=r?r.state:'等待推理';$('state-icon').textContent=r?(r.state==='NORMAL'?'✓':r.state==='SUSPECT'?'!':'○'):'○';
  $('sequence').textContent=`帧号 ${r?r.sequence:'—'}`;
  $('p-normal').textContent=percent(r?.normal_probability);$('p-suspect').textContent=percent(r?.suspect_probability);
  $('bar-normal').style.width=`${(r?.normal_probability || 0)*100}%`;$('bar-suspect').style.width=`${(r?.suspect_probability || 0)*100}%`;
  $('confidence').textContent=percent(r?.confidence);$('threshold').textContent=fmt(r?.decision_threshold ?? d?.decision_threshold,2);
  $('quality').textContent=r?(r.quality_passed?'通过':'拒绝'):'—';$('reason').textContent=reasonText(r?.reason);
  metric('m-transfer',r?.transfer_ms,'ms');metric('m-preprocess',r?.preprocess_us==null?null:r.preprocess_us/1000,'ms');
  metric('m-inference',r?.inference_us==null?null:r.inference_us/1000,'ms');metric('m-e2e',r?.end_to_end_ms,'ms');metric('m-fps',r?.actual_fps,'FPS');
  $('quality-ms').textContent=r?`${fmt(r.quality_us/1000)} ms`:'—';$('total-ms').textContent=r?`${fmt(r.total_us/1000)} ms`:'—';$('saturation').textContent=r?.saturated_values==null?'—':`${r.saturated_values} / 76,800`;
  $('versions').textContent=d?`${d.firmware_version} / ${d.model_version}`:'—';
  const memory=r || d;
  $('ram').textContent=memory?`${fmt(memory.free_internal/1024)} KiB`:'—';$('psram').textContent=memory?`${fmt(memory.free_psram/1024/1024,2)} MiB`:'—';
  $('model-load').textContent=d?.model_load_us==null?'—':`${fmt(d.model_load_us/1000)} ms`;
  $('model-hash').textContent=d?`${d.model_sha256.slice(0,12)}…${d.model_sha256.slice(-8)}`:'—';$('model-hash').title=d?.model_sha256 || '';
  $('device-transport').textContent=d?`${s.config.port} · ${d.transport}`:'USB Serial/JTAG';
  $('run-directory').textContent=s.run_directory?`实验记录：${s.run_directory}`:'实验记录：等待设备连接';
  $('log-count').textContent=s.logs.length;
  const logs=$('logs'),atBottom=logs.scrollHeight-logs.scrollTop-logs.clientHeight<40;
  logs.replaceChildren();
  if (!s.logs.length) {const empty=document.createElement('div');empty.className='log-empty';empty.textContent='暂无日志';logs.append(empty);}
  for (const item of s.logs) { const row=document.createElement('div');row.className=`log-entry ${item.level}`;const time=document.createElement('time');time.textContent=new Date(item.timestamp).toLocaleTimeString('zh-CN',{hour12:false});const message=document.createElement('span');message.className='message';message.textContent=item.message;row.append(time,message);logs.append(row); }
  if(atBottom)logs.scrollTop=logs.scrollHeight;
  chart(s.history);showError(s.error);
}
async function poll() {try {const response=await fetch('/api/status');if(!response.ok)throw new Error('服务暂不可用');render(await response.json());}catch(error){showError(`本地服务连接中断：${error.message}`);}finally{setTimeout(poll,500);} }
async function preview() {if(state?.camera_connected && !previewBusy){previewBusy=true;try{const response=await fetch('/api/preview');if(response.status===200){const url=URL.createObjectURL(await response.blob());$('preview').src=url;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=url;}}catch{}finally{previewBusy=false;}}setTimeout(preview,180);}
$('camera-toggle').onclick=()=>action(state?.camera_connected?'disconnect_camera':'connect_camera',{camera:Number($('camera').value)});
$('device-toggle').onclick=()=>action(state?.device_connected?'disconnect_device':'connect_device',{port:$('port').value,baudrate:Number($('baudrate').value)});
$('single').onclick=()=>action('single',{rotation:Number($('rotation').value),mirror:$('mirror').checked});
$('start').onclick=()=>state?.running?action('stop'):action('start',config());
$('save').onclick=()=>action('save');$('clear').onclick=()=>action('clear_logs');
$('download').onclick=event=>{if(!state?.run_directory){event.preventDefault();showError('连接 ESP32 并记录实验后可下载');}};
fetch('/api/ports').then(r=>r.json()).then(ports=>{for(const p of ports)if(![...$('port').options].some(x=>x.value===p.port))$('port').add(new Option(p.port,p.port));if(!$('port').value&&ports.length)$('port').value=ports[0].port;}).catch(()=>{});
poll();preview();
