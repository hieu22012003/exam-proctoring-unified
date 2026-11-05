(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ExamWatchMic = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  const constraints = () => ({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  function wav16(samples, rate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
    const word = (offset, value) => { for (let i=0;i<value.length;i++) view.setUint8(offset+i,value.charCodeAt(i)); };
    word(0,'RIFF');view.setUint32(4,36+samples.length*2,true);word(8,'WAVE');word(12,'fmt ');
    view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
    view.setUint32(24,rate,true);view.setUint32(28,rate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);
    word(36,'data');view.setUint32(40,samples.length*2,true);
    for(let i=0;i<samples.length;i++) {
      const value=Math.max(-1,Math.min(1,Number.isFinite(samples[i])?samples[i]:0));
      view.setInt16(44+i*2,Math.round(value*(value<0?32768:32767)),true);
    }
    return buffer;
  }
  function analyse(samples, rate) {
    let power=0, peak=0, clipped=0, maxFrameRms=0;
    const frameSize=Math.max(1,Math.round(rate*.02));
    for(let offset=0;offset<samples.length;offset+=frameSize) {
      let framePower=0;
      const end=Math.min(samples.length,offset+frameSize);
      for(let i=offset;i<end;i++) {
        const value=Number.isFinite(samples[i])?samples[i]:0;
        framePower+=value*value;peak=Math.max(peak,Math.abs(value));
        if(Math.abs(value)>=.999)clipped++;
      }
      power+=framePower;
      maxFrameRms=Math.max(maxFrameRms,Math.sqrt(framePower/(end-offset)));
    }
    const rms=Math.sqrt(power/Math.max(1,samples.length));
    return {seconds:samples.length/rate,rate,rms,peak,maxFrameRms,
      clippedFraction:clipped/Math.max(1,samples.length),
      level:clipped/Math.max(1,samples.length)>.01?'clipped':maxFrameRms<.008?'quiet':'usable'};
  }
  async function capture(sharedStream, progress=()=>{}) {
    let stream=sharedStream, owned=false, context, source, node;
    try {
      if(!stream?.getAudioTracks().some(t=>t.readyState==='live')) {
        stream=await navigator.mediaDevices.getUserMedia(constraints());owned=true;
      }
      const track=stream.getAudioTracks()[0];
      context=new AudioContext({sampleRate:16000});
      await context.audioWorklet.addModule('/static/mic-probe-worklet.js');
      await context.resume();
      source=context.createMediaStreamSource(stream);
      node=new AudioWorkletNode(context,'examwatch-mic-probe');
      const data=await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Không nhận đủ âm thanh. Hãy kiểm tra micro rồi thử lại.')),12000);
        node.onprocessorerror=()=>{clearTimeout(timeout);reject(new Error('Không đọc được dữ liệu từ micro.'));};
        node.port.onmessage=({data})=>{
          if(data.samples){clearTimeout(timeout);resolve(data);}
          else if(data.seconds!==undefined)progress(data.seconds);
        };
        source.connect(node);node.connect(context.destination);
      });
      const stats=analyse(data.samples,data.rate);
      return {wav:new Blob([wav16(data.samples,data.rate)],{type:'audio/wav'}), stats,
        device:track.label||'Micro mặc định', settings:track.getSettings()};
    } finally {
      node?.disconnect();source?.disconnect();
      if(owned)stream?.getTracks().forEach(t=>t.stop());
      if(context&&context.state!=='closed')await context.close();
    }
  }
  return {constraints,wav16,analyse,capture};
});
