/* Vertex — theme, charts, loading and shared AI assistant */
(function(){
  var saved;
  try{saved=localStorage.getItem('vertex-theme');}catch(e){saved=null;}
  document.documentElement.setAttribute('data-theme',saved==='light'?'light':'dark');
})();

function vxTheme(){return document.documentElement.getAttribute('data-theme')||'dark';}
function vxToggleTheme(){
  var next=vxTheme()==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',next);
  try{localStorage.setItem('vertex-theme',next);}catch(e){}
  var btn=document.getElementById('theme-toggle');
  if(btn)btn.textContent=next==='dark'?'☀':'☾';
  window.dispatchEvent(new CustomEvent('vx-themechange',{detail:{theme:next}}));
}
function vxInitToggle(){var btn=document.getElementById('theme-toggle');if(!btn)return;btn.textContent=vxTheme()==='dark'?'☀':'☾';btn.onclick=vxToggleTheme;}

function vxChartColors(){
  var s=getComputedStyle(document.documentElement),v=function(n){return s.getPropertyValue(n).trim();};
  return{paper:v('--surface'),plot:v('--surface-2'),grid:v('--chart-grid'),line:v('--chart-line'),text:v('--text-dim'),accent:v('--accent'),ok:v('--ok'),warn:v('--warn'),bad:v('--bad'),danger:v('--danger')};
}
var VX_SERIES=['#5c8df6','#ef7474','#30b893','#e7aa45','#9b6be6','#50acd6','#df6bb5','#78b857','#e48552','#5aa6a6'];
function vxBaseLayout(extra){
  var c=vxChartColors();
  var base={paper_bgcolor:c.paper,plot_bgcolor:c.plot,font:{color:c.text,size:12,family:"'Pretendard Variable', Pretendard, 'Segoe UI', sans-serif"},margin:{l:65,r:20,t:24,b:56},legend:{bgcolor:'rgba(0,0,0,0)',bordercolor:c.line,borderwidth:1,orientation:'v',yanchor:'top',y:1,xanchor:'left',x:1.01},xaxis:{gridcolor:c.grid,linecolor:c.line,zerolinecolor:c.line},yaxis:{gridcolor:c.grid,linecolor:c.line,zerolinecolor:c.line},hovermode:'x unified'};
  if(!extra)return base;
  var out=Object.assign({},base,extra);
  if(extra.xaxis)out.xaxis=Object.assign({},base.xaxis,extra.xaxis);
  if(extra.yaxis)out.yaxis=Object.assign({},base.yaxis,extra.yaxis);
  if(extra.legend)out.legend=Object.assign({},base.legend,extra.legend);
  return out;
}
function vxHeatmapScale(){
  if(vxTheme()==='dark')return[[0,'#111827'],[.16,'#1d3150'],[.34,'#284d73'],[.52,'#2f7c8f'],[.7,'#44a982'],[.86,'#b6d66b'],[1,'#f4d66b']];
  return[[0,'#dce8f8'],[.2,'#aecce8'],[.4,'#75b6d5'],[.6,'#47a99d'],[.8,'#69b86a'],[1,'#d0b84f']];
}

var _vxLovRaf=null,_vxLovStepTimer=null;
function vxShowLoading(opts){
  opts=opts||{};var title=opts.title||'처리 중...',steps=opts.steps&&opts.steps.length?opts.steps:['처리 중…'];
  var titleEl=document.querySelector('#lov .lv-title'),stepEl=document.getElementById('lv-step');if(titleEl)titleEl.textContent=title;
  var canvas=document.getElementById('lv-canvas');
  if(canvas){var ctx=canvas.getContext('2d');canvas.width=window.innerWidth;canvas.height=window.innerHeight;var pts=Array.from({length:55},function(){return{x:Math.random()*canvas.width,y:Math.random()*canvas.height,r:Math.random()*1.6+.6,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,a:Math.random()*.45+.1};});
    (function frame(){ctx.clearRect(0,0,canvas.width,canvas.height);for(var i=0;i<pts.length;i++){var p=pts[i];p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=canvas.width;if(p.x>canvas.width)p.x=0;if(p.y<0)p.y=canvas.height;if(p.y>canvas.height)p.y=0;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(92,141,246,'+p.a+')';ctx.fill();for(var j=i+1;j<pts.length;j++){var dx=p.x-pts[j].x,dy=p.y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<105){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(92,141,246,'+((1-d/105)*.11)+')';ctx.lineWidth=.7;ctx.stroke();}}}_vxLovRaf=requestAnimationFrame(frame);})();
  }
  if(stepEl){var si=0;stepEl.textContent=steps[0];clearInterval(_vxLovStepTimer);_vxLovStepTimer=setInterval(function(){si=(si+1)%steps.length;stepEl.textContent=steps[si];},opts.interval||1200);}
  var lov=document.getElementById('lov');if(lov)lov.classList.add('show');
}
function vxHideLoading(){if(_vxLovRaf){cancelAnimationFrame(_vxLovRaf);_vxLovRaf=null;}clearInterval(_vxLovStepTimer);var lov=document.getElementById('lov');if(lov)lov.classList.remove('show');}

function vxEsc(v){return String(v==null?'':v).replace(/[&<>"']/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function vxNl(v){return vxEsc(v).replace(/\n/g,'<br>');}
function vxFmtHours(h){h=Number(h);if(!isFinite(h)||h<=0)return'—';if(h>=876000)return'약 '+(h/8760).toLocaleString('ko-KR',{maximumFractionDigits:0})+'년';if(h>=8760)return'약 '+(h/8760).toLocaleString('ko-KR',{maximumFractionDigits:1})+'년';if(h>=1000000)return(h/1000000).toFixed(2)+'M h';if(h>=1000)return(h/1000).toFixed(1)+'k h';return h.toLocaleString('ko-KR',{maximumFractionDigits:1})+' h';}
function vxAnswerText(obj){if(obj==null)return'';if(typeof obj==='string')return obj;return obj.answer||obj.content||obj.message||'';}
function vxIntentLabel(v){return({prediction:'Prediction',what_if:'What-if',knowledge:'Knowledge',general_chat:'Chat'})[v]||v;}

function vxMountAssistant(opts){
  opts=opts||{};var embedded=!!opts.embedded,target=embedded?document.querySelector(opts.target||'#assistant-root'):document.body;if(!target)return null;
  var id='vx-assistant-'+Math.random().toString(36).slice(2,8);
  var panel=document.createElement('section');panel.id=id;panel.className='vx-assistant'+(embedded?' vx-assistant-embedded open':'');
  panel.innerHTML='<div class="vx-ai-head"><div class="vx-ai-identity"><div class="vx-ai-logo">V</div><div><div class="vx-ai-title">Vertex AI</div><div class="vx-ai-status">RAG · Predictor · What-if</div></div></div><button class="vx-ai-close" type="button" aria-label="닫기">×</button></div><div class="vx-ai-context">'+vxEsc(opts.contextLabel||'문헌 기반 재료 분석 어시스턴트')+'</div><div class="vx-ai-messages"></div><div class="vx-ai-quick"></div><form class="vx-ai-form"><textarea rows="1" placeholder="재료, 수명, 조성 변경에 대해 질문하세요"></textarea><button class="vx-ai-send" type="submit" aria-label="전송">↑</button></form>';
  target.appendChild(panel);

  var launcher=null;
  if(!embedded){
    launcher=document.createElement('button');
    launcher.type='button';
    launcher.className='vx-ai-launcher';
    launcher.innerHTML='<span class="ai-orb"><span class="ai-spark">✦</span></span><span class="ai-launch-copy"><b>Vertex AI</b><small>무엇이든 질문하세요</small></span><span class="ai-launch-arrow">→</span>';
    document.body.appendChild(launcher);
  }

  var messages=panel.querySelector('.vx-ai-messages'),
      form=panel.querySelector('form'),
      input=panel.querySelector('textarea'),
      sendBtn=panel.querySelector('.vx-ai-send'),
      quick=panel.querySelector('.vx-ai-quick'),
      contextEl=panel.querySelector('.vx-ai-context'),
      busy=false;

  var prompts=opts.quickPrompts||[
    '9Cr강에서 W의 역할은?',
    'Cr을 많이 넣으면 장기 크립 성능이 좋아져?',
    '크립 강도와 크립 수명의 차이는?'
  ];

  prompts.forEach(function(q){
    var b=document.createElement('button');
    b.type='button';
    b.textContent=q;
    b.onclick=function(){input.value=q;form.requestSubmit();};
    quick.appendChild(b);
  });

  function scrollEnd(){messages.scrollTop=messages.scrollHeight;}

  function add(role,html,meta){
    var row=document.createElement('div');
    row.className='vx-msg '+role;
    row.innerHTML='<div class="vx-msg-bubble">'+html+(meta?'<div class="vx-msg-meta">'+meta+'</div>':'')+'</div>';
    messages.appendChild(row);
    scrollEnd();
    return row;
  }

  add('ai','안녕하세요. <b>Vertex AI</b>입니다.<br>궁금한 것을 물어보세요.');

  function renderSources(k){
    var src=(k&&k.sources)||[];
    if(!src.length)return'';

    var seen={},
        html='<div class="vx-ai-sources"><div class="vx-ai-result-label">Sources</div>';

    src.slice(0,5).forEach(function(s){
      var title=s.title||s.source||'문헌',
          author=s.author||'',
          year=s.year||'',
          ps=s.page_start!=null?s.page_start:s.page,
          pe=s.page_end!=null?s.page_end:ps,
          page=ps!=null?(ps===pe?'p. '+ps:'p. '+ps+'–'+pe):'',
          key=title+'|'+page;

      if(seen[key])return;
      seen[key]=1;

      html+='<div class="vx-ai-source"><div class="vx-ai-source-title">'+vxEsc(title)+'</div><div class="vx-ai-source-meta">'+vxEsc([author,year,page].filter(Boolean).join(' · '))+'</div></div>';
    });

    return html+'</div>';
  }

  function changedComposition(w){
    try{
      var a=w.before.composition||{},
          b=w.after.composition||{},
          out=[];

      Object.keys(Object.assign({},a,b)).forEach(function(k){
        var av=Number(a[k]||0),
            bv=Number(b[k]||0);

        if(Math.abs(av-bv)>1e-10)
          out.push(k+' '+av+' → '+bv);
      });

      return out.slice(0,6).join(' · ');
    }catch(e){
      return'';
    }
  }

  function renderPayload(data){
    var r=data.results||{},
        routing=data.routing||{},
        intents=routing.intents||[],
        html='';

    if(intents.length){
      html+='<div class="vx-ai-tags">'+intents.map(function(x){
        return'<span class="vx-ai-tag">'+vxEsc(vxIntentLabel(x))+'</span>';
      }).join('')+'</div>';
    }

    if(r.general_chat){
      html+='<div>'+vxNl(vxAnswerText(r.general_chat))+'</div>';
    }

    if(r.what_if){
      var w=r.what_if,
          bh=w.before&&w.before.prediction?w.before.prediction.life_hours:null,
          ah=w.after&&w.after.prediction?w.after.prediction.life_hours:null,
          cp=w.comparison||{},
          changed=changedComposition(w);

      html+='<div class="vx-ai-result-card">'+
        '<div class="vx-ai-result-label">What-if comparison</div>'+
        '<div class="vx-ai-compare">'+
        '<div class="vx-ai-mini"><span>Before</span><b>'+vxEsc(vxFmtHours(bh))+'</b></div>'+
        '<div class="vx-ai-mini"><span>After</span><b>'+vxEsc(vxFmtHours(ah))+'</b></div>'+
        '</div>'+
        '<div class="vx-ai-result-sub">변화 '+(cp.life_change_percent==null?'—':Number(cp.life_change_percent).toFixed(1)+'%')+
        (changed?'<br>'+vxEsc(changed):'')+
        '</div></div>';
    }
    else if(r.prediction){
      html+='<div class="vx-ai-result-card">'+
        '<div class="vx-ai-result-label">Predicted creep life</div>'+
        '<div class="vx-ai-result-value">'+vxEsc(vxFmtHours(r.prediction.life_hours))+'</div>'+
        '<div class="vx-ai-result-sub">log₁₀(life[h]) = '+Number(r.prediction.log_life).toFixed(3)+' · 프로젝트 예측 모델 결과</div>'+
        '</div>';
    }

    if(r.knowledge){
      var kt=vxAnswerText(r.knowledge);
      html+='<div'+(r.prediction||r.what_if?' style="margin-top:10px"':'')+'>'+vxNl(kt)+'</div>'+renderSources(r.knowledge);
    }

    if(!html){
      html='<span class="vx-ai-error">답변 결과를 표시할 수 없습니다.</span>';
    }

    var t=data.timings||{},
        meta=t.total_seconds!=null?'총 '+Number(t.total_seconds).toFixed(1)+'초':'';

    add('ai',html,meta);
  }

  async function submitQuestion(q){
    if(busy||!q.trim())return;

    busy=true;
    sendBtn.disabled=true;
    add('user',vxNl(q));
    input.value='';

    var wait=add(
      'ai',
      '<span class="vx-ai-typing"><i></i><i></i><i></i></span> <span class="vx-muted">검색하고 답변을 구성하는 중입니다…</span>'
    );

    var started=Date.now(),
        timer=setInterval(function(){
          var meta=wait.querySelector('.vx-msg-meta');

          if(!meta){
            meta=document.createElement('div');
            meta.className='vx-msg-meta';
            wait.querySelector('.vx-msg-bubble').appendChild(meta);
          }

          meta.textContent=Math.floor((Date.now()-started)/1000)+'초 경과';
        },1000);

    try{
      var state=typeof opts.stateProvider==='function'?opts.stateProvider():null;

      if(typeof opts.contextLabelProvider==='function'){
        var label=opts.contextLabelProvider();
        if(label)
          contextEl.innerHTML='<strong>현재 컨텍스트</strong> · '+vxEsc(label);
      }

      var resp=await fetch('/api/assistant',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          question:q,
          state:state
        })
      });

      var data=await resp.json().catch(function(){return{};});

      if(!resp.ok||data.ok===false)
        throw new Error(data.message||('서버 오류 '+resp.status));

      wait.remove();
      renderPayload(data);

    }catch(e){
      wait.remove();

      add(
        'ai',
        '<span class="vx-ai-error">'+vxEsc(e.message||'요청 처리 중 오류가 발생했습니다.')+'</span>'+
        '<div class="vx-msg-meta">Knowledge 모델 또는 What-if API가 준비되어 있는지 확인해주세요.</div>'
      );

    }finally{
      clearInterval(timer);
      busy=false;
      sendBtn.disabled=false;
      input.focus();
    }
  }

  form.addEventListener('submit',function(e){
    e.preventDefault();
    submitQuestion(input.value);
  });

  input.addEventListener('keydown',function(e){
    if(e.key==='Enter'&&!e.shiftKey){
      e.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener('input',function(){
    this.style.height='auto';
    this.style.height=Math.min(this.scrollHeight,110)+'px';
  });

  panel.querySelector('.vx-ai-close').onclick=function(){
    if(embedded)return;
    panel.classList.remove('open');
  };

  if(embedded)
    panel.querySelector('.vx-ai-close').style.display='none';

  if(launcher)
    launcher.onclick=function(){
      panel.classList.toggle('open');

      if(panel.classList.contains('open'))
        setTimeout(function(){input.focus();},120);
    };

  return{
    open:function(){panel.classList.add('open');},
    close:function(){panel.classList.remove('open');},
    send:submitQuestion
  };
}

document.addEventListener('DOMContentLoaded',vxInitToggle);