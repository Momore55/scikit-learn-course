/* DOM integration checks. Install linkedom separately to run; no browser automation. */
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('node:assert/strict');
const {parseHTML}=require('linkedom');
const root=path.resolve(__dirname,'../dist');const course=JSON.parse(fs.readFileSync(path.join(root,'course.json')));
const {window}=parseHTML(fs.readFileSync(path.join(root,'index.html'),'utf8'));const {document}=window;
const memory=new Map(),session=new Map();const storage=m=>({getItem:k=>m.get(k)||null,setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)});
const fixture={edition:1,completed:[],reviewed:[],tests:{},rubrics:{},notes:{},days:[],attempts:{},last:1};
for(let i=1;i<=10;i++){fixture.tests[i]={score:100,confirmed:true};fixture.rubrics[i]={scores:[2,2,2,2],evidence:'Existing project evidence for a retained-progress fixture.'};}
memory.set('scikit-studio-v1',JSON.stringify(fixture));
const location={hash:'#home',reload(){}};window.location=location;window.scrollTo=()=>{};window.print=()=>{};window.HTMLElement.prototype.focus=function(){};window.HTMLElement.prototype.scrollIntoView=function(){};
let copied='';
const ctx={window,document,location,console,localStorage:storage(memory),sessionStorage:storage(session),navigator:{clipboard:{writeText:async s=>{copied=s;}}},fetch:async()=>({ok:true,json:async()=>course}),setTimeout:()=>1,clearTimeout:()=>{},setInterval:()=>1,clearInterval:()=>{},confirm:()=>true,Blob,URL,Math,Date};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),ctx);
const route=hash=>{location.hash=hash;window.dispatchEvent(new window.Event('hashchange'));};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const click=el=>{assert.ok(el);el.dispatchEvent(new window.Event('click',{bubbles:true}));};
(async()=>{
 await tick();assert.ok(document.querySelector('.hero'));
 let labs=0;
 for(const lesson of course.lessons){route('#lesson/'+lesson.id);assert.ok(document.querySelector('h1').textContent.includes(lesson.title));assert.equal(document.querySelectorAll('.deeper-lab').length,(lesson.labs||[]).length);labs+=(lesson.labs||[]).length;assert.equal(document.querySelectorAll('#chapter-quiz .quiz-card').length,5);}
 assert.equal(labs,19);
 route('#lesson/32');click(document.querySelector('[data-action="copy"][data-lab="0"]'));await tick();assert.equal(copied,course.lessons[31].labs[0].code);assert.ok(!copied.includes('OUTPUT ·'));
 assert.equal(document.querySelectorAll('img.lesson-image').length,1);
 route('#lesson/13');const form=document.querySelector('#chapter-quiz');const l=course.lessons[12];const bank=course.modules[3].questions.filter(q=>q.chapter===13);const knowledge=bank.filter(q=>q.type==='Knowledge');const output=bank.filter(q=>q.type==='Predict the output');const extra=knowledge.find(q=>q.id.includes('applied-check'));const chosen=[...knowledge.slice(0,2),...output.slice(0,2),extra];chosen.forEach((q,i)=>{const offset=[...q.id].reduce((sum,c)=>sum+c.charCodeAt(0),0)%q.options.length;const correct=(q.correct-offset+q.options.length)%q.options.length;form.querySelector(`input[name="cq${i}"][value="${correct}"]`).setAttribute('checked','');});form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));assert.ok(JSON.parse(memory.get('scikit-studio-v1')).reviewed.includes(13));
 route('#test/4');assert.ok(document.querySelector('.callout.yellow').textContent.includes('earlier question-bank revision'));click(document.querySelector('[data-action="start-test"]'));const attempt=JSON.parse(session.get('studio-active-test'));assert.equal(attempt.questions.length,20);const counts={};attempt.questions.forEach(q=>counts[q.type]=(counts[q.type]||0)+1);assert.equal(counts.Knowledge,8);assert.equal(counts['Predict the output'],6);assert.equal(counts.Debugging,4);assert.equal(counts['Written explanation'],2);assert.equal(attempt.revision,2);
 attempt.questions.forEach((q,i)=>{if(q.type==='Written explanation'){const el=document.querySelector(`textarea[name="tq${i}"]`);el.value=q.explanation;el.dispatchEvent(new window.Event('input',{bubbles:true}));}else{const el=document.querySelector(`input[name="tq${i}"][value="${q.correct}"]`);el.checked=true;el.dispatchEvent(new window.Event('change',{bubbles:true}));}});document.querySelector('#module-test').dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));let result=JSON.parse(memory.get('scikit-studio-v1')).tests[4];assert.equal(result.correct,18);assert.equal(result.confirmed,false);assert.equal(result.revision,2);
 attempt.questions.map((q,i)=>q.type==='Written explanation'?i:null).filter(i=>i!==null).forEach(i=>{const el=document.querySelector(`[data-written-score="${i}"]`);Object.defineProperty(el,'value',{value:'1',configurable:true});el.dispatchEvent(new window.Event('change',{bubbles:true}));});result=JSON.parse(memory.get('scikit-studio-v1')).tests[4];assert.equal(result.score,100);assert.equal(result.confirmed,true);
 route('#project/4');assert.ok(document.querySelector('main').textContent.includes('multiclass classification_report'));
 route('#resources');assert.ok(document.querySelector('a[href="downloads/practitioner-tooling.md"]'));assert.ok(document.querySelector('main').textContent.includes('revised totals await a learner pilot'));
 console.log('PASS: all 47 routes, 19 lab renderings, lab code copy, plots, mixed lesson checks, retained progress notices, revised test mix and grading, project additions, and tooling resources.');
})().catch(error=>{console.error(error);process.exit(1);});
