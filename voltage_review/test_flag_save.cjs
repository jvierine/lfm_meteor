const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/app.js','utf8');
const saveSource=source.slice(source.indexOf('async function save('),source.indexOf('\nfunction table('));
function fixture(conflict){
 const annotation={version:1,excluded:[],bad:false,note:''},calls=[],elements={};
 const context={structuredClone,data:{event:'test',fingerprint:'old',annotation},busy:false,request:0,pendingEvent:null,undo:[],catalog:[],
  $:id=>elements[id]??={},controls(){},sidebar(){},table(){},draw(){},status(message){context.message=message;},
  async api(path,body){calls.push({path,body});if(calls.length===1)throw Error('Fit data changed; reload before saving');
   if(!body)return {event:'test',fingerprint:'new',annotation:{...annotation,version:conflict?2:1}};
   return {...body,version:2};}};
 vm.createContext(context);vm.runInContext(saveSource,context);return {context,calls,elements};
}
test('selected pulse flags survive a concurrently completed refit',async()=>{
 const {context,calls,elements}=fixture(false);await context.save({excluded:['0:123']});
 assert.equal(calls.length,3);assert.equal(calls[2].body.fingerprint,'new');
 assert.deepEqual(calls[2].body.excluded,['0:123']);assert.equal(elements.saved.textContent,'Saved revision 2');
 assert.equal(context.busy,false);
});
test('a conflicting review is never overwritten',async()=>{
 const {context,calls,elements}=fixture(true);await context.save({excluded:['0:123']});
 assert.equal(calls.length,2);assert.equal(elements.saved.textContent,'Not saved');
 assert.match(context.message,/Review changed elsewhere/);
});
test('beam plots take keyboard focus before selection',()=>{
 const beam=fs.readFileSync(__dirname+'/beam.js','utf8');
 assert.match(beam,/pointerdown.*canvas\.focus/);
});
test('navigation waits for a confirmed save and invalidates earlier loads',async()=>{
 const {context}=fixture(false);context.pendingEvent='next';
 context.load=async id=>{assert.equal(context.busy,false);assert.equal(context.data.annotation.version,2);context.navigated=id;};
 await context.save({excluded:['0:123']});
 assert.equal(context.request,1);assert.equal(context.navigated,'next');assert.equal(context.pendingEvent,null);
});
test('failed save does not navigate away and lose selected work',async()=>{
 const {context}=fixture(true);context.pendingEvent='next';
 context.load=()=>assert.fail('must stay on unsaved event');
 await context.save({excluded:['0:123']});assert.equal(context.pendingEvent,null);
});
