import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'dist'
d=json.loads((p/'course.json').read_text())
assert [l['id'] for l in d['lessons']]==list(range(1,48))
assert len(d['modules'])==11 and len(d['challenge'])==14
assert all((p/'notebooks'/f"chapter-{i:02}.ipynb").exists() for i in range(1,48))
for m in d['modules']:
    bank=m['questions']
    if m['id']<10:
        for kind,n in [('Knowledge',8),('Predict the output',6),('Debugging',4),('Written explanation',2)]:
            assert sum(q['type']==kind for q in bank)>=n,(m['id'],kind)
    for q in bank:
        if q['type']!='Written explanation':assert 0<=q['correct']<len(q['options'])
for l in d['lessons']:
    assert len(l['terms'])==4 and l['idea'] and l['task'] and l['solution']
    if l['code']:assert l['output']
    if l.get('image'):assert (p/l['image']).exists()
report=json.loads((p/'downloads'/'execution-report.json').read_text())
assert len(report['examples'])==45 and all(x['passed'] for x in report['examples'])
assert len(report.get('labs',[])) == 19 and all(x['passed'] for x in report['labs'])
assert report.get('transfer_snippets_verified') == 94
for lesson in d['lessons']:
    for lab in lesson.get('labs',[]):
        assert lab['output'] and lab['reading'] and lab['solution']
        if lab.get('image'): assert (p/lab['image']).exists()
for module in d['modules']:
    for question in module['questions']:
        assert 'frozen environment, which output' not in question['prompt']
        assert 'first line printed' not in question['prompt']
        assert '-review-' not in question['id']
    assert len({q['id'] for q in module['questions']})==len(module['questions'])
assert (p/'downloads'/'practitioner-tooling.md').exists()
print('PASS: 47 chapters, 45 base examples, 19 applied labs, 94 verified transfer snippets, notebooks, assets, assessment quotas and challenge briefs.')
