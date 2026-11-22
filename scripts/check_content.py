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
print('PASS: 47 chapters, 45 executed examples, 47 notebooks, 11 modules, assessment quota pools, 14 challenge briefs and output artifacts.')
