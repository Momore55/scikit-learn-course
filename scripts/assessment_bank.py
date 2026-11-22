from transfer_questions import output_questions, debugging_questions
from lesson_labs import CHECKS

def build_question_banks(modules, lessons):
    # The first twenty questions are selected from chapter-specific items, not repeated placeholder questions.
    for module in modules:
        bank=[]
        for l in lessons:
            if l['module']!=module['id']:continue
            for j,t in enumerate(l['terms']):
                distractors=[x['meaning'] for x in l['terms'] if x!=t]
                bank.append(dict(id=f"{l['id']}-term-{j}",chapter=l['id'],type='Knowledge',prompt=f'What does “{t["term"]}” mean in this lesson?',options=[t['meaning']]+distractors,correct=0,explanation=t['meaning']))
            bank.extend(output_questions(l['id']))
            for j,(prompt,answer,wrong) in enumerate(CHECKS.get(l['id'], [])):
                bank.append(dict(id=f"{l['id']}-applied-check-{j}",chapter=l['id'],type='Knowledge',prompt=prompt,options=[answer,wrong],correct=0,explanation=answer))
            bank.append(dict(id=f"{l['id']}-reason",chapter=l['id'],type='Written explanation',prompt=l['question'],options=[],correct=0,explanation=l['answer']))
            for j,lab in enumerate(l['labs']):
                bank.append(dict(id=f"{l['id']}-lab-reason-{j}",chapter=l['id'],type='Written explanation',prompt=lab['task'],options=[],correct=0,explanation=lab['solution']))
        bank.extend(debugging_questions(module['id'],module['start']))
        module['questions']=bank
    return modules
