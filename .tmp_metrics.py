import csv
from pathlib import Path
rank_path=Path(r'd:/tox-agent/models/dualhead_model_ranking.csv')
rows=list(csv.DictReader(rank_path.open()))
print('RANK_ROWS', len(rows))
print('TOP3', '; '.join(['{}:{:.4f}'.format(r['model'], float(r['joint_auc_beta3'])) for r in rows[:3]]))
vals=[float(r['joint_auc_beta3']) for r in rows]
print('JOINT_AUC_RANGE {:.4f}-{:.4f}'.format(min(vals), max(vals)))
print('JOINT_AUC_MEAN {:.4f}'.format(sum(vals)/len(vals)))

task_path=Path(r'd:/tox-agent/models/tox21_gatv2_model/tox21_task_metrics.csv')
tasks=list(csv.DictReader(task_path.open()))
auc=[float(r['auc_roc']) for r in tasks]
pra=[float(r['pr_auc']) for r in tasks]
top_tasks=sorted(tasks,key=lambda r:float(r['auc_roc']), reverse=True)[:3]
print('TASK_ROWS', len(tasks))
print('TOP_TASKS', '; '.join(['{}:{:.3f}'.format(r['task'], float(r['auc_roc'])) for r in top_tasks]))
print('TASK_AUC_MEAN {:.3f}'.format(sum(auc)/len(auc)))
print('TASK_PR_MEAN {:.3f}'.format(sum(pra)/len(pra)))
