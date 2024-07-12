#%%
import wandb
import argparse
import pandas as pd
import numpy as np
import sqlite3
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--group',type=str,required=False)
parser.add_argument('--new_group',type=str,required=False)
parser.add_argument('--job_id',type=str,required=False)
args,_ = parser.parse_known_args()

#%%
api = wandb.Api(overrides = {'project':'classification_results'})
df = pd.read_csv('classification_results.csv',index_col = 0)
con = sqlite3.connect("classification_scores.db")
cur = con.cursor()

#%%
train_acc = []
valid_acc = []
test_acc = []

#%%
group = args.new_group if args.new_group is not None else args.group
for i,run in enumerate(api.runs(filters = {'group':args.group})):
    if i==5:
        break
    train_acc.append(run.summary_metrics['test_acc/dataloader_idx_0'])
    valid_acc.append(run.summary_metrics['test_acc/dataloader_idx_1'])
    test_acc.append(run.summary_metrics['test_acc/dataloader_idx_2'])

exec_stmnt = (f'INSERT INTO balanced_accuracy VALUES'
    f'("{args.job_id}","{group}",'
    f'{np.mean(train_acc):.2},{np.std(train_acc):.2},'
    f'{np.mean(valid_acc):.2},{np.std(valid_acc):.2},'
    f'{np.mean(test_acc):.2},{np.std(test_acc):.2});').replace('+','')

print(exec_stmnt)
cur.execute(exec_stmnt)

con.commit()
con.close()
