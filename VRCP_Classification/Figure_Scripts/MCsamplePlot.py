import pandas as pd
import numpy as np

eps = 0.03
smooth = [256, 512, 1024, 2048, 4096]
methods = ["HPS_verif","HPS_verif_pre","HPS_RSCP+","PTT_HPS_RSCP+"]

noisy_results = pd.DataFrame()
for sm in smooth:
    directory = "Results/CIFAR10/epsilon_"+str(eps)+"/sigma_model_0/sigma_smooth_"+str(eps*2)+"/n_smooth_"+str(sm)+"/Robust/simple/sample_holdout/"
    with open(directory + "results10000CIFAR10_"+str(eps)+".csv", 'rb') as f:
        results = pd.read_csv(f)
    results = results.drop(['Size list'],axis=1)
    results['Number of MC samples'] = str(sm)
    results = results[(results['Method'].isin(methods)) & (results['noise_L2_norm'] != 0.0)]
    print(results.head())
    noisy_results = pd.concat([noisy_results, results])

noisy_results = noisy_results.replace(methods,['VRCP_I','VRCP_C','RSCP+','RSCP+ (PTT)'])

noisy_results.rename(columns = {'noise_L2_norm':'Epsilon', 'Size':'Average Set Size', 'Coverage':'Marginal Coverage'}, inplace = True) 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager


sns.set_theme(style="darkgrid")
plt.style.use('ggplot')

COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
fig = sns.lineplot(data=noisy_results, x="Number of MC samples", y="Marginal Coverage", hue='Method')
plt.tight_layout()
plt.savefig("./Results/variedMCsCoverage.pdf")
plt.close() 
new = sns.lineplot(data=noisy_results, x="Number of MC samples", y="Average Set Size", hue='Method')
plt.tight_layout()
plt.savefig("./Results/variedMCs.pdf")