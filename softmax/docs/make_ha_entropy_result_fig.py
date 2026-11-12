import numpy as np, pandas as pd, sys, re, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location("szp","structure_zero_polymer.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.append('../code/tests'); from references import get_reference
_,ref=get_reference('zero_polymer'); L=len(ref); ha2=re.search(r"G[IL]FGAIAGF",ref).start()
ent=pd.read_csv('nsites_study/structure/site_entropy_by_T.csv')
nat={k:pd.read_csv(v).entropy_bits.values for k,v in {
  'group 2, w=0.95':'ha_natural/ha_group2_w95_566.csv','all A, w=0.95':'ha_natural/ha_allA_w95_566.csv',
  'human H3N2':'../../FASTA/H3N2/site_entropy_566.csv'}.items()}
k12=np.where(ent['T12'].values<0.05)[0]
x=np.arange(L)
fig=plt.figure(figsize=(14,10.5))
gs=fig.add_gridspec(5,3,height_ratios=[1,1,1,1,1.25],hspace=.75,wspace=.32)
def spans(ax):
    for (a,b),c,n in [((0,16),'0.7','signal'),((16,ha2),'tab:blue','HA1'),((ha2,L),'tab:green','HA2')]:
        ax.axvspan(a,b,color=c,alpha=.08)
    ax.set_xlim(0,L)
rows=list(nat.items())+[('sampled, T12',ent['T12'].values)]
for i,(name,v) in enumerate(rows):
    ax=fig.add_subplot(gs[i,:]); ax.bar(x,np.nan_to_num(v),width=1.,color='0.25' if i<3 else 'tab:red')
    spans(ax); ax.set_ylabel(name,fontsize=8)
    if i==3:
        for k in k12: ax.axvline(k,color='tab:orange',alpha=.5,lw=.8)
        ax.set_xlabel("site (0-based); orange: the 53 T12 K-sites")
    else: ax.set_xticklabels([])
    if i==0:
        for (a,b),n in [((0,16),'signal'),((16,ha2),'HA1 (head)'),((ha2,L),'HA2 (stalk)')]:
            ax.text((a+b)/2,ax.get_ylim()[1]*.88,n,ha='center',fontsize=8)
ax=fig.add_subplot(gs[4,0])
reg=[('signal',0,16),('HA1',16,ha2),('HA2',ha2,L)]
obs=[((k12>=a)&(k12<b)).sum() for _,a,b in reg]; exp=[len(k12)*(b-a)/L for _,a,b in reg]
w=.38; p=np.arange(3)
ax.bar(p-w/2,obs,w,label='observed',color='tab:orange'); ax.bar(p+w/2,exp,w,label='if uniform',color='0.7')
ax.set_xticks(p,[r[0] for r in reg]); ax.set_ylabel('T12 K-sites'); ax.legend(fontsize=7); ax.set_title('K-sites by region',fontsize=9)
ax=fig.add_subplot(gs[4,1])
g2=nat['group 2, w=0.95']; ok=np.isfinite(g2); mask=np.zeros(L,bool); mask[k12]=True
ax.boxplot([g2[mask&ok],g2[~mask&ok]],showfliers=False); ax.set_xticks([1,2],[f'K-sites\n(n={(mask&ok).sum()})',f'rest\n(n={(~mask&ok).sum()})'],fontsize=8)
ax.set_ylabel('natural entropy [bits]'); ax.set_title('group 2, w=0.95',fontsize=9)
ax=fig.add_subplot(gs[4,2])
for name,v in nat.items():
    ax.plot([5,12,32,100],[m.spearman(ent[l].values,v) for l in ['T5','T12','T32','T100']],'o-',label=name,ms=4)
ax.set_xscale('log'); ax.axhline(0,color='0.6',lw=.8); ax.set_xlabel('T'); ax.set_ylabel('Spearman')
ax.legend(fontsize=6.5); ax.set_title('rank correlation vs T',fontsize=9)
fig.suptitle("Sampled vs natural per-site entropy of HA: the T12 K-sites sit in the HA1 head, not the HA2 stalk",fontsize=11)
fig.savefig('../docs/ha_entropy_result_fig.pdf',dpi=130,bbox_inches='tight')
print("ok")
