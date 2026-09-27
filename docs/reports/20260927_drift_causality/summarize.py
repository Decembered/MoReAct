from diagnose import *


def averages(rows):
    keys=set.intersection(*(set(r) for r in rows))
    return {k:float(np.mean([r[k] for r in rows])) for k in sorted(keys)
            if isinstance(rows[0][k],(int,float))}


def main():
    results={}
    for model in MODELS:
        results[model]={}
        for file,group_key in [('injections.json','case'),('probes.json','mode')]:
            rows=json.loads((OUT/file).read_text())
            results[model][file[:-5]]={key:averages([r['metrics'] for r in rows if r['model']==model and r[group_key]==key])
                for key in sorted(set(r[group_key] for r in rows))}
        rows=json.loads((OUT/'rollouts.json').read_text())
        results[model]['rollouts']={s:averages([r for r in rows if r['model']==model and r['split']==s]) for s in ('train','test')}
        rows=json.loads((OUT/'gradients.json').read_text())
        results[model]['gradients']={}
        for case in ('clean','x+30','x-30'):
            subset=[r for r in rows if r['model']==model and r['case']==case]
            results[model]['gradients'][case]={}
            for term in subset[0]['terms']:
                values=[r['terms'][term] for r in subset]
                results[model]['gradients'][case][term]={**averages(values),
                    'negative_cosine_fraction':float(np.mean([r['cosine_root']<0 for r in values]))}
        rows=json.loads((OUT/'decoder_probe.json').read_text())
        results[model]['decoder']={}
        for case in ('clean','x+30','x-30'):
            results[model]['decoder'][case]={str(step):averages([r['metrics'] for r in rows
                if r['model']==model and r['case']==case and r['optimization_steps']==step])
                for step in ([0] if case=='clean' else [0,10,50])}
        rows=json.loads((OUT/'history_probe.json').read_text())
        results[model]['history_probe']={mode:averages([r['metrics'] for r in rows if r['model']==model and r['mode']==mode])
            for mode in ('original','gt_local_state_at_drift')}
    save('summary.json',results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes=axes.flatten()
    for name,color in [('root30','#128e83'),('baseline','#c45476')]:
        vals=results[name]['injections']
        labels=['x-30','x-10','clean','x+10','x+30']
        axes[0].plot([-30,-10,0,10,30],[vals[k]['root_cm'] for k in labels],'-o',label=name,color=color)
        modes=['original','root_reset','heading_reset','both_reset','gt_history']
        axes[1].plot(range(5),[results[name]['probes'][k]['root_cm'] for k in modes],'-o',label=name,color=color)
        stages=['0','10','50']
        axes[2].plot([0,10,50],[np.mean([results[name]['decoder'][k][s]['root_cm'] for k in ('x+30','x-30')]) for s in stages],'-o',color=color,label=name)
        axes[3].plot([0,10,50],[np.mean([results[name]['decoder'][k][s]['boundary_step_cm'] for k in ('x+30','x-30')]) for s in stages],'-o',color=color,label=name)
    axes[0].set(xlabel='Injected history X offset (cm)',ylabel='Next 8-frame root error (cm)',title='Frozen sampling response')
    axes[1].set_xticks(range(5));axes[1].set_xticklabels(['Original','Root','Heading','Both','GT history'],rotation=20)
    axes[1].set(ylabel='Next 8-frame root error (cm)',title='Oracle history interventions')
    axes[2].set(xlabel='Latent-only optimization steps',ylabel='Next 8-frame root error (cm)',title='Shared frozen VAE: root-only oracle fit')
    axes[3].set(xlabel='Latent-only optimization steps',ylabel='Boundary root displacement (cm/frame)',title='Root fitting increases boundary jump')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Drift diagnosis | 12 episodes (6 train + 6 test) | fixed checkpoints')
    fig.savefig(OUT/'diagnosis.png',dpi=150)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
