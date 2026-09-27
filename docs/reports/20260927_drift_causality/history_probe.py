"""Replace local history shape/dynamics while preserving endpoint root and yaw."""
from diagnose import *


def main():
    selection=json.loads((OUT/'protocol.json').read_text())['selection'];rows=[]
    for model,path in MODELS.items():
        g=ReactionGenerator(path,'cuda:0')
        for item in selection:
            with np.load(CACHE/item['file']) as z:
                data={k:torch.from_numpy(z[k].copy()).to(g.device) for k in
                      ('features','betas','genders','offsets','text_embeddings')}
            feat=data['features']
            with torch.no_grad():
                for seed in (0,1,2):
                    rh=feat[1,:2][None].clone()
                    for cutoff in range(2,83,8):
                        ah=feat[0,cutoff-2:cutoff][None];gh=feat[1,cutoff-2:cutoff][None];gt=feat[1,cutoff:cutoff+8][None]
                        segseed=seed*1000+cutoff;pred=predict(g,ah,rh,data,segseed)
                        if cutoff in (42,82):
                            delta=(rh[0,-1,TRANSL]-gh[0,-1,TRANSL]).tolist()
                            angle=float(wrap(yaw(rh)[0,-1]-yaw(gh)[0,-1]))
                            alternative=move(gh,data['offsets'][None,1],delta,angle)
                            future=predict(g,ah,alternative,data,segseed)
                            for mode,h,p in [('original',rh,pred),('gt_local_state_at_drift',alternative,future)]:
                                rows.append(dict(model=model,episode=item['episode'],split=item['split'],seed=seed,
                                    cutoff=cutoff,mode=mode,metrics=score(p,h,gt,gh),
                                    endpoint_position_match_max_m=float((alternative[0,-1,TRANSL]-rh[0,-1,TRANSL]).abs().max()),
                                    endpoint_yaw_match_deg=float(wrap(yaw(alternative)[0,-1]-yaw(rh)[0,-1]).abs()*180/math.pi)))
                        rh=pred[:,-2:]
            save('history_probe.json',rows)
            print('history',model,item['episode'],'complete',flush=True)
        del g;torch.cuda.empty_cache()


if __name__=='__main__':main()
