"""Oracle reconstruction and latent-only optimization, with frozen networks."""
from diagnose import *


def main():
    selection=json.loads((OUT/'protocol.json').read_text())['selection']
    rows=[]
    for model,path in MODELS.items():
        g=ReactionGenerator(path,'cuda:0')
        for item in selection:
            with np.load(CACHE/item['file']) as z:
                data={k:torch.from_numpy(z[k].copy()).to(g.device) for k in
                      ('features','betas','genders','offsets','text_embeddings')}
            feat=data['features'];ah=feat[0,40:42][None];gh=feat[1,40:42][None];gt=feat[1,42:50][None]
            for name,shift in [('clean',(0,0,0)),('x+30',(.3,0,0)),('x-30',(-.3,0,0))]:
                rh=move(gh,data['offsets'][None,1],shift)
                a,r,origin,basis=condition_window(ah,rh,data['offsets'][None],g.stats)
                local=transform_features(gt,origin,basis,data['offsets'][None,1])
                target=(local-g.stats['mean'])/g.stats['std']
                with torch.no_grad():
                    mu=g.vae.encode_mean(a,r,target)
                latent=mu.clone().requires_grad_(True)
                opt=torch.optim.Adam([latent],lr=.05)
                for step in range(51 if name!='clean' else 1):
                    pred=g.vae.decode(latent,a,r)
                    physical=pred*g.stats['std']+g.stats['mean']
                    loss=F.mse_loss(physical[...,TRANSL],local[...,TRANSL])
                    if step in (0,10,50):
                        with torch.no_grad():
                            world=transform_features(physical,origin,basis,data['offsets'][None,1],inverse=True)
                            world=g.body.repair(world,rh,data['betas'][None,1],data['genders'][None,1])
                            metrics=score(world,rh,gt,gh)
                            metrics['latent_shift_norm']=float((latent-mu).norm())
                        rows.append(dict(model=model,episode=item['episode'],split=item['split'],case=name,
                            optimization_steps=step,metrics=metrics))
                    if step<50 and name!='clean':
                        opt.zero_grad();loss.backward();opt.step()
            save('decoder_probe.json',rows)
            print('decoder',model,item['episode'],'complete',flush=True)
        del g
        torch.cuda.empty_cache()


if __name__=='__main__':main()
