import sys,json,pickle,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
sys.path.insert(0,str(Path.cwd()))
from moreact.geometry import JOINTS
from moreact.render import PARENTS
out=Path('outputs/remogen_generation_comparison'); results={}
for item in json.loads((out/'selection.json').read_text()):
 name=item['category']; ours=np.load(f'outputs/stage2_best_gifs/{name}/motion.npz')
 r=pickle.loads((out/'official/text/seed_0'/f'{item["record_id"]}.pkl').read_bytes())
 a,b,g=[ours[k][:,JOINTS].reshape(-1,22,3) for k in ['actor','reactor','target']]
 ra,rb,rg=[r[k]['joints'][2:] for k in ['actor','reactor','gt_reactor']]
 n=min(len(b),len(rb));a,b,g,ra,rb,rg=[x[:n] for x in [a,b,g,ra,rb,rg]]
 # Static rigid coordinate alignment from ground-truth bodies only, never predictions.
 X=np.concatenate([ra.reshape(-1,3),rg.reshape(-1,3)]);Y=np.concatenate([a.reshape(-1,3),g.reshape(-1,3)])
 u,s,vt=np.linalg.svd((X-X.mean(0)).T@(Y-Y.mean(0)));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);R=u@d@vt;t=Y.mean(0)-X.mean(0)@R
 ra,rb,rg=[x@R+t for x in [ra,rb,rg]]
 def metrics(p,q,actor):
  return dict(mpjpe_cm=float(np.linalg.norm(p-q,axis=-1).mean()*100),root_ade_cm=float(np.linalg.norm(p[:,0]-q[:,0],axis=-1).mean()*100),aligned_mpjpe_cm=float(np.linalg.norm((p-p[:,:1])-(q-q[:,:1]),axis=-1).mean()*100),pair_distance_error_cm=float(np.abs(np.linalg.norm(p[:,0]-actor[:,0],axis=-1)-np.linalg.norm(q[:,0]-actor[:,0],axis=-1)).mean()*100))
 results[name]=dict(frames=n,MoReAct=metrics(b,g,a),ReMoGen=metrics(rb,rg,ra),gt_alignment_residual_cm=float(np.linalg.norm(X@R+t-Y,axis=-1).mean()*100))
 np.savez_compressed(out/f'{name}_aligned.npz',actor=a,moreact=b,truth=g,remogen_actor=ra,remogen=rb,remogen_truth=rg)
 points=np.concatenate([x.reshape(-1,3) for x in [a,b,g,ra,rb]]);low=points.min(0);high=points.max(0);center=(low+high)/2;rad=max((high-low)[:2].max()/2+.2,1.)
 fig=plt.figure(figsize=(15,5)); lines=[]
 for i,(actor,pred,title) in enumerate([(a,b,'MoReAct | step 8750'),(ra,rb,'ReMoGen | official HHI'),(a,g,'Dataset reference')]):
  ax=fig.add_subplot(1,3,i+1,projection='3d');ax.set(xlim=(center[0]-rad,center[0]+rad),ylim=(center[1]-rad,center[1]+rad),zlim=(min(0,low[2]-.1),max(2,high[2]+.1)),title=title);ax.view_init(elev=18,azim=-65);ax.set_box_aspect((2*rad,2*rad,max(2,high[2]+.1)-min(0,low[2]-.1)))
  for x,col in [(actor,'#2475bd'),(pred,'#ec7934')]:
   for j,p in enumerate(PARENTS):
    if p>=0:
     line,=ax.plot([],[],[],color=col,lw=2);lines.append((line,x,p,j))
 label=fig.suptitle('');fig.text(.5,.03,'Blue: observed actor | Orange: reactor | Same source frames, seed 0 | Display: 30 source frames/s (timing under audit)',ha='center',fontsize=10)
 writer=FFMpegWriter(fps=15,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
 with writer.saving(fig,str(out/f'{name}.mp4'),dpi=90):
  for f in range(0,n,2):
   for line,x,p,j in lines:
    xyz=x[f,[p,j]];line.set_data(xyz[:,0],xyz[:,1]);line.set_3d_properties(xyz[:,2])
   label.set_text(f'{name.title()} | source sample {f+2}');writer.grab_frame()
   if f in [0,n//4*2,n//2*2-2]:fig.savefig(out/f'{name}_frame{f}.png')
 plt.close(fig)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/f'{name}.mp4'),'-filter_complex_threads','1','-filter_complex','fps=15,scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse','-loop','0',str(out/f'{name}.gif')],check=True)
 print(name,results[name],flush=True)
(out/'metrics.json').write_text(json.dumps(results,indent=2))
