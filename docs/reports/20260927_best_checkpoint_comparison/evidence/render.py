import sys,subprocess,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
sys.path.insert(0,str(Path.cwd()))
from moreact.geometry import JOINTS
from moreact.render import PARENTS
out=Path('outputs/best_compare_20260927')
for name in ['hug','handshake']:
 paths=[Path('outputs/stage2_best_gifs')/name,out/'best'/name,out/'best_rollout'/name]
 data=[np.load(p/'motion.npz') for p in paths];actor=data[0]['actor'][:,JOINTS].reshape(-1,22,3);truth=data[0]['target'][:,JOINTS].reshape(-1,22,3)
 pred=[z['reactor'][:,JOINTS].reshape(-1,22,3) for z in data]
 for z in data:np.testing.assert_array_equal(z['actor'],data[0]['actor']);np.testing.assert_array_equal(z['target'],data[0]['target'])
 points=np.concatenate([x.reshape(-1,3) for x in [actor,truth,*pred]]);low=points.min(0);high=points.max(0);center=(low+high)/2;rad=max((high-low)[:2].max()/2+.2,1.)
 fig=plt.figure(figsize=(16,4.8));lines=[]
 for i,(p,title) in enumerate(zip([*pred,truth],['Previous | 8750','Current best.pt | 13250','Current best_rollout.pt | 19750','Ground truth'])):
  ax=fig.add_subplot(1,4,i+1,projection='3d');ax.set(xlim=(center[0]-rad,center[0]+rad),ylim=(center[1]-rad,center[1]+rad),zlim=(min(0,low[2]-.1),max(2,high[2]+.1)),title=title);ax.view_init(elev=18,azim=-65);ax.set_box_aspect((2*rad,2*rad,max(2,high[2]+.1)-min(0,low[2]-.1)))
  for x,col in [(actor,'#2475bd'),(p,'#ec7934')]:
   for j,parent in enumerate(PARENTS):
    if parent>=0:
     line,=ax.plot([],[],[],color=col,lw=2);lines.append((line,x,parent,j))
 label=fig.suptitle('');fig.text(.5,.04,'Blue: actor | Orange: reactor | Identical initial history, text, seed 0, guidance 1 | Source-frame synchronized',ha='center',fontsize=10)
 writer=FFMpegWriter(fps=15,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
 with writer.saving(fig,str(out/f'{name}.mp4'),dpi=100):
  for f in range(0,len(truth),2):
   for line,x,p,j in lines:
    xyz=x[f,[p,j]];line.set_data(xyz[:,0],xyz[:,1]);line.set_3d_properties(xyz[:,2])
   label.set_text(f'{name.title()} | Source frame {f+2}');writer.grab_frame()
   if f in [len(truth)//4*2,len(truth)//2*2-2]:fig.savefig(out/f'{name}_{f}.png')
 plt.close(fig)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/f'{name}.mp4'),'-filter_complex_threads','1','-filter_complex','fps=15,scale=1400:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse','-loop','0',str(out/f'{name}.gif')],check=True)
 print(name,flush=True)
