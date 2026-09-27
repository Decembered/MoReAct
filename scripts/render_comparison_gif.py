"""Render paired generated/reference body skeletons from a rollout artifact."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moreact.geometry import JOINTS
from moreact.render import PARENTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--title',default='Reaction generation')
    args=parser.parse_args()
    root=Path(args.directory)
    meta=json.loads((root/'metadata.json').read_text())
    with np.load(root/'motion.npz') as data:
        actor=data['actor'][:,JOINTS].reshape(-1,22,3)
        pred=data['reactor'][:,JOINTS].reshape(-1,22,3)
        truth=data['target'][:,JOINTS].reshape(-1,22,3)
        fps=int(data['fps'])
    points=np.concatenate([x.reshape(-1,3) for x in (actor,pred,truth)])
    assert np.isfinite(points).all()
    low,high=points.min(0),points.max(0)
    center=(low+high)/2
    radius=max(float((high-low)[:2].max())/2+.25,1.)
    fig=plt.figure(figsize=(10,5),facecolor='white')
    lines=[]
    for panel,(reactor,title) in enumerate([(pred,'Generated reaction'),(truth,'Dataset reference')]):
        ax=fig.add_subplot(1,2,panel+1,projection='3d')
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
               zlim=(min(0.,low[2]-.1),max(2.,high[2]+.1)),title=title,xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect((2*radius,2*radius,max(2.,high[2]+.1)-min(0.,low[2]-.1)))
        ax.view_init(elev=18,azim=-65)
        for joints,color,label in [(actor,'#2475bd','Observed actor'),(reactor,'#ec7934','Reactor')]:
            for j,parent in enumerate(PARENTS):
                if parent>=0:
                    line,=ax.plot([],[],[],color=color,lw=2.5,label=label if j==1 else None)
                    lines.append((line,joints,parent,j))
        ax.legend(loc='upper right',fontsize=8)
    label=fig.suptitle('',fontsize=12)
    fig.text(.5,.02,'Same actor observations on both sides | seed 0 | 2-frame initialization; generated reactor feedback',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,bottom=.1,top=.86,wspace=.13)
    mp4=root/'comparison.mp4'
    writer=FFMpegWriter(fps=fps/2,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
    with writer.saving(fig,str(mp4),dpi=100):
        for frame in range(0,len(pred),2):
            for line,joints,p,j in lines:
                xyz=joints[frame,[p,j]]
                line.set_data(xyz[:,0],xyz[:,1]);line.set_3d_properties(xyz[:,2])
            label.set_text(f'{args.title} | checkpoint {meta["checkpoint_step"]:,} | {frame/fps:.2f} s')
            writer.grab_frame()
            if frame==len(pred)//4*2:
                fig.savefig(root/'comparison_still.png',dpi=100)
    plt.close(fig)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-threads','2','-i',str(mp4),
        '-filter_complex_threads','1','-filter_complex',
        'fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
        '-loop','0',str(root/'comparison.gif')],check=True)
    print(root/'comparison.gif')


if __name__=='__main__':
    main()
