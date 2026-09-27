"""Check stored GT delta targets against the history used at a drifted boundary."""
from diagnose import *
from moreact.geometry import DTRANS, DJOINTS

rows=[]
for item in json.loads((OUT/'protocol.json').read_text())['selection']:
    with np.load(CACHE/item['file']) as data:
        f=data['features'][1]
        for offset in (0., .3):
            shift=np.array([offset,0.,0.])
            gap=f[42,TRANSL]-(f[41,TRANSL]+shift)-f[42,DTRANS]
            jgap=f[42,JOINTS].reshape(22,3)-(f[41,JOINTS].reshape(22,3)+shift)-f[42,DJOINTS].reshape(22,3)
            rows.append(dict(episode=item['episode'],offset_m=offset,
                root_delta_gap_m=float(np.linalg.norm(gap)),joint_delta_gap_m=float(np.linalg.norm(jgap,axis=1).mean())))
            assert abs(np.linalg.norm(gap)-offset)<1e-6
save('target_consistency.json',rows)
print('PASS stored GT delta boundary consistency: clean ~0, translated history 0.30m gap')
