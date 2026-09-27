from __future__ import annotations

from pathlib import Path
import numpy as np

from .geometry import JOINTS

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def render_video(motion, output, title="", show_target=True):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter

    with np.load(motion) as data:
        fps = int(data["fps"])
        series = [("Observed actor", data["actor"], "#397bba", 1.),
                  ("Generated reactor", data["reactor"], "#ee7733", 1.)]
        if show_target:
            series.append(("Reference reactor", data["target"], "#888888", 0.35))
        series = [(name, x[:, JOINTS].reshape(-1, 22, 3), color, alpha)
                  for name, x, color, alpha in series]
    values = np.concatenate([x.reshape(-1, 3) for _, x, _, _ in series])
    low, high = values.min(0), values.max(0)
    center = (high + low) / 2
    radius = max(float((high - low).max()) / 2, 1.) + .2
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[1] - radius, center[1] + radius),
           zlim=(min(0., low[2] - .1), max(2., high[2] + .1)), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-65)
    floor = np.array([[center[0] - radius, center[0] + radius]] * 2)
    fy = np.array([[center[1] - radius] * 2, [center[1] + radius] * 2])
    ax.plot_surface(floor, fy, np.zeros((2, 2)), color="gray", alpha=.08)
    lines = []
    for name, joints, color, alpha in series:
        bones = []
        for j, parent in enumerate(PARENTS):
            if parent >= 0:
                line, = ax.plot([], [], [], color=color, alpha=alpha, linewidth=2,
                                label=name if j == 1 else None)
                bones.append((line, parent, j))
        lines.append((joints, bones))
    ax.legend(loc="upper right")
    label = ax.set_title(title[:100], fontsize=10)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(len(series[0][1])):
            for joints, bones in lines:
                for line, p, j in bones:
                    pts = joints[frame, [p, j]]
                    line.set_data(pts[:, 0], pts[:, 1])
                    line.set_3d_properties(pts[:, 2])
            label.set_text(title[:100] + "\nFrame " + str(frame))
            writer.grab_frame()
    plt.close(fig)
