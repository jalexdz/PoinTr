import time
import numpy as np
import open3d as o3d
from PIL import Image

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1],
]))

pcd.colors = o3d.utility.Vector3dVector(np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.2, 0.2, 0.2],
]))

width, height = 640, 480
cam_center = np.array([0.0, 0.0, 0.0])
cam_pos = np.array([0.5, -0.5, 0.5])
cam_up = np.array([0.0, 0.0, 1.0])

vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height, visible=True)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
ctr.set_lookat(cam_center.tolist())
front = (cam_center - cam_pos)
front = front / (np.linalg.norm(front) + 1e-12)
ctr.set_front(front.tolist())
ctr.set_up(cam_up.tolist())

ctr.set_zoom(0.6)

opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
opt.point_size = 5.0

for _ in range(8):
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.03)


img = vis.capture_screen_float_buffer(do_render=True)
vis.destroy_window()

if img is None:
    print("CAPTURE FAILED")
else:
    img8 = (np.clip(np.asarray(img), 0, 1) * 255).astype('uint8')
    Image.fromarray(img8).save('test_vis_capture.pdf')
    print('Saved image')
