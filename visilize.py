import os
import numpy as np
import open3d as o3d
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cam_to_base_transform = [[ 6.3758686e-02 ,9.2318553e-01,-3.7902945e-01 ,4.5398907e+01],
 [ 9.8811066e-01,-5.1557920e-03 ,1.5365793e-01,-7.5876160e+02],
 [ 1.3990058e-01,-3.8432005e-01,-9.1253817e-01 ,9.6543054e+02],
 [ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00]]


color_map={"back_ground":[0,0,128],
           "cover":[0,100,0],
           "gear_container":[0,255,0],
           "charger":[255,255,0],
           "bottom":[255,165,0],
           "bolts":[255,0,0],
           "side_bolts":[255,0,255]}

def vis_PointCloud(sampled, corner_box=None):
    #get only the koordinate from sampled
    sampled = np.asarray(sampled)
    PointCloud_koordinate = sampled[:, 0:3]
    label=sampled[:,6]
    labels = np.asarray(label)
    print(labels.shape)
    max_label = label.max()
    cmap = ListedColormap(["navy", "darkgreen", "lime", "yellow", "orange", "magenta", "red"])
    colors = plt.get_cmap(cmap)(label / (max_label + 1))
    if corner_box is not None:    #visuell the point cloud and 3d bounding box
        lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                 [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        color = [[0, 1, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box.T)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(color)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([point_cloud, line_set])
    else:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([point_cloud])


def save_scene2img(patch_motor, corner_box=None, FileName=None):
    sampled = np.asarray(patch_motor)
    PointCloud_koordinate = sampled[:, 0:3]
    label=sampled[:,6]
    labels = np.asarray(label)
    print(labels.shape)
    max_label = label.max()
    cmap = ListedColormap(["navy", "darkgreen", "lime", "lavender", "yellow", "orange", "magenta", "red"])
    colors = plt.get_cmap(cmap)(label / (max_label + 1))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    vis.add_geometry(point_cloud)
    if corner_box is not None:
        lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                 [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        color = [[0, 1, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(color)
        vis.add_geometry(line_set)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(FileName)
    vis.destroy_window()


def save_cuboid2img(patch_motor, FileName=None):
    sampled = np.asarray(patch_motor)
    PointCloud_koordinate = sampled[:, 0:3]
    label=sampled[:,6]
    labels = np.asarray(label)
    print(labels.shape)
    max_label = labels.max()
    cmap = ListedColormap(["navy", "darkgreen", "lime", "lavender", "yellow", "orange","red"])
    # cmap = ListedColormap(["navy", "darkgreen", "lime", "lavender", "yellow", "orange", "magenta","red"])
    colors = plt.get_cmap(cmap)(labels / (max_label if max_label>0 else 1))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image(FileName)
    vis.destroy_window()

def save_cuboid2img_manmade(patch_motor, FileName=None):
    sampled = np.asarray(patch_motor)
    PointCloud_koordinate = sampled[:, 0:3]
    colors=[]
    for i in range(sampled.shape[0]):
        r=color_map["bolts"][0]
        g=color_map["bolts"][1]
        b=color_map["bolts"][2]
        colors.append([r,g,b])
    colors=np.array(colors)
    colors=colors/255
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image(FileName)
    vis.destroy_window()


def save_cuboid2img_original(patch_motor, FileName=None):
    sampled = np.asarray(patch_motor)
    PointCloud_koordinate = sampled[:, 0:3]
    colors=sampled[:, 3:6]
    colors=colors/255
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image(FileName)
    vis.destroy_window()




def read_cam_motor(csv_path):
    cam_pos = []
    motor_deflection = []
    with open(csv_path + '\\camera_motor_setting.csv', "r+") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            cam_pos.append(line[:6])
            motor_deflection.append(line[6:9])
    return cam_pos, motor_deflection



def rotation_matrix(alpha, beta, theta):
    M = np.array([[math.cos(theta)*math.cos(beta), -math.sin(theta)*math.cos(alpha)+math.cos(theta)*math.sin(beta)*math.sin(alpha),
                   math.sin(theta)*math.sin(alpha)+math.cos(theta)*math.sin(beta)*math.cos(alpha)],
                  [math.sin(theta)*math.cos(beta), math.cos(theta)*math.cos(alpha)+math.sin(theta)*math.sin(beta)*math.sin(alpha),
                   -math.cos(theta)*math.sin(alpha)+math.sin(theta)*math.sin(beta)*math.cos(alpha)],
                  [-math.sin(beta), math.cos(beta)*math.sin(alpha), math.cos(beta)*math.cos(alpha)]])
    return M


def deflect(alpha, beta, theta, points):
    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    M = rotation_matrix(alpha, beta, theta)
    points_motor = M.dot(points.T)
    return points_motor


def transfer_obj2cam(cam_pos_x, cam_pos_y, cam_pos_z, alpha, beta, theta, points):
    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    cam = (float(cam_pos_x), float(cam_pos_y), float(cam_pos_z))
    cam_pos = np.full((points.shape[1], 3), cam)
    points = points - cam_pos.T
    # M = rotation_matrix(alpha, beta, theta)
    c_mw = np.array([[math.cos(beta) * math.cos(theta), math.cos(beta) * math.sin(theta), -math.sin(beta)],
                     [-math.cos(alpha) * math.sin(theta) + math.sin(alpha) * math.sin(beta) * math.cos(theta),
                      math.cos(alpha) * math.cos(theta) + math.sin(alpha) * math.sin(beta) * math.sin(theta),
                      math.sin(alpha) * math.cos(beta)],
                     [math.sin(alpha) * math.sin(theta) + math.cos(alpha) * math.sin(beta) * math.cos(theta),
                      -math.sin(alpha) * math.cos(theta) + math.cos(alpha) * math.sin(beta) * math.sin(theta),
                      math.cos(alpha) * math.cos(beta)]])
    cor_new = c_mw.dot(points)
    return cor_new


def get_bbox(bbox_csv, cam_motor_csv, k):
    bbox = []
    with open(bbox_csv + '\\motor_3D_bounding_box.csv', "r+") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            bbox.append(line[1:10])
    x = float(bbox[k][0])
    y = float(bbox[k][1])
    z = float(bbox[k][2])
    h = float(bbox[k][3])
    w = float(bbox[k][4])
    l = float(bbox[k][5])
    cor_box = np.array([[x - l / 2, y - w / 2, z - h / 2], [x + l / 2, y - w / 2, z - h / 2],
                           [x - l / 2, y + w / 2, z - h / 2], [x + l / 2, y + w / 2, z - h / 2],
                           [x - l / 2, y - w / 2, z + h / 2], [x + l / 2, y - w / 2, z + h / 2],
                           [x - l / 2, y + w / 2, z + h / 2], [x + l / 2, y + w / 2, z + h / 2]])
    cam_info_all, motor_deflection_all = read_cam_motor(cam_motor_csv)
    cam_info = cam_info_all[k+1]
    motor_def = motor_deflection_all[k+1]
    deflected_motor = deflect(motor_def[0], motor_def[1], motor_def[2], cor_box)
    corner_box = transfer_obj2cam(cam_info[0], cam_info[1], cam_info[2], cam_info[3], cam_info[4], cam_info[5], deflected_motor)
    return corner_box

def camera_to_base(xyz, calc_angle=False):
    '''
    '''
        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]


    xyz_transformed2 = np.matmul(cam_to_base_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]

# s = 'E:\\test\TypeA1\Motor_0038\TypeA1_0038_cuboid.npy'
# t = np.load(s)
# vis_PointCloud(t)

def Read_PCD(file_path,FileName=None):

    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    points__=[]
    patch_motor= np.concatenate([points, colors], axis=-1)
    for i in range(patch_motor.shape[0]):
        if patch_motor[i][0]<1000 and patch_motor[i][2]>300:
            points__.append(patch_motor[i])
    sampled = np.asarray(points__)
    PointCloud_koordinate = sampled[:, 0:3]
    colors=sampled[:, 3:6]
    # colors=colors/255
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image(FileName)
    vis.destroy_window()



def main():

    # save_dir='/home/bi/study/thesis/data/current_finetune/A1/TrainingA1_1.npy'
    # if save_dir.split('.')[1]=='txt':
      
    #     patch_motor=np.loadtxt(save_dir)  
    # else:
    #     patch_motor=np.load(save_dir)   
    # print(len(patch_motor))
    # Visuell_PointCloud(patch_motor)
    # file_path= "/home/bi/study/thesis/pyqt/pcdfile/demo/A1_9.pcd"
    # file_path="/home/bi/study/thesis/pyqt/result"
    file_path="/home/bi/study/thesis/data/synthetic/finetune"
    save_path="/home/bi/Pictures/Pictures/hh.jpg"
    # Read_PCD(file_path,save_path)
    List_motor = os.listdir(file_path)
    if 'display.py' in List_motor :
        List_motor.remove('display.py')
    if '.DS_Store' in List_motor :
        List_motor.remove('.DS_Store')
    List_motor.sort()
    for dirs in List_motor :
        Motor_path = file_path + '/' + dirs
        # if "TypeA" in dirs:
        if True:
            if dirs.split('.')[1]=='txt':
            
                patch_motor=np.loadtxt(Motor_path)  
            else:
                patch_motor=np.load(Motor_path)  
            # if "Training" in dirs:     
                print(len(patch_motor))
                patch_motor[:,0:3]=camera_to_base(patch_motor[:,0:3])
                # save_cuboid2img(patch_motor,FileName=save_path)
                # save_cuboid2img_manmade(patch_motor,FileName=save_path)
                save_cuboid2img_original(patch_motor,FileName=save_path)
                # vis_PointCloud(patch_motor)
                hh=1



if __name__ == '__main__':
    main()