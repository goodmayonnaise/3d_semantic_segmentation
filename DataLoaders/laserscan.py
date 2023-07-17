#!/usr/bin/env python3
import numpy as np
import cv2


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, front=False, project=False, sem_color_dict=None, H=256, W=1024, fov_up=3.0, fov_down=-25.0):
    self.front = front
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

    # make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)
    
    self.proj_remission_pad = np.full((self.proj_H, self.proj_W), -1,
                                       dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask
    
    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=float)              # [H,W,3] color


  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()
  
  def read_calib(self, calib_path):
      """
      :param calib_path: Path to a calibration text file.
      :return: dict with calibration matrices.
      """
      calib_all = {}
      with open(calib_path, 'r') as f:
          for line in f.readlines():
              if line == '\n':
                  break
              key, value = line.split(':', 1)
              calib_all[key] = np.array([float(x) for x in value.split()])

      # reshape matrices
      calib_out = {}
      calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
      calib_out['Tr'] = np.identity(4)  # 4x4 matrix
      calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

      return calib_out

  def select_points_in_frustum(self, points_2d, x1, y1, x2, y2):
      """
      Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
      :param points_2d: point cloud projected into 2D
      :param points_3d: point cloud
      :param x1: left bound
      :param y1: upper bound
      :param x2: right bound
      :param y2: lower bound
      :return: points (2D and 3D) that are in the frustum
      """
      keep_ind = (points_2d[:, 0] > x1) * \
                  (points_2d[:, 1] > y1) * \
                  (points_2d[:, 0] < x2) * \
                  (points_2d[:, 1] < y2)

      return keep_ind

  def open_scan(self, filename, calib=None):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    if calib != None:
      calib = self.read_calib(calib)
      proj_matrix = np.matmul(calib["P2"], calib["Tr"])
      self.proj_matrix = proj_matrix

    return self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      return self.do_range_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    
    if self.front:
      keep_idx = self.points[:,0] > 0
      self.points = self.points[keep_idx] # x y z
      # points_hcoords = np.concatenate([self.points, np.ones([keep_idx.sum(),1], dtype=np.float32)], axis=1) # x y z 1
      # img_points =(self.proj_matrix @ points_hcoords.T).T
      # img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
      # keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, self.proj_W, self.proj_H) # size down 
      # img_points = np.fliplr(img_points)

      # points_img = img_points[keep_idx_img_pts] # proj x, proj y 
      # proj_y, proj_x = points_img[:,0], points_img[:,1]
      # depth = np.linalg.norm(self.points, 2, axis=1) # 12466(N)
      # depth = depth[keep_idx_img_pts]

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1) # 12466(N)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / (depth + 1e-8))

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)                  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    # proj_x *= self.proj_W                              # in [0.0, W]
    # proj_y *= self.proj_H                              # in [0.0, H]
    
    proj_x = ((proj_x - proj_x.min())/(proj_x.max()-proj_x.min()))*self.proj_W
    proj_y = ((proj_y - proj_y.min())/(proj_y.max()-proj_y.min()))*self.proj_H
    
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # padding point 

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.float32)
      
    self.pad_rem(remission, proj_y, proj_x)
    if self.front:
      return {'range' : self.proj_range, # 256 1024
              'xyz' : self.proj_xyz, # 256 1024 3 (x, y, z)
              'remission' : self.proj_remission, # 256 1024
              'idx': self.proj_idx, # 256 1024
              'mask' : self.proj_mask, # 256 1024
              'pad_remission':self.proj_remission_pad
              }, keep_idx
    else:
      return {'range' : self.proj_range, # 256 1024
              'xyz' : self.proj_xyz, # 256 1024 3 (x, y, z)
              'remission' : self.proj_remission, # 256 1024
              'idx': self.proj_idx, # 256 1024
              'mask' : self.proj_mask, # 256 1024
              'pad_remission':self.proj_remission_pad}
    
  def pad_rem(self, remission, proj_y, proj_x):
      proj_x_m, proj_y_m = np.maximum(proj_x-1, 0), np.maximum(proj_y-1, 0)
      proj_x_p, proj_y_p = np.minimum(proj_x+1, self.proj_W-1), np.minimum(proj_y+1, self.proj_H-1)
      self.proj_remission_pad[proj_y_m, proj_x_m] = remission
      self.proj_remission_pad[proj_y, proj_x_m] = remission
      self.proj_remission_pad[proj_y_p, proj_x_m] = remission
      self.proj_remission_pad[proj_y_m, proj_x] = remission
      self.proj_remission_pad[proj_y, proj_x] = remission
      self.proj_remission_pad[proj_y_p, proj_x] = remission
      self.proj_remission_pad[proj_y_m, proj_x_p] = remission
      self.proj_remission_pad[proj_y, proj_x_p] = remission
      self.proj_remission_pad[proj_y_p, proj_x_p] = remission

  def open_label(self, filename, keep_idx=None):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    if self.front:
      label = label[keep_idx]
    label = label.reshape((-1))

    # set it
    return self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")
    # self.sem_label = label & 0xFFFF  # semantic label in lower half
    # self.inst_label = label >> 16    # instance id in upper half

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      return self.do_label_projection()

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
    import cv2
    return {'label' : self.proj_sem_label, # 256 1024
            'color_label' : self.proj_sem_color, # 256 1024 3
            'inst_label': self.inst_label, # 256*1024
            'inst_color_label' : self.proj_inst_color} # 256 1024 3

  def set_data(self, x, y, calib=None):
    if self.front:
      x, keep_idx = self.open_scan(x, calib)
      y =self.open_label(y, keep_idx)
    else:
      x = self.open_scan(x)
      y = self.open_label(y)

    return x, y
  
if __name__ == "__main__":

  def load_config():
      # cfg_path = './data/semantic-kitti.yaml'
      cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
      try:
          print("Opening config file %s" % {cfg_path})
          import yaml
          CFG = yaml.safe_load(open(cfg_path, 'r'))
      except Exception as e:
          print(e)
          print("Error opening yaml file.")
          quit()
      return CFG
    
  x = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/velodyne/000000.bin'
  y = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/labels/000000.label'
  calib = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/calib.txt'
  CFG = load_config()
  sem_color_dict = CFG['color_map']
  scan = LaserScan(project=True, sem_color_dict=CFG['color_map'], front=False)
  scan.set_data(x, y)
  # scan.set_data(x, y, calib)
  # scan = SemLaserScan(sem_color_dict, project=True)
  # scan.open_label(y)
