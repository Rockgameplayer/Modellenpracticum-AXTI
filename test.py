import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import time

# W = H = L = 768
WIDTH, HEIGHT = 20, 20
DIM = (1, 2, HEIGHT, WIDTH)  # format (N=batch_size, C=channel, H, W)

def generateIdentity():
  '''
  Generate a normalised identity flow field of dimension (N=1, DEPTH, HEIGHT, WIDTH, C=2)
  '''
  # permute into (N=1, depth, height, width, C=2)
  affine_identity = torch.eye(4)[:3].expand((1, 3, 4))
  return F.affine_grid(affine_identity, DIM, align_corners=True).permute(0,4,1,2,3)

def max_norm(field, exclude : int = 0):
  '''
  Calculate the maximum norm of a tensor of dimension (N, C, H, W).
  '''
  return torch.max(torch.linalg.vector_norm(field, dim=(1,))).item()  # calculate norm over dimension 1 = C

def plot_field(field, title='', **kwargs):
  fig, ax = plt.subplots()

  id = generateIdentity()
  plt.quiver(id[0][0], id[0][1], field[0][0], field[0][1], **kwargs)
  plt.title(title)

  # ax.invert_yaxis()  # does not work?
  plt.axis('square')
  plt.show()
  plt.savefig("output.png")

def timestepFD(flow_field):
  '''
  Calculate N so that max_norm(flow_field / 2**N) <= 0.5
  '''
  maxnorm = max_norm(flow_field)
  if maxnorm <= 0:
    return 0
  else:
    return math.ceil(max(math.log2(2*maxnorm), 1))

def fastVectorFieldExponential(flow_field, N=None):
  '''
  Calculate the exponential of a vector field of dimension (N, C, H, W).
  '''
  # calculate N
  # print('Max norm:', max_norm(flow_field))
  if N is None:
    N = timestepFD(flow_field)
  dt = 2**(-N)
  # print(f'{N=}, {dt=}')

  # normalize flow_field
  d, h, w = flow_field.size()[2:]
  flow_field_copy = torch.clone(flow_field)
  flow_field_copy[0, 0, ...] /= w/2
  flow_field_copy[0, 1, ...] /= h/2
  flow_field_copy[0, 2, ...] /= d/2

  identity = generateIdentity()
  # printu(u); print('-------------')
  v = identity + dt*flow_field_copy

  for _ in range(N):
    v_permuted = v.permute(0, 2, 3, 4, 1)  # grid_sample grid argument needs (N, D, H, W, C)
    v = F.grid_sample(v, v_permuted, align_corners=True, padding_mode='border')
  v -= identity

  # denormalize
  # v[0][0] *= w/2
  # v[0][1] *= h/2
  return v

angle = math.pi/4
affine_matrix = torch.tensor([[[math.cos(angle), -math.sin(angle), 0],[math.sin(angle), math.cos(angle), 0]]], dtype=torch.float).expand((1, 2, 3))
affine_rotation = F.affine_grid(affine_matrix, DIM, align_corners=True).permute(0,3,1,2)
u = (affine_rotation - generateIdentity()) * WIDTH/2


plot_field(u, title='Flow field')