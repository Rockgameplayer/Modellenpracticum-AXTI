import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import time
print('cuda available:', torch.cuda.is_available())
torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# W = H = D = 768
WIDTH, HEIGHT, DEPTH = 20, 20, 3
# DIM = (1, 2, HEIGHT, WIDTH)  # format (N=batch_size, C=channel, H, W)
DIM = (1, 2, DEPTH, HEIGHT, WIDTH) # format (N=batch_size, C=channel, D, H, W)

def generateIdentity():
    '''
    Generate a normalised identity flow field of dimension (N=1, DEPTH, HEIGHT, WIDTH, C=2)
    '''
    # permute into (N=1, depth, height, width, C=2)
    affine_identity = torch.eye(4)[:3].expand((1, 3, 4))
    return F.affine_grid(affine_identity, DIM, align_corners=True).permute(0,4,1,2,3)


def plot_field(field, title='', **kwargs):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(projection='3d') # Correctly assign the 3D axes

    id = generateIdentity()
    # Corrected: use id[0][0] for X-coords, id[0][1] for Y-coords, id[0][2] for Z-coords
    # and field[0][0] for X-component, field[0][1] for Y-component, field[0][2] for Z-component
    ax.quiver(id[0][0], id[0][1], id[0][2], field[0][0], field[0][1], field[0][2], **kwargs)
    ax.set_title(title)

    # Add axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # If depth is 1, manually adjust z-limits to prevent squishing
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1]) # Make axes have equal ratio

    plt.savefig('testplot.png')

def plot_field(field, title='', **kwargs):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(projection='3d') # Correctly assign the 3D axes

    id = generateIdentity()
    # Corrected: use id[0][0] for X-coords, id[0][1] for Y-coords, id[0][2] for Z-coords
    # and field[0][0] for X-component, field[0][1] for Y-component, field[0][2] for Z-component
    ax.quiver(id[0][0], id[0][1], id[0][2], field[0][0], field[0][1], field[0][2], **kwargs)
    ax.set_title(title)

    # Add axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # If depth is 1, manually adjust z-limits to prevent squishing
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1]) # Make axes have equal ratio

    plt.show()

def max_norm(field, exclude : int = 0):
    '''
    Calculate the maximum norm of a tensor of dimension (N, C, H, W).
    '''
    return torch.max(torch.linalg.vector_norm(field, dim=(1,))).item()  # calculate norm over dimension 1 = C

def remove_border(field, border_percentage):
    '''
    Zero out the border of a field for a given percentage of the size.
    '''
    d, h, w = field.size()[2:]
    bd, bh, bw = int(d*border_percentage), int(h*border_percentage), int(w*border_percentage)

    ret = torch.zeros(field.size())

    ret[0, :, bd:d-bd, bh:h-bh, bw:w-bw] = field[0, :, bd:d-bd, bh:h-bh, bw:w-bw]

    return ret

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

def fastVectorFieldExponential_new(flow_field, N=None):
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
    v = dt*flow_field_copy

    for i in range(N):
        v_permuted = (identity + v).permute(0, 2, 3, 4, 1)  # grid_sample grid argument needs (N, D, H, W, C)
        v += F.grid_sample(v, v_permuted, align_corners=True, padding_mode='border')

    # denormalize
    # v[0][0] *= w/2
    # v[0][1] *= h/2
    return v

def normalVectorFieldExponential_new(flow_field, N=None):
    '''
    Calculate the exponential of a vector field.
    '''
    if N is None:
        N = timestepFD(flow_field)
    dt = 2**(-N)

    # normalize u
    d, h, w = flow_field.size()[2:]
    flow_field_copy = torch.clone(flow_field)
    flow_field_copy[0, 0, ...] /= w/2
    flow_field_copy[0, 1, ...] /= h/2
    flow_field_copy[0, 2, ...] /= d/2

    identity = generateIdentity()
    v = dt*flow_field_copy
    s = v

    for _ in range(2**N):
        s = F.grid_sample(v, (s + identity).permute(0, 2, 3, 4, 1), align_corners=True, padding_mode='border') + s

    # denormalize
    # v[0][0] *= w/2
    # v[0][1] *= h/2
    return s

def normalVectorFieldExponential(flow_field, N=None):
    '''
    Calculate the exponential of a vector field.
    '''
    if N is None:
        N = timestepFD(flow_field)
        # print(N)
    dt = 2**(-N)

    # normalize u
    d, h, w = u.size()[2:]
    flow_field_copy = torch.clone(flow_field)
    flow_field_copy[0, 0, ...] /= w/2
    flow_field_copy[0, 1, ...] /= h/2
    flow_field_copy[0, 2, ...] /= d/2

    identity = generateIdentity()

    v = identity + dt*flow_field_copy
    v_start = v.permute(0, 2, 3, 4, 1)  # grid_sample grid argument needs (N, D, H, W, C)

    for _ in range(2**N):
        v = F.grid_sample(v, v_start, align_corners=True, padding_mode='border')
    v -= identity

    # denormalize
    # v[0][0] *= w/2
    # v[0][1] *= h/2
    return v

def compareForwardBackward(flow_field, exp_function, N=None):
    start = time.time()

    # print()
    forward = exp_function(flow_field, N)
    backward = exp_function(-flow_field, N)

    id = generateIdentity()
    #plot_field(forward, title=exp_function.__name__, scale_units='xy', scale=1)

    residual = forward - F.grid_sample(backward, (forward+id).permute(0,2,3,4,1), align_corners=True, padding_mode='border')

    elapsed = (time.time() - start)*1000
    # print(f'Elapsed time {exp_function.__name__} for {N=}: {elapsed}ms')
    # plot_field(residual)
    max_error = max_norm(remove_border(residual, 0.15))
    plot_field(remove_border(residual, 0.15), title=exp_function.__name__, length=0.1)
    return max_error, elapsed

def plot_error(flow_field, exp_function):
    fig, ax = plt.subplots()

    N_default = timestepFD(flow_field)
    Ns = list(range(1, (2*N_default if exp_function.__name__.startswith('normal') else 30*N_default)))
    errs, times = zip(*(compareForwardBackward(flow_field, exp_function, N) for N in Ns))

    ax.set_xlabel('N')
    ax.set_ylabel('error (px)', color='tab:blue')
    ax.plot(Ns, errs, color='tab:blue', marker='o')
    ax.plot(N_default, compareForwardBackward(flow_field, exp_function, N_default)[0], 'yo')
    # ax.ylim(bottom=0, top=max(errs)*1.4)
    ax.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel('time (ms)', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(Ns, times, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax.set_title(exp_function.__name__)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# u = torch.ones(dim, dtype = torch.float)

## 1/2 rescaling
# u = -generateIdentity() / 2 * WIDTH/2

### rotate by angle
x_angle = math.pi/4
y_angle = 0
z_angle = math.pi/4
z_rot = torch.tensor([[math.cos(z_angle), -math.sin(z_angle), 0],[math.sin(z_angle), math.cos(z_angle), 0],[0,0,1]])
y_rot = torch.tensor([[math.cos(y_angle), 0, math.sin(y_angle)],[0,1,0],[-math.sin(y_angle), 0, math.cos(y_angle)]])
x_rot = torch.tensor([[1,0,0],[0,math.cos(x_angle), -math.sin(x_angle)],[0,math.sin(x_angle),math.cos(x_angle)]])
rot_matrix = torch.matmul(z_rot,torch.matmul(y_rot,x_rot))

affine_rotation = F.affine_grid(torch.column_stack((rot_matrix, torch.tensor([0, 0, 0]))).expand(1,3,4), DIM, align_corners=True).permute(0,4,1,2,3)
u = (affine_rotation - generateIdentity()) * WIDTH/2

# print_tensor(u)
# plot_field(u, title='Flow field')

# plot_field(fastVectorFieldExponential_new(u), title='fastVectorFieldExponential_new', scale_units='xy', scale=1)

# a = animate_vector_field_exponential2(u)

# from matplotlib import rc
# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# from IPython.display import HTML
# HTML(a.to_html5_video())
# a


# v = fastVectorFieldExponential_new(u)
# plot_field(fastVectorFieldExponential_new(u), scale_units='xy', scale=1, title='Displacement field (forward)')
# plot_field(fastVectorFieldExponential_new(-u), scale_units='xy', scale=1, title='Displacement field (backward)')
# plot_field(remove_border(u, 0.15))

fns = (fastVectorFieldExponential, fastVectorFieldExponential_new, normalVectorFieldExponential, normalVectorFieldExponential_new)
for fn in fns:
    error, _ = compareForwardBackward(u, fn)
    # plot_error(u, fn)
    pass
    print(error)