import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import itertools

print('cuda available:', torch.cuda.is_available())
torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# W = H = D = 768
WIDTH, HEIGHT, DEPTH = 20, 20, 20
# DIM = (1, 2, HEIGHT, WIDTH)  # format (N=batch_size, C=channel, H, W)
DIM = (1, 2, DEPTH, HEIGHT, WIDTH) # format (N=batch_size, C=channel, D, H, W)

def generateIdentity():
    '''
    Generate a normalised identity flow field of dimension (N=1, DEPTH, HEIGHT, WIDTH, C=2)
    '''
    # permute into (N=1, depth, height, width, C=2)
    affine_identity = torch.eye(4)[:3].expand((1, 3, 4))
    return F.affine_grid(affine_identity, DIM, align_corners=True).permute(0,4,1,2,3)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_field(field, title='', cmap='viridis', normalize=True, **kwargs):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(projection='3d')

    id = generateIdentity().cpu()
    field_cpu = field.cpu()

    X = id[0][0].numpy()
    Y = id[0][1].numpy()
    Z = id[0][2].numpy()

    U = field_cpu[0][0].numpy()
    V = field_cpu[0][1].numpy()
    W = field_cpu[0][2].numpy()

    # Compute magnitude
    magnitude = np.sqrt(U**2 + V**2 + W**2)

    # Normalize (optional but recommended)
    if normalize:
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    else:
        mag_norm = magnitude

    # Map to colors
    colors = cm.get_cmap(cmap)(mag_norm)

    ax.quiver(
        X, Y, Z,
        U, V, W,
        colors=colors.reshape(-1, 4),  # flatten to match quiver input
        **kwargs
    )

    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1])

    # Optional colorbar
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(magnitude)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Magnitude')

    plt.savefig(f'img_{title}.png')

def max_norm(field, exclude : int = 0):
    '''
    Calculate the maximum norm of a tensor of shape (N, C, D, H, W).

    Parameters:
    field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension

    Returns:
    float: the maximum norm over all vectors on the HxW grid, the C dimension
    '''
    return torch.max(torch.linalg.vector_norm(field, dim=(1,))).item()  # calculate norm over dimension 1 = C

def remove_border(field, border_percentage):
    '''
    Zero out the border of a field for a given percentage of the size.

    Parameters:
    field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    border_percentage (float): percentage of D, H, W to zero out from the border

    Returns:
    tensor: the modified tensor with a border of zeroes
    '''
    d, h, w = field.size()[2:]
    bd, bh, bw = int(d*border_percentage), int(h*border_percentage), int(w*border_percentage)

    ret = torch.zeros(field.size())

    ret[0, :, bd:d-bd, bh:h-bh, bw:w-bw] = field[0, :, bd:d-bd, bh:h-bh, bw:w-bw]

    return ret

def timestepFD(flow_field):
    '''
    Calculates N so that the maximum norm of the vectors of the flow field /2**N is less than a half

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension

    Returns:
    float: the integer N such that max_norm(flow_field / 2**N) <= 0.5
    '''
    maxnorm = max_norm(flow_field)
    if maxnorm <= 0:
        return 0
    else:
        return math.ceil(max(math.log2(2*maxnorm), 1))

def fastVectorFieldExponential(flow_field, N=None):
    '''
    Calculate the exponential of a vector field with the fast vector field exponential algorithm.

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    N (int): the amount of steps for the algorithm to take

    Returns:
    tensor: of shape (N, C, D, H, W) containing the exponential of the flow field, so the the displacement field
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
    Calculate the exponential of a vector field with the fast vector field exponential algorithm.

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    N (int): the amount of steps for the algorithm to take

    Returns:
    tensor: of shape (N, C, D, H, W) containing the exponential of the flow field, so the the displacement field
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
    Calculate the exponential of a vector field in a linearly integrated way.

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    N (int): the amount of steps for the algorithm to take

    Returns:
    tensor: of shape (N, C, D, H, W) containing the exponential of the flow field, so the the displacement field
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
    Calculate the exponential of a vector field in a linearly integrated way.

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    N (int): the amount of steps for the algorithm to take

    Returns:
    tensor: of shape (N, C, D, H, W) containing the exponential of the flow field, so the the displacement field
    '''
    if N is None:
        N = timestepFD(flow_field)
        # print(N)
    dt = 2**(-N)

    # normalize u
    d, h, w = flow_field.size()[2:]
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
    '''
    Calculates the residual and elapsed time of the given algorithm

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    exp_function (func): the exponential algorithm to test
    N: the amount of steps the algorithm should take

    Returns:
    float: the maximum residual pixel error when composing forward and backward displacement field
    float: the elapsed time it took for the algorithm to complete
    '''

    start = time.time()

    # print()
    forward = exp_function(flow_field, N)
    backward = exp_function(-flow_field, N)

    id = generateIdentity()
    #plot_field(forward, title=exp_function.__name__, scale_units='xy', scale=1)

    residual = forward + F.grid_sample(backward, (forward+id).permute(0,2,3,4,1), align_corners=True, padding_mode='border')

    # normalize u
    d, h, w = flow_field.size()[2:]
    residual_copy = torch.clone(residual)
    residual_copy[0, 0, ...] *= w/2
    residual_copy[0, 1, ...] *= h/2
    residual_copy[0, 2, ...] *= d/2

    elapsed = (time.time() - start)*1000
    # print(f'Elapsed time {exp_function.__name__} for {N=}: {elapsed}ms')
    # plot_field(residual, title=exp_function.__name__, length=0.1)
    max_error = max_norm(remove_border(residual_copy, 0.05))
    # plot_field(remove_border(residual, 0.15), title=exp_function.__name__, length=50)
    return max_error, elapsed

def plot_error(flow_field, exp_function):
    '''
    Plots the residual pixel error and elapsed time for different values of N for the given algorithm

    Parameters:
    flow_field (tensor): tensor of shape (N, C, D, H, W) containing flow field on C dimension
    exp_function (func): the exponential algorithm to test

    Returns:
    Null
    '''
    fig, ax = plt.subplots()
    
    N_default = timestepFD(flow_field)
    Ns = list(range(1, (int(1.5*N_default) if exp_function.__name__.startswith('normal') else 2*N_default)))
    errs, times = zip(*(compareForwardBackward(flow_field, exp_function, N) for N in Ns))

    ax.set_xlabel('N')
    ax.set_ylabel('error (px)', color='tab:blue')
    ax.set_yscale('log')
    ax.plot(Ns, errs, color='tab:blue', marker='o')
    ax.plot(N_default, compareForwardBackward(flow_field, exp_function, N_default)[0], 'yo')
    ax.axhline(y=0.5, color='g', linestyle='--')

    # ax.ylim(bottom=0, top=max(errs)*1.4)
    ax.tick_params(axis='y', labelcolor='tab:blue')

    
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    if exp_function.__name__.startswith('normal'):
        ax2.set_yscale('log')
    ax2.set_ylabel('time (ms)', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(Ns, times, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax.set_title(exp_function.__name__)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(f'img_{exp_function.__name__}.png')

def compareFunctions(flow_field, f,g, N = None):
    border_percentage = 0.15
    forward_f = remove_border(f(flow_field, N),border_percentage)
    forward_g = remove_border(g(flow_field, N),border_percentage)
    print(f"error: {max_norm(forward_f - forward_g):.6f}")

def main():

    a, b, c = 0.1, 0, 0  # in normalized coordinates!
    
    trans_matrix = torch.eye(3)
    translation = torch.tensor([a, b, c])

    affine = torch.column_stack((trans_matrix, translation))

    affine_grid = F.affine_grid(
        affine.expand(1, 3, 4),
        DIM,
        align_corners=True
    ).permute(0, 4, 1, 2, 3)

    # u = (affine_grid - generateIdentity()) * WIDTH/2

    # u = torch.ones(dim, dtype = torch.float)

    ## 1/2 rescaling
    u = -generateIdentity() / 2 * WIDTH/2

    ### rotate by angle
    x_angle = 0 # math.pi/4
    y_angle = 0
    z_angle = math.pi/4
    z_rot = torch.tensor([[math.cos(z_angle), -math.sin(z_angle), 0],[math.sin(z_angle), math.cos(z_angle), 0],[0,0,1]])
    y_rot = torch.tensor([[math.cos(y_angle), 0, math.sin(y_angle)],[0,1,0],[-math.sin(y_angle), 0, math.cos(y_angle)]])
    x_rot = torch.tensor([[1,0,0],[0,math.cos(x_angle), -math.sin(x_angle)],[0,math.sin(x_angle),math.cos(x_angle)]])
    rot_matrix = torch.matmul(z_rot,torch.matmul(y_rot,x_rot))

    affine_rotation = F.affine_grid(torch.column_stack((rot_matrix, torch.tensor([0, 0, 0]))).expand(1,3,4), DIM, align_corners=True).permute(0,4,1,2,3)
    # u = (affine_rotation - generateIdentity()) * WIDTH/2

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
    # for f,g in itertools.combinations(fns, r = 2):
    #     print(f.__name__, g.__name__)
    #     compareFunctions(u,f,g)

    for fn in fns:
        error, elapsed = compareForwardBackward(u, fn)
        plot_error(u, fn)
        pass
        print(f'{fn.__name__}:\t{error=:.3f}\t{elapsed=:.3f}ms')

if __name__ == '__main__':
    main()
