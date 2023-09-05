# -*- coding = utf-8 -*-
# @File Name : loss_new
# @Date : 2023/9/3 22:45
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn.functional as F
from PTF import propagate_frame


# --------- Utility Functions ---------
def preproc_output(output):
    """
    Pre-process the output of the network
    :param output: Output dictionary of the network
    :return: vessel: 2D image [B, H, W, 2], this is optimal directions
                     3D image [B, H, W, D, 3]. this is rotation for local frame; {pitch, yaw, roll}.
             radius: 2D image [B, H, W, 1], 3D image [B, H, W, D, 1].
    """
    vessel = output['vessel']                               # [B, 3, H, W, D] / [B, 2, H, W]
    radius = output['radius']                               # [B, 1, H, W, D] / [B, 1, H, W]
    k1 = output['k1']                                       # [B, 1, H, W, D] / [B, 1, H, W]
    k2 = output['k2'] if 'k2' in output.keys() else None    # [B, 1, H, W, D] / [B, 1, H, W]
    # if 3D image such as CT and MRA
    if vessel.dim() == 5:
        vessel = vessel.permute(0, 2, 3, 4, 1)              # change to [B, H, W, D, 3]
        radius = radius.permute(0, 2, 3, 4, 1)              # change to [B, H, W, D, 1]
    # if 2D image such as OCT
    else:
        vessel = vessel.permute(0, 2, 3, 1)                 # change to [B, H, W, 2]
        radius = radius.permute(0, 2, 3, 1)                 # change to [B, H, W, 1]
        k1 = k1.permute(0, 2, 3, 1)                         # change to [B, H, W, 1]
        vessel = F.normalize(vessel, dim=-1)                # normalize the optimal dir
    return vessel, radius, k1, k2


def get_local_reference_frame(vessel):
    """
    Get local reference frames
    :param vessel: 3D image [B, H, W, D, 3], {pitch, yaw, roll}.
                   2D image [B, H, W, 2], optimal direction
    :return: 3D image [B, H, W, D, n(3), 3] / 2D image [B, H, W, n(2), 2]
    """
    # if 3D image such as CT and MRA
    if vessel.dim() == 5:
        ''' code for calculate the rotated optimal direction and local frame '''
        basis = None
    # if 2D image such as OCT
    else:
        index = torch.LongTensor([1, 0]).to(vessel.device)
        ortho_dir_1 = torch.index_select(vessel, -1, index)
        ortho_dir_1[:, :, :, 1] = -ortho_dir_1[:, :, :, 1]
        ortho_dir_1 = ortho_dir_1 / ortho_dir_1.norm(dim=3, keepdim=True) + 1e-10
        basis = torch.stack((vessel, ortho_dir_1), dim=3)
    return basis


def get_sampling_vec(num_pts, estimated_r):
    """
    Get the sampling vectors, sphere or circle
    :param num_pts: sampling num points
    :param estimated_r: estimated radius, used to parse the device
    :return: sampling vectors
    """
    # if 3D image such as CT and MRA
    if estimated_r.dim() == 5:
        indices = torch.arange(0, num_pts, dtype=torch.float32)
        phi = torch.arccos(1 - 2 * indices / num_pts)
        theta = torch.pi * (1 + 5 ** 0.5) * indices
        x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
        # flip coordinates according to the sample grid
        vectors = torch.vstack((z, y, x)).T.to(estimated_r.device)      # This is a sphere sampling
    else:
        # if 2D image such as OCT
        angle = 2.0 * torch.pi * torch.arange(0, num_pts) / num_pts
        x, y = 1.0 * torch.cos(angle), 1.0 * torch.sin(angle)
        vectors = torch.vstack((y, x)).T.to(estimated_r.device)         # This is a circle sampling
    return vectors


def get_gradients(image):
    """
    Get gradients of batch image
    :param image: 3D image [B, 1, H, W, D] / 2D image [B, 1, H, W]
    :return: gradients: 3D image [B, 3, H, W, D] / 2D image [B, 2, H, W]
    """
    dims = (2, 3, 4) if image.dim() == 5 else (2, 3)
    gradients = torch.cat(torch.gradient(image, dim=dims), dim=1)
    gradients += torch.randn(gradients.size(), device=gradients.device) * 1e-10
    return gradients


def get_grid_base(image):
    """
    Get the image grid
    meshgrid method will switch x and z axis
    :param image: [B, 1, H, W] / [B, 1, H, W, D]
    :return: grid : [B, H, W, 2] / [B, H, W, D, 3]
    """
    b, c, h, w = image.shape[:4]
    d = image.shape[4] if image.dim() == 5 else None
    dh = torch.linspace(-1.0, 1.0, h)
    dw = torch.linspace(-1.0, 1.0, w)
    if d:
        dd = torch.linspace(-1.0, 1.0, d)
        meshx, meshy, meshz = torch.meshgrid((dh, dw, dd), indexing='ij')
        # need to swap the order of xyz
        grid = torch.stack((meshz, meshy, meshx), dim=3).repeat((b, 1, 1, 1, 1))    # [B, H, W, D, 3]
    else:
        meshx, meshy = torch.meshgrid((dh, dw), indexing='ij')
        # need to swap the order of xy
        grid = torch.stack((meshy, meshx), dim=2).repeat((b, 1, 1, 1))              # [B, H, W, 2]
    return grid.to(image.device)


def sample_space_to_img_space(grid):
    """
    Convert the image space to sample space
        [[-1, 1], [-1, 1], [-1, 1]] -> [[0, H], [0, W], [0, D]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [-1, 1] scale to [0, H] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    b, h, w, d, c = grid.shape
    grid = (grid + 0.0) * 0.5 + 0.5
    grid[..., 0] = grid[..., 0] * h
    grid[..., 1] = grid[..., 1] * w
    if d:
        grid[..., 2] = grid[..., 2] * d
    return grid


def img_space_to_sample_space(grid):
    """
    Convert the image space to sample space
    [[0, H], [0, W], [0, D]] -> [[-1, 1], [-1, 1], [-1, 1]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [0, H] scale to [-1, 1] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    grid = grid + 0.0
    b, h, w, d, c = grid.shape
    grid[..., 0] = 2.0 * grid[..., 0] / h - 1
    grid[..., 1] = 2.0 * grid[..., 1] / w - 1
    if d:
        grid[..., 2] = 2.0 * grid[..., 2] / d - 1
    return grid


def grid_sample(image, sample_grid, permute):
    """
    Functional grid sample overload
    :param image: [B, 1, H, W] / [B, 1, H, W, D]
    :param sample_grid: [B, H, W, 2] / [B, H, W, D, 3]
    :param permute: bool, indicate to change the shape or not
    :return:  [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    """
    sampled = F.grid_sample(image, sample_grid, align_corners=True, padding_mode='border')
    if image.dim() == 5:
        if permute:
            sampled = sampled.permute(0, 2, 3, 4, 1).unsqueeze(4)
    else:
        if permute:
            sampled = sampled.permute(0, 2, 3, 1).unsqueeze(3)
    return sampled


def get_sample_grid(curr_dir, curr_rad, grid_base):
    """
    Compute the sampling grid based on the radius and sample direction
    :param curr_dir: sampling direction 3D vector
    :param curr_rad: radius corresponding to this direction
    :param grid_base: grid base for sampling of the original image [B, H, W, D, 3]
    :return: the sampling grid, same size as grid base
    """
    b, h, w, d, c = grid_base.shape                                         # get the shape of the image
    # indices = torch.tensor([2, 1, 0], device=curr_dir.device)
    # curr_dir = torch.index_select(curr_dir, -1, indices)
    dir_scaled = torch.mul(curr_dir.repeat((b, h, w, d, 1)), curr_rad)      # scale the sample dir with radius
    sample_grid = img_space_to_sample_space(grid_base + dir_scaled)         # convert to [-1, 1]
    return sample_grid


def calc_dir_response(curr_dir, curr_rad, gradients, basis, grid_base):
    """
    Compute the projected response for given direction and radius
    :param curr_dir: sampling direction vector, 3d or 2d
    :param curr_rad: radius corresponding to this direction
    :param gradients: image gradients [B, 3, H, W, D] / [B, 2, H, W]
    :param basis: basis for vessel flow of each voxel [B, H, W, D, 3, 3] / [B, H, W, D, 2, 2]
    :param grid_base: grid base for sampling [B, H, W, D, 3】/ [B, H, W. 2]
    :return: projected responses [B, H, W, D, 3] / [B, H, W, 2]
    """
    """ Step 1. Get the sample grid according direction and radius """
    sample_grid = get_sample_grid(curr_dir, curr_rad, grid_base)                    # get sample grid [B, H, W, D, 3/2]
    """ Step 2. Grid sample the gradients based on the sample grid and project on the basis """
    sampled_gradients = grid_sample(gradients, sample_grid, permute=True)           # [B, H, W, D, 3/2, 3/2]
    """ Step 3. Project the sampled gradients on the basis """
    projected_gradients = torch.sum(torch.mul(sampled_gradients, basis), dim=-1)    # [B, H, W, D, 3/2]
    flux = torch.mul(projected_gradients.unsqueeze(-1), basis)                      # [B, H, W, D, 3/2, 3/2]
    """ Step 4. Project the flux on the sample direction  """
    flux = torch.sum(torch.mul(flux, curr_dir), dim=-1)                             # [B, H, W, D, 3/2]
    return flux


# --------- Loss Functions ---------
def flux_loss(image, output, sample_num):
    """
    Compute the None-Symmetry Flux Loss of the output of the network
    find the minimum magnitude of the direction and the opposite direction
    :param image: original image [B, 1, H, W, D] / [B, 1, H, W]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param sample_num: num of sampling directions of a sphere
    :return: flux response, mean flux loss
    """
    """ Step 1. Get the output vessel, radius and two principle curvatures """
    vessel, estimated_r, _, _ = preproc_output(output)                      # [B, H, W, D, 3], [B, H, W, D, N]
    """ Step 2. Compute the basis based on the vessel flow direction """
    basis = get_local_reference_frame(vessel)                               # [B, H, W, D, 3, 3]
    optimal_dir = basis[:, :, :, :, 1] if vessel.dim() == 5 else vessel
    """ Step 3. Get sampling sphere vectors """
    sample_vecs = get_sampling_vec(sample_num, estimated_r)                 # [sample_num, 3]
    """ Step 4. Compute image gradients """
    gradients = get_gradients(image)                                        # [B, 3, H, W, D]
    """ Step 5. Get the base sampling grid """
    grid_base = sample_space_to_img_space(get_grid_base(gradients))         # convert to [0, H], [B, H, W, D, 3]
    """ Step 6. Start the for loop to calculate the response """
    response = torch.zeros(optimal_dir.size(), device=optimal_dir.device)   # [B, H, W, D, 3]
    for i in range(int(sample_num / 2)):
        j = i + int(sample_num / 2)
        sample_dir1, sample_dir2 = sample_vecs[i], sample_vecs[j]           # get the sampling and opposite dir
        curr_rad1 = estimated_r[..., i:i+1] if estimated_r.size(-1) > 1 else estimated_r    # sampling radius of dir1
        curr_rad2 = estimated_r[..., j:j+1] if estimated_r.size(-1) > 1 else estimated_r    # opposite radius of dir2
        # calculate the directional response
        res1 = calc_dir_response(sample_dir1, curr_rad1, gradients, basis, grid_base)   # [B, H, W, D, 3]
        res2 = calc_dir_response(sample_dir2, curr_rad2, gradients, basis, grid_base)   # [B, H, W, D, 3]
        # find the min responses of the two dirs
        response += torch.maximum(res1, res2) * 2 / sample_num
    response = torch.clip(- torch.sum(response[..., 1:], dim=-1), min=0.0)  # clip the response by 0, [B, H, W, D]
    response = torch.mul(response, 1.0 + output['attention']) if 'attention' in output.keys() else response
    mean_flux_loss = - response.mean()                                      # use negative loss to maximize this
    return response, mean_flux_loss


def ptf_loss(image, output, flux_response, propagate_num, step_len):
    """
    Compute the continuity loss based on parallel transport frame
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention', 'curvature'
    :param flux_response: flux response [B, H, W] / [B, H, W, D]
    :param propagate_num: number of propagations of PTF
    :param step_len: step length of each propagation, float
    :return: mean direction_loss and mean intensity loss
    """
    """ Step 1. Get the output vessel, radius and two principle curvatures """
    vessel, _, k1, k2 = preproc_output(output)                              # [B, H, W, D, 3], [B, H, W, D, N]
    """ Step 2. Compute the basis based on the vessel flow direction """
    basis = get_local_reference_frame(vessel)                               # [B, H, W, D, 3, 3] / [B, H, W, D, 2, 2]
    optimal_dir = torch.select(basis, dim=-2, index=0)                      # [B, H, W, D, 3] / [B, H, W, D, 2]
    if k2:
        optimal_dir = optimal_dir.permute(0, 4, 1, 2, 3)
    else:
        optimal_dir = optimal_dir.permute(0, 3, 1, 2)
    """ Step 3. Construct the parrallel transport frames """
    img_coords = sample_space_to_img_space(get_grid_base(image))            # [B, H, W, D, 3] / [B, H, W, 2]
    curr_frame = torch.stack((img_coords.unsqueeze(-2), basis), dim=-2)     # [B, H, W, D, 4, 3] / [B, H, W, D, 3, 2]
    """ Step 4. Propagate the curves based on the image frame """
    direction_loss = 0.0
    intensity_loss = 0.0
    for i in range(propagate_num):
        curr_frame = propagate_frame(curr_frame, step_len, k1, k2)
        curr_coords = torch.select(curr_frame, dim=-2, index=0)
        curr_dirs = torch.select(curr_frame, dim=-2, index=1)
        # swap the orientation before offset the grid
        indices = [2, 1, 0] if k2 else [1, 0]
        indices = torch.tensor(indices, device=curr_coords.device)
        curr_grid = torch.index_select(curr_coords, -1, indices)
        # offset to get the sample grid based on the base grid
        sampled_response = grid_sample(flux_response, curr_grid, permute=False)
        sampled_direction = grid_sample(optimal_dir, curr_grid, permute=False)
        # intensity continuity loss
        intensity_loss += F.l1_loss(flux_response, sampled_response) / propagate_num
        # direction continuity loss
        direction_loss += F.cosine_similarity(curr_dirs, sampled_direction)
    return direction_loss, intensity_loss


def recon_loss(image, output, sup=False):
    """
    Compute the reconstruction loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: reconstructed image [B, 1, H, W] / [B, 1, H, W, D]
    :param sup: whether supervision is used
    :return: reconstruction loss
    """
    recon = output['recon']
    if sup:
        recon = torch.sigmoid(recon)
    rec_loss = F.mse_loss(image, recon)
    return rec_loss


def attention_loss(output, mean_val):
    """
    Compute the attention loss
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention', 'k1', 'k2'
    :param mean_val: mean attention value
    :return: attentional loss
    """
    att_loss = 0.0
    attentions = output['attentions']
    for i in range(len(attentions)):
        att_loss += torch.pow((attentions[i].mean() - mean_val), 2)
    return att_loss


def vessel_loss(image, output, loss_config):
    """
    Aggregate all the vessel loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param loss_config: dict, store the loss configurations
    :return: losses: dict, store the losses
    """
    """ Step 1. Calculate the flux loss """
    lambda_flux = loss_config['lambda_flux']
    flux_sample_num = loss_config['flux_sample_num']
    flux_response, mean_flux_loss = lambda_flux * flux_loss(image, output, flux_sample_num)
    """ Step 2. Calculate the continuity loss """
    lambda_direction = loss_config['lambda_direction']
    lambda_intensity = loss_config['lambda_intensity']
    step_len = loss_config['step_len']
    propagate_sample_num = loss_config['propagate_sample_num']
    direction_loss, intensity_loss = ptf_loss(image, output, flux_response, propagate_sample_num, step_len)
    direction_loss, intensity_loss = lambda_direction * direction_loss, lambda_intensity * intensity_loss
    """ Step 3. Calculate the reconstruction loss """
    lambda_recon = loss_config['lambda_recon']
    reconstruct_loss = lambda_recon * recon_loss(image, output)
    """ Step 4. Aggregate the total losses """
    total_loss = mean_flux_loss + reconstruct_loss + direction_loss + intensity_loss
    losses = {
        'flux_loss': mean_flux_loss,
        'dirs_loss': direction_loss,
        'ints_loss': intensity_loss,
        'rcon_loss': reconstruct_loss,
        'total_loss': total_loss
    }
    if 'attentions' in output.keys():
        mean_exposure = loss_config['mean_exp']
        if mean_exposure != 0:
            lambda_attention = loss_config['lambda_attention']
            mean_att_loss = lambda_attention * attention_loss(output, mean_exposure)
            total_loss += mean_att_loss
            losses['attn_loss'] = mean_att_loss
            losses['total_loss'] = total_loss
    return losses
