import torch
import torch.nn as nn
import math


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])


def hash(coords, log2_hashmap_size):
    """
    This function takes in a tensor of coordinates and returns a tensor of hash values.
    The hash values are computed using the following formula:
    hash(x) = x[0]*p[0] ^ x[1]*p[1] ^ x[2]*p[2] ^ x[3]*p[3] ^ x[4]*p[4] ^ x[5]*p[5] ^ x[6]*p[6]
    where p[i] is a prime number.
    The hash values are computed using the above formula for each coordinate in the input tensor.
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1,2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.remainder(xor_result,2**log2_hashmap_size)


def get_voxel_vertices(xyz, bounding_box, resolution, device = None):
    """
    Returns voxel vertices

    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    """
    box_min, box_max = bounding_box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)
    grid_size = (box_max-box_min)/resolution

    bottom_left_idx = torch.floor((xyz-box_min)/grid_size)
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    weights = (xyz - voxel_min_vertex)/grid_size # B x 3
    if weights.max().item()>1.1:
        print('first pass',weights.max())

    voxel_indices = bottom_left_idx.unsqueeze(1).long() + BOX_OFFSETS.to(device)

    hashed_voxel_indices=voxel_indices[...,0].clone()+voxel_indices[...,1].clone()*resolution+voxel_indices[...,2].clone()*(resolution**2)
        
    return weights, hashed_voxel_indices


def get_hash_table(xyz,bounding_box, resolution, device= None):
    """
    This function takes in a set of 3D points and a bounding box and returns a hash table of the voxels that are occupied
    by the points.

    :param xyz: 3D points of the object's mesh
    :param bounding_box: bounding borders of the box containing the grid
    :param resolution: voxel grid resolution
    :param device: device to move the hash table to
    :return: hash table
    """
    box_min, box_max = bounding_box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    hashed_voxel_indices=bottom_left_idx[:,0]+bottom_left_idx[:,1]*resolution+bottom_left_idx[:,2]*(resolution**2)    
    hash_table=torch.unique(hashed_voxel_indices)
    HASH_OFFSETS =torch.tensor([[-1,1,-resolution,resolution,-resolution**2,resolution**2]],device=device)
    for j in range(2):   
        hash_table = hash_table.unsqueeze(1) + HASH_OFFSETS
        hash_table=hash_table.view(-1)

    hash_table=torch.unique(hash_table)
    return hash_table


class HashEmbedder(nn.Module):
    """
    This class implements a neural hash embedder.
    The "forward" function takes a point cloud as input and outputs a feature vector for each point.
    The feature vector is computed by trilinear interpolation of the embedding vectors of the voxels that the point falls into.
    The embedding vectors are learned by a neural network during training.
    The voxels are defined by a hash table.
    The hash table (list of voxels) is defined by a bounding box, a base resolution and a finest resolution.
    Each voxel is defined by a center (3D point) and a resolution.
    """
    def __init__(self, bounding_box=[-1.0,1.0], n_levels=16, n_features_per_level=2,
                log2_hashmap_size=19, base_resolution=16., finest_resolution=512.,sparse=True,vertices=None, device = None):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        self.embedding_layers=[]
        self.hash_tables=[]
        for i in range(n_levels):
            resolution= math.floor(self.base_resolution * self.b**i)
            hash_table=get_hash_table(vertices,self.bounding_box, resolution, device = device)

            embedding_layer=nn.Embedding(hash_table.shape[0], self.n_features_per_level,sparse=sparse)
            self.embedding_layers.append(embedding_layer)
            self.hash_tables.append(hash_table)

        self.embeddings = nn.ModuleList(self.embedding_layers)
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-1e-4, b=1e-4)

    def trilinear_interp(self,weights, voxel_embedds):
        if weights.max().item()>1.1:
            print('second pass',weights.max())

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        shape=list(x.shape)
        shape[-1]=self.out_dim

        x=x.view(-1,3)
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = math.floor(self.base_resolution * self.b**i)
            weights, hashed_voxel_indices = get_voxel_vertices(
                                                x, self.bounding_box,
                                                resolution, device= x.device)
            hashed_voxel_indices_=torch.searchsorted(self.hash_tables[i], hashed_voxel_indices)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices_)

            x_embedded = self.trilinear_interp(weights, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1).view(shape)


class SHEncoder(nn.Module):
    """
    This class implements a spherical harmonics embedder.
    """
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result
