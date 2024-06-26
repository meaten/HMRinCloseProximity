"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""

import torch
import numpy as np
import scipy
from typing import Optional, Dict, List, Tuple
from scipy.io import loadmat

from torch.distributions.categorical import Categorical

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance

from ProHMR.prohmr.utils.pose_utils import Evaluator


class Timer(object):
    def __init__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        
    def start(self):
        torch.cuda.synchronize()
        self.starter.record()
    
    def end(self):
        self.ender.record()
        
    def elapsed_time(self):
        torch.cuda.synchronize()
        return self.starter.elapsed_time(self.ender)

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)


def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt(((S1_hat - S2) ** 2).sum(dim=-1))
    re_mean = re.mean(dim=-1)
    return re_mean.cpu().numpy(), re.cpu().numpy()


def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1))
    mpjpe_mean = mpjpe.mean(dim=-1)
    # Reconstuction_error
    r_error_mean, r_error = reconstruction_error(
        pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe_mean.cpu().numpy(), 1000 * r_error_mean, 1000 * mpjpe.cpu().numpy(), 1000 * r_error


def rel_change(prev_val: float, curr_val: float) -> float:
    """
    Compute relative change. Code from https://github.com/vchoutas/smplify-x
    Args:
        prev_val (float): Previous value
        curr_val (float): Current value
    Returns:
        float: Relative change
    """
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def load_downsample_mat(path):
    smpl_mesh_graph = np.load(path, allow_pickle=True, encoding='latin1')

    A = smpl_mesh_graph['A']
    U = smpl_mesh_graph['U']
    D = smpl_mesh_graph['D'] # shape: (2,)
    F = smpl_mesh_graph['F'] # shape: (3,)

    ptD = []
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

    Vmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431
    new_faces = F[2]
    return Vmap, new_faces


def unsort_unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


class CustomEvaluator(Evaluator):
    def __init__(self,
                 dataset_length: int,
                 keypoint_list: List,
                 pelvis_ind: int,
                 smpl,
                 metrics: List = ['mode_mpjpe', 'mode_re', 'min_mpjpe', 'min_re'],
                 n_measure_points=30):
        super().__init__(dataset_length, keypoint_list, pelvis_ind, metrics)
        
        Vmap, faces = load_downsample_mat('./data/datasets/mesh_downsampling.npz')
        self.Vmap = torch.where(Vmap.cuda() > 0)[1]
        
        self.faces_full = smpl.faces
        self.faces = torch.IntTensor(np.array(faces, dtype=int)).cuda()

        self.n_measure_points = n_measure_points
        
        self.timer = Timer()
        
    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            if not 'time' in metric:
                print(f'{metric}: {getattr(self, metric)[:self.counter].mean()} mm')
            else:
                print(f'{metric}: {getattr(self, metric)[:self.counter].mean()} ms \
                      ({np.sqrt(getattr(self, metric)[:self.counter].var())} ms)')
        print('***')

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None, flow_net=None, smpl=None):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """
        pred_keypoints_3d = output['pred_keypoints_3d'].detach()
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :,
                                                :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
        
        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]

        # Compute joint errors
        mpjpe, re, mpjpe_all, re_all = eval_pose(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[
                                                 :, self.keypoint_list], gt_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list])
        mpjpe = mpjpe.reshape(batch_size, num_samples)
        mpjpe_all = mpjpe_all.reshape(batch_size, num_samples, -1)
        re = re.reshape(batch_size, num_samples)
        re_all = re_all.reshape(batch_size, num_samples, -1)

        # Compute joint errors after optimization, if available.
        if opt_output is not None:
            opt_keypoints_3d = opt_output['model_joints']
            opt_keypoints_3d -= opt_keypoints_3d[:, [self.pelvis_ind]]
            opt_mpjpe, opt_re, opt_mpjpe_all, opt_re_all = eval_pose(
                opt_keypoints_3d[:, self.keypoint_list], gt_keypoints_3d[:, 0, self.keypoint_list])

        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_mpjpe'):
            mode_mpjpe = mpjpe[:, 0]
            self.mode_mpjpe[self.counter:self.counter+batch_size] = mode_mpjpe
        if hasattr(self, 'mode_re'):
            mode_re = re[:, 0]
            self.mode_re[self.counter:self.counter+batch_size] = mode_re
        if hasattr(self, 'mean_mpjpe'):
            min_mpjpe = mpjpe.mean(axis=-1)
            self.mean_mpjpe[self.counter:self.counter+batch_size] = min_mpjpe
        if hasattr(self, 'mean_re'):
            min_re = re.mean(axis=-1)
            self.mean_re[self.counter:self.counter+batch_size] = min_re
        if hasattr(self, 'min_mpjpe'):
            min_mpjpe = mpjpe.min(axis=-1)
            self.min_mpjpe[self.counter:self.counter+batch_size] = min_mpjpe
        if hasattr(self, 'min_re'):
            min_re = re.min(axis=-1)
            self.min_re[self.counter:self.counter+batch_size] = min_re
        
            
        if hasattr(self, 'rmms_mpjpe') or hasattr(self, 'rmsf_mpjpe') or hasattr(self, 'amms_mpjpe') or  hasattr(self, 'amsf_mpjpe'):
            
            n_measure_points = self.n_measure_points
        
            pred_vertices = output["pred_vertices"]
            target_vertices = batch['vertices']
            
            uv_path = "data/UV_Processed.mat"
            DP_UV = loadmat(uv_path)
            faces_densepose = torch.from_numpy((DP_UV['All_Faces'] - 1).astype(np.int64)).cuda()
            verts_map = torch.from_numpy(DP_UV['All_vertices'][0].astype(np.int64)).cuda() - 1
            target_mesh = Meshes(verts=target_vertices[:, verts_map, :], faces=faces_densepose[None].expand(batch_size, -1, -1))
            target_normals_ = target_mesh.verts_normals_packed().reshape(batch_size, -1, 3)
            target_normals = torch.empty_like(target_vertices)
            target_normals[:, verts_map] = target_normals_
            
            n_parallel = 64
            pred_normals_ = []
            for i in range(batch_size * num_samples // n_parallel):
                start = i * n_parallel
                end = (i+1) * n_parallel
                pred_mesh = Meshes(verts=pred_vertices.reshape(batch_size * num_samples, -1, 3)[start:end, verts_map, :],
                                   faces=faces_densepose[None].expand(n_parallel, -1, -1))
                pred_normals_.append(pred_mesh.verts_normals_packed())
            pred_normals_ = torch.cat(pred_normals_, dim=0).reshape(batch_size, num_samples, -1 , 3)
            pred_normals = torch.empty_like(pred_vertices)
            pred_normals[:, :, verts_map] = pred_normals_
            
        if hasattr(self, 'rmms_mpjpe'):
            idx_random_sampling = torch.randint(0, pred_vertices.shape[2], (batch_size, n_measure_points)).cuda()
            idx_min_error_rmms = self.filtering(num_samples, pred_vertices, target_vertices, pred_normals, target_normals, idx_random_sampling)
            
            rmms_mpjpe = torch.gather(torch.from_numpy(mpjpe), 1, idx_min_error_rmms[:, None].cpu())
            self.rmms_mpjpe[self.counter:self.counter+batch_size] = rmms_mpjpe.numpy().squeeze()
        
        if hasattr(self, 'rmms_re'):
            rmms_re = torch.gather(torch.from_numpy(re), 1, idx_min_error_rmms[:, None].cpu())
            self.rmms_re[self.counter:self.counter+batch_size] = rmms_re.numpy().squeeze()
            
        if hasattr(self, 'rmsf_mpjpe'):
            refined_keypoints = self.optimization(flow_net, smpl, target_vertices, output, idx_min_error_rmms, idx_random_sampling)
            rmsf_mpjpe, rmsf_re, _, _ = eval_pose(refined_keypoints[:, self.keypoint_list],
                                                  gt_keypoints_3d[:, 0, self.keypoint_list])
            self.rmsf_mpjpe[self.counter:self.counter+batch_size] = rmsf_mpjpe.squeeze()
            
        if hasattr(self, 'rmsf_re'):
            self.rmsf_re[self.counter:self.counter+batch_size] = rmsf_re.squeeze()
            
        if hasattr(self, 'amms_mpjpe') or hasattr(self, 'amms_re'):
            self.timer.start()
            idx_most_uncertain_vertex = self.simulated_active_measurement(pred_vertices, pred_normals, n_measure_points)
            self.timer.end()
            time_am = self.timer.elapsed_time()
            idx_min_error_amms = self.filtering(num_samples, pred_vertices, target_vertices, pred_normals, target_normals, idx_most_uncertain_vertex)
            amms_mpjpe = torch.gather(torch.from_numpy(mpjpe), 1, idx_min_error_amms[:, None].cpu())
            self.amms_mpjpe[self.counter:self.counter+batch_size] = amms_mpjpe.numpy().squeeze()
            
        if hasattr(self, 'amms_re'):
            amms_re = torch.gather(torch.from_numpy(re), 1, idx_min_error_amms[:, None].cpu())
            self.amms_re[self.counter:self.counter+batch_size] = amms_re.numpy().squeeze()
            
        if hasattr(self, 'time_am'):
            self.time_am[self.counter:self.counter+batch_size] = time_am

        if hasattr(self, 'amsf_mpjpe') or hasattr(self, 'amsf_re'):
            self.timer.start()
            refined_keypoints = self.optimization(flow_net, smpl, target_vertices, output, idx_min_error_amms, idx_most_uncertain_vertex)
            self.timer.end()
            time_sf = self.timer.elapsed_time()
            amsf_mpjpe, amsf_re, _, _ = eval_pose(refined_keypoints[:, self.keypoint_list],
                                                  gt_keypoints_3d[:, 0, self.keypoint_list])
            self.amsf_mpjpe[self.counter:self.counter+batch_size] = amsf_mpjpe.squeeze()    
            
        if hasattr(self, 'amsf_re'):
            self.amsf_re[self.counter:self.counter+batch_size] = amsf_re.squeeze()
            
        if hasattr(self, 'time_sf'):
            self.time_sf[self.counter:self.counter+batch_size] = time_sf
        
        self.counter += batch_size
        
        
    def simulated_active_measurement(self, pred_vertices, pred_normals, n_measure_points):
        pred_vertices_var = torch.var(pred_vertices, dim=1).sum(dim=-1)
        pred_normals_var = torch.var(pred_normals, dim=1).sum(dim=-1)
        var = pred_vertices_var * 100 - pred_normals_var * 1
        
        idxs = []
        dist_from_measured = torch.zeros_like(var)
        softmax = torch.nn.Softmax(dim=1)
        weight = 1
        while True:
            # idx = torch.argsort(var + dist_from_measured, dim=-1, descending=True)[:, :1]
            cost = var + dist_from_measured
            cost *= weight
            idx = Categorical(probs=softmax(cost)).sample()[:, None]
            idxs.append(idx)
            vertex_measured = torch.gather(pred_vertices[:, 0], 1, idx[:, :, None].expand(-1, -1, 3))
            dist_from_measured += torch.norm(pred_vertices[:, 0] - vertex_measured, dim=-1)
            num_unique = np.array([len(torch.unique(i)) for i in torch.cat(idxs, dim=-1)])
            if np.all(num_unique > n_measure_points):
                break
        idx_most_uncertain_vertex = torch.cat(idxs, dim=-1).cpu().numpy()
        idx_most_uncertain_vertex = torch.LongTensor(np.array([unsort_unique(idx)[:n_measure_points] for idx in idx_most_uncertain_vertex])).cuda()
        
        return idx_most_uncertain_vertex
    
        
    def filtering(self, num_samples, pred_vertices, target_vertices, pred_normals, target_normals, idx_vert, sigma_noise=0.02):
        pred_vertices_filter = torch.gather(pred_vertices, 2, idx_vert[:, None, :, None].expand(-1, num_samples, -1, 3))
        target_vertices_filter = torch.gather(target_vertices, 1, idx_vert[:, :, None].expand(-1, -1, 3)).unsqueeze(1)
        # noise on depth measurement (y axis in the camera coordinate system)
        noise = torch.randn_like(target_vertices_filter) * sigma_noise
        noise[..., [0, 2]] = 0
        target_vertices_filter_noise = target_vertices_filter + noise
        vertex_error = torch.norm(pred_vertices_filter - target_vertices_filter_noise, dim=-1).mean(dim=-1)
        
        pred_normals_filter = torch.gather(pred_normals, 2, idx_vert[:, None, :, None].expand(-1, num_samples, -1, 3))
        target_normals_filter = torch.gather(target_normals, 1, idx_vert[:, :, None].expand(-1, -1, 3))
        normal_error = - torch.einsum('ijkl,ikl->ijk', pred_normals_filter, target_normals_filter).mean(dim=-1)
        
        error = normal_error + vertex_error #- output["log_prob"] / 1000
        # error = vertex_error - output["log_prob"] / 1000
        idx_min_error = torch.argmin(error, dim=1)
        
        return idx_min_error


    def optimization(self, flow_net, smpl, target_vertices, output, idx_min_error, idx_vert, sigma_noise=0.02):
        target_vertices_filter = torch.gather(target_vertices, 1, idx_vert[:, :, None].expand(-1, -1, 3)).unsqueeze(1)
        # noise on depth measurement (y axis in the camera coordinate system)
        noise = torch.randn_like(target_vertices_filter) * sigma_noise
        noise[..., [0, 2]] = 0
        target_vertices_filter_noise = target_vertices_filter + noise
        pcls = Pointclouds(points=[v[0] for v in target_vertices_filter_noise])

        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()
        
        # Get predicted betas
        betas = torch.gather(output['pred_smpl_params']['betas'].detach().clone(), 1, idx_min_error[:, None, None].tile([1, 1, 10]))[:, 0]
        global_orient = torch.gather(output['pred_smpl_params']['global_orient'].detach().clone(), 1, idx_min_error[:, None, None, None, None].tile([1, 1, 1, 3, 3]))[:, 0]
        z_init = torch.gather(output['z'].detach().clone(), 1, idx_min_error[:, None, None].tile([1, 1, 144]))[:, 0]
        z = z_init.detach().clone()
        conditioning_feats = output['conditioning_feats'].detach().clone()

        # Make z, betas and camera_translation optimizable
        z.requires_grad = True

        # Setup optimizer
        opt_params = [z]
        optimizer = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=5, line_search_fn='strong_wolfe')
        
        def pose_prior(z1, z2):
            return ((z1 - z2) ** 2).mean(dim=1)

        # Define fitting closure
        def closure():
            optimizer.zero_grad(set_to_none=True)
            smpl_params, _, _, _, _ = flow_net(
                conditioning_feats, z=z.unsqueeze(1))
            smpl_params = {k: v.squeeze(1)
                            for k, v in smpl_params.items()}
            # Override regression betas with the optimizable variable
            smpl_params['betas'] = betas
            smpl_params['global_orient'] = global_orient
            smpl_output = smpl(**smpl_params, pose2rot=False)
            curr_vertices = smpl_output.vertices[:, self.Vmap]
            mesh = Meshes(verts=list(curr_vertices),
                            faces=[self.faces] * len(curr_vertices))
            # packed representation for faces
            verts_packed = mesh.verts_packed()
            faces_packed = mesh.faces_packed()
            tris = verts_packed[faces_packed]  # (T, 3, 3)
            tris_first_idx = mesh.mesh_to_faces_packed_first_idx()

            loss = point_face_distance(
                points, points_first_idx, tris, tris_first_idx, max_points, 0.005).mean()

            loss += pose_prior(z, z_init).mean() * 1e-2
            loss.backward()
            return loss

        # Run fitting until convergence
        gtol = 1e-9
        ftol = 1e-9
        prev_loss = None
        for i in range(10):
            loss = optimizer.step(closure)
            if i > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change < ftol:
                    break
            if all([torch.abs(var.grad.view(-1).max()).item() < gtol
                    for var in opt_params if var.grad is not None]):
                break
            prev_loss = loss.item()

        with torch.no_grad():
            smpl_params, _, _, _, _ = flow_net(
                conditioning_feats, z=z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1)
                            for k, v in smpl_params.items()}
            smpl_params['betas'] = betas
            smpl_output = smpl(**smpl_params, pose2rot=False)
            opt_keypoints = smpl_output.joints
            opt_vertices = smpl_output.vertices
        
        refined_vertices = opt_vertices
        refined_keypoints = opt_keypoints
        
        refined_keypoints -= refined_keypoints[:, [self.pelvis_ind]]
        
        return refined_keypoints
