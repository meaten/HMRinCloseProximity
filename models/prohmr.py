from typing import Dict

import torch
from prohmr.models.prohmr import ProHMR
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection

class CustomProHMR(ProHMR):
    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        if train:
            num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES
        else:
            num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES


        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            pred_smpl_params, pred_cam, log_prob, z, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat((pred_smpl_params_mode[k], v), dim=1) for k,v in pred_smpl_params.items()}
            pred_cam = torch.cat((pred_cam_mode, pred_cam), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
            z = torch.cat([z_0, z], dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        # Store useful regression outputs to the output dict
        output = {}
        
        output['z'] = z
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['log_prob'] = log_prob.detach()
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, num_samples, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, :, 1],
                                  pred_cam[:, :, 2],
                                  2*focal_length[:, :, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        
        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        #pred_smpl_params['betas'] = batch["smpl_params"]['betas'][:, None].expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        
        batch["vertices"] = self.smpl(
            global_orient=aa_to_rotmat(batch["smpl_params"]["global_orient"]),
            body_pose=aa_to_rotmat(batch["smpl_params"]["body_pose"].reshape(-1,3)).reshape(batch_size, -1, 3, 3),
            betas=batch["smpl_params"]["betas"]
        ).vertices
        
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, num_samples, -1, 2)
        return output