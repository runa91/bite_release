
import torch

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


from combined_model.model_shape_v7_withref_withgraphcnn import ModelImageTo3d_withshape_withproj 


soft_max = torch.nn.Softmax(dim=1)


def get_summarized_bite_result(output, output_unnorm, output_reproj, output_ref_unnorm, output_orig_ref_comparison, output_ref_unnorm_new=None, output_orig_ref_comparison_new=None, result_networks=['ref']):
    all_sum_res = {}
    for result_network in result_networks:
        assert result_network in ['normal', 'ref', 'multref']
        # variabled that are not refined
        res = {}
        res['hg_keyp_scores'] = output['keypoints_scores']
        res['hg_keyp_norm'] = output['keypoints_norm']
        res['hg_keyp_256'] = (output['keypoints_norm']+1)/2*(256-1)
        res['hg_silh_prep'] = soft_max(output['seg_hg'])[:, 1, :, :]    # (bs, 256, 256)
        res['betas'] = output_reproj['betas']
        res['betas_limbs'] = output_reproj['betas_limbs']
        res['z'] = output_reproj['z']
        if result_network == 'normal':
            # STEP 1: normal network
            res['vertices_smal'] = output_reproj['vertices_smal']
            res['flength'] = output_unnorm['flength']
            res['pose_rotmat'] = output_unnorm['pose_rotmat']
            res['trans'] = output_unnorm['trans']
            res['pred_keyp'] = output_reproj['keyp_2d']
            res['pred_silh'] = output_reproj['silh']
            res['prefix'] = 'normal_'
        elif result_network == 'ref':
            # STEP 1: refinement network
            res['vertices_smal'] = output_ref_unnorm['vertices_smal']
            res['flength'] = output_ref_unnorm['flength']
            res['pose_rotmat'] = output_ref_unnorm['pose_rotmat']
            res['trans'] = output_ref_unnorm['trans']
            res['pred_keyp'] = output_ref_unnorm['keyp_2d']
            res['pred_silh'] = output_ref_unnorm['silh']
            res['prefix'] = 'ref_'
            if 'vertexwise_ground_contact' in output_ref_unnorm.keys():
                res['vertexwise_ground_contact'] = output_ref_unnorm['vertexwise_ground_contact']
            ''''
            if return_mesh_with_gt_groundplane and 'gc' in target_dict.keys():
                bs = vertices_smal.shape[0]
                target_gc_class = target_dict['gc'][:, :, 0]
                sel_verts = torch.index_select(output_ref_unnorm['vertices_smal'], dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((bs, remeshing_relevant_faces.shape[0], 3, 3))
                verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)
                target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)
            '''
            res['isflat_prep'] = soft_max(output_ref_unnorm['isflat'])[:, 1]


        else:
            # STEP 1: next loop in refinemnet network
            assert (output_ref_unnorm_new is not None) 
            res['vertices_smal'] = output_ref_unnorm_new['vertices_smal']
            res['flength'] = output_ref_unnorm_new['flength']
            res['pose_rotmat'] = output_ref_unnorm_new['pose_rotmat']
            res['trans'] = output_ref_unnorm_new['trans']
            res['pred_keyp'] = output_ref_unnorm_new['keyp_2d']
            res['pred_silh'] = output_ref_unnorm_new['silh']
            res['prefix'] = 'multref_'
            if 'vertexwise_ground_contact' in output_ref_unnorm_new.keys():
                res['vertexwise_ground_contact'] = output_ref_unnorm_new['vertexwise_ground_contact']
        all_sum_res[result_network] = res
    return all_sum_res


class BITEInferenceModel():        #(nn.Module):
    def __init__(self, cfg, path_model_file_complete, norm_dict, device='cuda'):
        # def __init__(self, bp, model_weight_path=None, model_weight_stackedhg_path=None, device='cuda'):
        # self.bp = bp
        self.cfg = cfg
        self.device = device
        self.norm_dict = norm_dict

        # prepare complete model
        self.complete_model = ModelImageTo3d_withshape_withproj(
            smal_model_type=cfg.smal.SMAL_MODEL_TYPE, smal_keyp_conf=cfg.smal.SMAL_KEYP_CONF, \
            num_stage_comb=cfg.params.NUM_STAGE_COMB, num_stage_heads=cfg.params.NUM_STAGE_HEADS, \
            num_stage_heads_pose=cfg.params.NUM_STAGE_HEADS_POSE, trans_sep=cfg.params.TRANS_SEP, \
            arch=cfg.params.ARCH, n_joints=cfg.params.N_JOINTS, n_classes=cfg.params.N_CLASSES, \
            n_keyp=cfg.params.N_KEYP, n_bones=cfg.params.N_BONES, n_betas=cfg.params.N_BETAS, n_betas_limbs=cfg.params.N_BETAS_LIMBS, \
            n_breeds=cfg.params.N_BREEDS, n_z=cfg.params.N_Z, image_size=cfg.params.IMG_SIZE, \
            silh_no_tail=cfg.params.SILH_NO_TAIL, thr_keyp_sc=cfg.params.KP_THRESHOLD, add_z_to_3d_input=cfg.params.ADD_Z_TO_3D_INPUT,
            n_segbps=cfg.params.N_SEGBPS, add_segbps_to_3d_input=cfg.params.ADD_SEGBPS_TO_3D_INPUT, add_partseg=cfg.params.ADD_PARTSEG, n_partseg=cfg.params.N_PARTSEG, \
            fix_flength=cfg.params.FIX_FLENGTH, structure_z_to_betas=cfg.params.STRUCTURE_Z_TO_B, structure_pose_net=cfg.params.STRUCTURE_POSE_NET,
            nf_version=cfg.params.NF_VERSION, ref_net_type=cfg.params.REF_NET_TYPE, graphcnn_type=cfg.params.GRAPHCNN_TYPE, isflat_type=cfg.params.ISFLAT_TYPE, shaperef_type=cfg.params.SHAPEREF_TYPE) 
        
        # load trained model
        print(path_model_file_complete)
        assert os.path.isfile(path_model_file_complete)
        print('Loading model weights from file: {}'.format(path_model_file_complete))
        checkpoint_complete = torch.load(path_model_file_complete)
        state_dict_complete = checkpoint_complete['state_dict']
        self.complete_model.load_state_dict(state_dict_complete)    # , strict=False)        
        self.complete_model = self.complete_model.to(self.device)
        self.complete_model.eval()

        self.smal_model_type = self.complete_model.smal.smal_model_type

    def get_selected_results(self, preds_dict=None, input_img_prep=None, result_networks=['ref']):
        assert ((preds_dict is not None) or (input_img_prep is not None))
        if preds_dict is None:
            preds_dict = self.get_all_results(input_img_prep)
        all_sum_res = get_summarized_bite_result(preds_dict['output'], preds_dict['output_unnorm'], preds_dict['output_reproj'], preds_dict['output_ref_unnorm'], preds_dict['output_orig_ref_comparison'], result_networks=result_networks)
        return all_sum_res

    def get_selected_results_multiple_refinements(self, preds_dict=None, input_img_prep=None, result_networks=['multref']):
        assert ((preds_dict is not None) or (input_img_prep is not None))
        if preds_dict is None:
            preds_dict = self.get_all_results_multiple_refinements(input_img_prep)
        all_sum_res = get_summarized_bite_result(preds_dict['output'], preds_dict['output_unnorm'], preds_dict['output_reproj'], preds_dict['output_ref_unnorm'], preds_dict['output_orig_ref_comparison'], preds_dict['output_ref_unnorm_new'], preds_dict['output_orig_ref_comparison_new'], result_networks=result_networks)
        return all_sum_res


    def get_all_results(self, input_img_prep):
        output, output_unnorm, output_reproj, output_ref, output_ref_comp = self.complete_model(input_img_prep, norm_dict=self.norm_dict)        
        preds_dict = {'output': output,
                        'output_unnorm': output_unnorm,
                        'output_reproj': output_reproj,
                        'output_ref_unnorm': output_ref, 
                        'output_orig_ref_comparison': output_ref_comp
                        }
        return preds_dict


    def get_all_results_multiple_refinements(self, input_img_prep):
        preds_dict = self.complete_model.forward_with_multiple_refinements(input_img_prep, norm_dict=self.norm_dict)        
        # output, output_unnorm, output_reproj, output_ref, output_ref_comp, output_ref_unnorm_new, output_orig_ref_comparison_new    
        return preds_dict




















