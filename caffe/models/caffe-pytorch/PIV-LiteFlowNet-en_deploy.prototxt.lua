require 'nn'
local model = {}
-- warning: module 'CustomData1 [type CustomData]' not found
-- warning: module 'Silence_blob3 [type Silence]' not found
-- warning: module 'Eltwise1 [type Eltwise]' not found
-- warning: module 'blob4_Eltwise1_0_split [type Split]' not found
-- warning: module 'Eltwise2 [type Eltwise]' not found
-- warning: module 'img0s_aug [type DataAugmentation]' not found
-- warning: module 'img0_aug_img0s_aug_0_split [type Split]' not found
-- warning: module 'blob7_img0s_aug_1_split [type Split]' not found
-- warning: module 'aug_params1 [type GenerateAugmentationParameters]' not found
-- warning: module 'blob8_aug_params1_0_split [type Split]' not found
-- warning: module 'img1s_aug [type DataAugmentation]' not found
-- warning: module 'img1_aug_img1s_aug_0_split [type Split]' not found
-- warning: module 'FlowAugmentation1 [type FlowAugmentation]' not found
-- warning: module 'FlowScaling [type Eltwise]' not found
-- warning: module 'scaled_flow_gt_aug_FlowScaling_0_split [type Split]' not found
table.insert(model, {'conv1', nn.SpatialConvolution(3, 32, 7, 7, 1, 1, 3, 3)})
table.insert(model, {'ReLU1a', nn.ReLU(true)})
-- warning: module 'F0_L1_ReLU1a_0_split [type Split]' not found
table.insert(model, {'ReLU1b', nn.ReLU(true)})
-- warning: module 'F1_L1_ReLU1b_0_split [type Split]' not found
table.insert(model, {'conv2_1', nn.SpatialConvolution(32, 32, 3, 3, 2, 2, 1, 1)})
table.insert(model, {'ReLU2a_1', nn.ReLU(true)})
table.insert(model, {'ReLU2b_1', nn.ReLU(true)})
table.insert(model, {'conv2_2', nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU2a_2', nn.ReLU(true)})
table.insert(model, {'ReLU2b_2', nn.ReLU(true)})
table.insert(model, {'conv2_3', nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU2a_3', nn.ReLU(true)})
-- warning: module 'F0_L2_ReLU2a_3_0_split [type Split]' not found
table.insert(model, {'ReLU2b_3', nn.ReLU(true)})
-- warning: module 'F1_L2_ReLU2b_3_0_split [type Split]' not found
table.insert(model, {'conv3_1', nn.SpatialConvolution(32, 64, 3, 3, 2, 2, 1, 1)})
table.insert(model, {'ReLU3a_1', nn.ReLU(true)})
table.insert(model, {'ReLU3b_1', nn.ReLU(true)})
table.insert(model, {'conv3_2', nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU3a_2', nn.ReLU(true)})
-- warning: module 'F0_L3_ReLU3a_2_0_split [type Split]' not found
table.insert(model, {'ReLU3b_2', nn.ReLU(true)})
-- warning: module 'F1_L3_ReLU3b_2_0_split [type Split]' not found
table.insert(model, {'conv4_1', nn.SpatialConvolution(64, 96, 3, 3, 2, 2, 1, 1)})
table.insert(model, {'ReLU4a_1', nn.ReLU(true)})
table.insert(model, {'ReLU4b_1', nn.ReLU(true)})
table.insert(model, {'conv4_2', nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU4a_2', nn.ReLU(true)})
-- warning: module 'F0_L4_ReLU4a_2_0_split [type Split]' not found
table.insert(model, {'ReLU4b_2', nn.ReLU(true)})
-- warning: module 'F1_L4_ReLU4b_2_0_split [type Split]' not found
table.insert(model, {'conv5', nn.SpatialConvolution(96, 128, 3, 3, 2, 2, 1, 1)})
table.insert(model, {'ReLU5a', nn.ReLU(true)})
-- warning: module 'F0_L5_ReLU5a_0_split [type Split]' not found
table.insert(model, {'ReLU5b', nn.ReLU(true)})
-- warning: module 'F1_L5_ReLU5b_0_split [type Split]' not found
table.insert(model, {'conv6', nn.SpatialConvolution(128, 192, 3, 3, 2, 2, 1, 1)})
table.insert(model, {'ReLU6a', nn.ReLU(true)})
-- warning: module 'F0_L6_ReLU6a_0_split [type Split]' not found
table.insert(model, {'ReLU6b', nn.ReLU(true)})
-- warning: module 'F1_L6_ReLU6b_0_split [type Split]' not found
-- warning: module 'corr_L6 [type Correlation]' not found
table.insert(model, {'ReLU_corr_L6', nn.ReLU(true)})
table.insert(model, {'conv1_D1_L6', nn.SpatialConvolution(49, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU1_D1_L6', nn.ReLU(true)})
table.insert(model, {'conv2_D1_L6', nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU2_D1_L6', nn.ReLU(true)})
table.insert(model, {'conv3_D1_L6', nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU3_D1_L6', nn.ReLU(true)})
table.insert(model, {'scaled_flow_D1_L6', nn.SpatialConvolution(32, 2, 3, 3, 1, 1, 1, 1)})
-- warning: module 'scaled_flow_D1_L6_scaled_flow_D1_L6_0_split [type Split]' not found
-- warning: module 'Downsample_L6 [type Downsample]' not found
-- warning: module 'scaled_flow_label_L6_Downsample_L6_0_split [type Split]' not found
-- warning: module 'scaled_flow_D1_L6_loss [type L1Loss]' not found
-- warning: module 'FlowUnscaling_L6_D2 [type Eltwise]' not found
-- warning: module 'flow_D1_L6_FlowUnscaling_L6_D2_0_split [type Split]' not found
-- warning: module 'gxy_L6 [type Grid]' not found
-- warning: module 'gxy_L6_gxy_L6_0_split [type Split]' not found
-- warning: module 'coords_D1_L6 [type Eltwise]' not found
-- warning: module 'warped_F1_L6 [type Warp]' not found
-- warning: module 'F_D2_L6 [type Concat]' not found
table.insert(model, {'conv1_D2_L6', nn.SpatialConvolution(386, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU1_D2_L6', nn.ReLU(true)})
table.insert(model, {'conv2_D2_L6', nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU2_D2_L6', nn.ReLU(true)})
table.insert(model, {'conv3_D2_L6', nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'ReLU3_D2_L6', nn.ReLU(true)})
table.insert(model, {'scaled_flow_D2_res_L6', nn.SpatialConvolution(32, 2, 3, 3, 1, 1, 1, 1)})
-- warning: module 'scaled_flow_D2_L6 [type Eltwise]' not found
-- warning: module 'scaled_flow_D2_L6_scaled_flow_D2_L6_0_split [type Split]' not found
-- warning: module 'scaled_flow_D2_L6_loss [type L1Loss]' not found
-- warning: module 'slice_scaled_flow_D_L6 [type Slice]' not found
-- warning: module 'scaled_flow_D_L6_x_slice_scaled_flow_D_L6_0_split [type Split]' not found
-- warning: module 'scaled_flow_D_L6_y_slice_scaled_flow_D_L6_1_split [type Split]' not found
-- warning: module 'reshaped_scaled_flow_D_L6_x [type Im2col]' not found
-- warning: module 'reshaped_scaled_flow_D_L6_y [type Im2col]' not found
-- warning: module 'mean_scaled_flow_D_L6_x [type Reduction]' not found
-- warning: module 'scaled_flow_D_nomean_L6_x [type Bias]' not found
-- warning: module 'mean_scaled_flow_D_L6_y [type Reduction]' not found
-- warning: module 'scaled_flow_D_nomean_L6_y [type Bias]' not found
-- warning: module 'FlowUnscaling_L6_R [type Eltwise]' not found
-- warning: module 'flow_D2_L6_FlowUnscaling_L6_R_0_split [type Split]' not found
-- warning: module 'Downsample_img0_aug_L6 [type Downsample]' not found
-- warning: module 'Downsample_img1_aug_L6 [type Downsample]' not found
-- warning: module 'coords_R_L6 [type Eltwise]' not found
-- warning: module 'warped_img1_aug_L6 [type Warp]' not found
-- warning: module 'img_diff_L6 [type Eltwise]' not found
-- warning: module 'channelNorm_L6 [type ChannelNorm]' not found
-- warning: module 'concat_F0_R_L6 [type Concat]' not found
table.insert(model, {'conv1_R_L6', nn.SpatialConvolution(195, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv1_R_L6', nn.ReLU(true)})
table.insert(model, {'conv2_R_L6', nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv2_R_L6', nn.ReLU(true)})
table.insert(model, {'conv3_R_L6', nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv3_R_L6', nn.ReLU(true)})
table.insert(model, {'conv4_R_L6', nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv4_R_L6', nn.ReLU(true)})
table.insert(model, {'conv5_R_L6', nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv5_R_L6', nn.ReLU(true)})
table.insert(model, {'conv6_R_L6', nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_conv6_R_L6', nn.ReLU(true)})
table.insert(model, {'dist_R_L6', nn.SpatialConvolution(32, 9, 3, 3, 1, 1, 1, 1)})
-- warning: module 'sq_dist_R_L6 [type Power]' not found
-- warning: module 'neg_sq_dist_R_L6 [type Eltwise]' not found
return model