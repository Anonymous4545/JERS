from model import *
from utils import *


def train_jers(img_size,
                  abn_stage,
                  seg_stage,
                  train_set_name,
                  val_set_name,
                  test_set_name,
                  batch_size,
                  num_epochs,
                  learning_rate,
                  model_name,
                  reg_loss_name,
                  mask_smooth_loss_func,
                  lamda_mask,
                  gamma,
                  beta,
                  save_every_epoch,
                  dice_label,
                  if_compute_dice,
                  if_train_aug,
                  fixed_set_name,
                  save_start_epoch):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model_name.to(device)

    mask_smooth = mask_smooth_loss_func.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if reg_loss_name == "NCC":
        reg_loss_func = NCC().loss
    elif reg_loss_name == "GCC":
        reg_loss_func = GCC()
    elif reg_loss_name == "MSE":
        reg_loss_func = nn.MSELoss().to(device)

    seg_loss_func = nn.CrossEntropyLoss()
    ssim_loss_func = mutual_information

    cur_path = os.getcwd()
    result_path = cur_path + '/result'

    loss_log_path = result_path + '/loss_log'
    create_folder(result_path, 'loss_log')

    sample_img_path = result_path + '/sample_img'
    create_folder(result_path, 'sample_img')

    model_save_path = result_path + '/model'
    create_folder(result_path, 'model')

    model_str = str(model)[0:str(model).find("(")]
    # smooth_str=str(smooth)[0:str(smooth).find("(")]
    lamda_mask_str = str(lamda_mask)
    gamma_str = str(gamma)
    beta_str = str(beta)

    lr_str = str(learning_rate)
    dataset_str = train_set_name[0:str(train_set_name).find(".")]

    seg_stage_str = str(seg_stage)
    abn_stage_str = str(abn_stage)

    if if_train_aug == True:
        modal_name = model_str + "_" + seg_stage_str + "_" + abn_stage_str + "_" + lamda_mask_str + "_" + gamma_str + "_" + beta_str + "_" + lr_str + "_" + dataset_str + "_trainAug_true"
    else:
        modal_name = model_str + "_" + seg_stage_str + "_" + abn_stage_str + "_" + lamda_mask_str + "_" + gamma_str + "_" + beta_str + "_" + lr_str + "_" + dataset_str + "_trainAug_Flase"

    modal_path = sample_img_path + "/" + modal_name
    create_folder(sample_img_path, modal_name)

    sample_o_path = modal_path + "/" + "o"
    sample_t_path = modal_path + "/" + "t"
    create_folder(modal_path, "o")
    create_folder(modal_path, "t")

    sample_reginv_mask_path = modal_path + "/" + "reginv_am"
    sample_segpred_mask_path = modal_path + "/" + "segpred_am"
    create_folder(modal_path, "reginv_am")
    create_folder(modal_path, "segpred_am")

    for i in range(int(seg_stage)):
        idx = i + 1
        s_name = "s_" + str(idx)
        s_mask_name = "s_" + str(idx) + "_mask"
        sample_s_path = modal_path + "/" + s_name
        sample_s_mask_path = modal_path + "/" + s_mask_name

        create_folder(modal_path, s_name)
        create_folder(modal_path, s_mask_name)

    for q in range(int(abn_stage)):
        qdx = q + 1
        r_name = "r_" + str(qdx)
        r_grid_name = "r_" + str(qdx) + "_grid"
        sample_r_path = modal_path + "/" + r_name
        sample_r_grid_path = modal_path + "/" + r_grid_name

        create_folder(modal_path, r_name)
        create_folder(modal_path, r_grid_name)

    if if_train_aug == True:
        modal_info = "Model: {}    seg_stage: {}    abn_stage: {}      λ_mask: {}    γ: {}    β: {}    lr: {}    dataset: {}    train_Agu: {}".format(
            model_str, seg_stage_str, abn_stage_str, lamda_mask_str, gamma_str, beta_str, lr_str, dataset_str, "True_2")

    else:
        modal_info = "Model: {}    seg_stage: {}    abn_stage: {}      λ_mask: {}    γ: {}    β: {}    lr: {}    dataset: {}    train_Agu: {}".format(
            model_str, seg_stage_str, abn_stage_str, lamda_mask_str, gamma_str, beta_str, lr_str, dataset_str, "False")

    create_log(modal_info, loss_log_path, modal_name)

    print(modal_info)

    train_loader = load_data_no_fix(train_set_name, batch_size)
    val_loader = load_data_no_fix(val_set_name, batch_size)
    test_loader = load_data_no_fix(test_set_name, batch_size)

    for epoch in range(num_epochs):

        total_loss_train = []
        total_seg_loss_train = []
        total_sim_loss_train = []

        start = time.time()

        fixed_data = torch.from_numpy(np.load('./dataset/' + fixed_set_name + "_fixed_96.npy")).to(device).view(-1, 1,
                                                                                                                img_size,
                                                                                                                img_size,
                                                                                                                img_size).float()
        fixed_label_am = torch.from_numpy(np.load('./dataset/' + fixed_set_name + "_fixed_amlabel_96.npy")).to(
            device).view(-1, 1, img_size, img_size, img_size).float()
        fixed_data_val = torch.from_numpy(np.load('./dataset/' + fixed_set_name + "_fixed_96.npy")).to(device).view(-1,
                                                                                                                    1,
                                                                                                                    img_size,
                                                                                                                    img_size,
                                                                                                                    img_size).float()
        fixed_label_am_val = torch.from_numpy(np.load('./dataset/' + fixed_set_name + "_fixed_amlabel_96.npy")).to(
            device).view(-1, 1, img_size, img_size, img_size).float()

        for i, x in enumerate(train_loader):
            moving_data, moving_label_bm, moving_label_am = x

            # fixed_data=fixed_data.to(device).view(-1,1,img_size,img_size,img_size).float()
            moving_data = moving_data.to(device).view(-1, 1, img_size, img_size, img_size).float()
            moving_label_am = moving_label_am.to(device).view(-1, 1, img_size, img_size, img_size).float()

            if if_train_aug == True:

                if fixed_set_name == "LPBA40":
                    degree = 5
                    voxel = 5
                    scale_min = 0.98
                    scale_max = 1.02

                elif fixed_set_name == "CC359":
                    degree = 2
                    voxel = 2
                    scale_min = 0.98
                    scale_max = 1.02

                moving_data, moving_label_am = train_aug(moving_data, moving_label_am, img_size, degree, voxel,
                                                         scale_min, scale_max)

            optimizer.zero_grad()

            striped_list, mask_list, warped_list, theta_list, theta_list_inv, am_mov_pred, am_fix_2_moving = model(
                fixed_data,
                moving_data,
                fixed_label_am,
                if_train=True)

            # seg_net
            am_fix_2_moving_Nd = am_1d_2_Nd_torch(am_fix_2_moving)

            am_moving_pred_Nd = am_mov_pred

            seg_loss_train = beta * seg_loss_func(am_moving_pred_Nd, am_fix_2_moving_Nd)
            ####

            sim_loss_train = reg_loss_func(warped_list[-1], fixed_data)

            mask_smooth_loss = lamda_mask * sum([mask_smooth(i) for i in mask_list])

            loss_train = sim_loss_train + mask_smooth_loss + seg_loss_train
            loss_train.backward()
            optimizer.step()

            total_loss_train.append(loss_train.item())
            total_seg_loss_train.append(seg_loss_train.item())
            total_sim_loss_train.append(sim_loss_train.item())

        ave_loss_train, std_loss_train = get_ave_std(total_loss_train)
        ave_seg_loss_train, std_seg_loss_train = get_ave_std(total_seg_loss_train)
        ave_sim_loss_train, std_sim_loss_train = get_ave_std(total_sim_loss_train)

        if epoch % save_every_epoch == 0:
            model.eval()
            with torch.no_grad():

                total_loss_val_seg = []
                total_loss_val_sim_af = []

                total_f1_ext_val = []
                total_ssmi_reg_val = []
                total_dice_forward_val = []

                total_dice_fix_inv_val = []
                total_dice_seg_pred_val = []

                for j, y in enumerate(val_loader):

                    moving_data_val, moving_label_bm_val, moving_label_am_val = y

                    # fixed_data_val=fixed_data_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_data_val = moving_data_val.to(device).view(-1, 1, img_size, img_size, img_size).float()
                    # fixed_label_am_val=fixed_label_am_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_bm_val = moving_label_bm_val.to(device).view(-1, 1, img_size, img_size,
                                                                              img_size).float()
                    moving_label_am_val = moving_label_am_val.to(device).view(-1, 1, img_size, img_size,
                                                                              img_size).float()

                    striped_list_val, mask_list_val, warped_list_val, theta_list_val, theta_list_inv_val, am_mov_pred_val, am_fix_2_moving_val = model(
                        fixed_data_val,
                        moving_data_val,
                        fixed_label_am_val,
                        if_train=False)

                    # seg_loss_val=seg_loss_func(mask_list_val[-1],moving_label_bm_val)
                    sim_loss_val = reg_loss_func(warped_list_val[-1], fixed_data_val)

                    # total_loss_val_seg.append(seg_loss_val.item())
                    total_loss_val_sim_af.append(sim_loss_val.item())

                    merged_mask_list_val = torch.prod(torch.cat(mask_list_val), 0).unsqueeze(0)

                    f1_ext_val = f1_dice_loss(merged_mask_list_val, moving_label_bm_val)
                    total_f1_ext_val.append(f1_ext_val)

                    # jaccard_val=jaccard_loss(mask_list_val[-1],moving_label_bm_val)

                    moving_label_am_skulled_val = merged_mask_list_val * moving_label_am_val
                    reg_grid = F.affine_grid(theta_list_val[-1], moving_label_am_skulled_val.size(), align_corners=True)
                    warped_label_am_skulled_val = F.grid_sample(moving_label_am_skulled_val, reg_grid,
                                                                mode="nearest", align_corners=True,
                                                                padding_mode="zeros")

                    ###seg_net
                    am_fix_2_moving_Nd_val = am_1d_2_Nd_torch(am_fix_2_moving_val)

                    am_moving_pred_Nd_val = am_mov_pred_val
                    am_moving_pred_1d_val = torch.argmax(am_moving_pred_Nd_val, axis=1)

                    seg_loss_val = beta * seg_loss_func(am_moving_pred_Nd_val, am_fix_2_moving_Nd_val)
                    total_loss_val_seg.append(seg_loss_val.item())
                    ###

                    if if_compute_dice == True:
                        dice_forward_val = compute_label_dice(fixed_label_am_val, warped_label_am_skulled_val,
                                                              dice_label)
                        dice_fix_inv_val = compute_label_dice(am_fix_2_moving_val, moving_label_am_val, dice_label)
                        dice_seg_pred_val = compute_label_dice(am_moving_pred_1d_val, moving_label_am_val, dice_label)
                        ssmi_val = ssim_loss_func(warped_list_val[-1], fixed_data_val)

                    total_dice_forward_val.append(dice_forward_val)
                    total_dice_fix_inv_val.append(dice_fix_inv_val)
                    total_dice_seg_pred_val.append(dice_seg_pred_val)
                    total_ssmi_reg_val.append(ssmi_val)

                ave_loss_val_seg, std_loss_val_seg = get_ave_std(total_loss_val_seg)
                ave_loss_val_sim, std_loss_val_sim = get_ave_std(total_loss_val_sim_af)

                ave_f1_ext_val, std_ext_f1_val = get_ave_std(total_f1_ext_val)
                ave_ssmi_val, std_ssmi_val = get_ave_std(total_ssmi_reg_val)

                # ave_jaccard_val,std_jaccard_val=get_ave_std(total_jaccard_val)
                ave_dice_forward_val, std_dice_forward_val = get_ave_std(total_dice_forward_val)
                ave_dice_fix_inv_val, std_dice_fix_inv_val = get_ave_std(total_dice_fix_inv_val)
                ave_dice_seg_pred_val, std_dice_seg_pred_val = get_ave_std(total_dice_seg_pred_val)

                loss_info = "Epoch[{}/{}], All Training loss: {:.4f}/{:.4f} , Ext_val: {:.4f}/{:.4f}  ,  Reg_val: {:.4f}/{:.4f}   ,  Dice_seg_pred_val: {:.4f}/{:.4f}".format(
                    epoch, num_epochs,
                    ave_loss_train, std_loss_train,
                    ave_f1_ext_val, std_ext_f1_val,
                    ave_ssmi_val, std_ssmi_val,
                    ave_dice_seg_pred_val, std_dice_seg_pred_val)

                print(loss_info)
                append_log(loss_info, loss_log_path, modal_name)

                if epoch > save_start_epoch:

                    save_sample_any(epoch, "o", fixed_data_val, sample_o_path)
                    # save_nii_any(epoch,"o",fixed_data_val,sample_o_path)

                    save_sample_any(epoch, "t", moving_data_val, sample_t_path)
                    # save_nii_any(epoch,"t",moving_data_val,sample_t_path)

                    save_sample_any(epoch, "o_am", fixed_label_am_val, sample_o_path)
                    # save_nii_any(epoch,"o_am",fixed_label_am_val,sample_o_path)

                    save_sample_any(epoch, "t_am", moving_label_am_val, sample_t_path)
                    # save_nii_any(epoch,"t_am",moving_label_am_val,sample_t_path)

                    save_sample_any(epoch, "reginv_am", am_fix_2_moving_val, sample_reginv_mask_path)
                    # save_nii_any(epoch,"reginv_am",am_fix_2_moving_val,sample_reginv_mask_path)

                    save_sample_any(epoch, "segpred_am", am_moving_pred_1d_val.unsqueeze(0), sample_segpred_mask_path)
                    # save_nii_any(epoch,"segpred_am",am_moving_pred_1d_val.unsqueeze(0),sample_segpred_mask_path)

                    for t in range(int(seg_stage)):
                        tdx = t + 1
                        s_name = "s_" + str(tdx)
                        s_mask_name = "s_" + str(tdx) + "_mask"

                        sample_s_path = modal_path + "/" + s_name
                        sample_s_mask_path = modal_path + "/" + s_mask_name

                        save_sample_any(epoch, s_name, striped_list_val[t], sample_s_path)
                        # save_nii_any(epoch,s_name,striped_list_val[t],sample_s_path)

                        save_sample_any(epoch, s_mask_name, mask_list_val[t], sample_s_mask_path)

                    for y in range(int(abn_stage)):
                        ydx = y + 1
                        r_name = "r_" + str(ydx)
                        sample_r_path = modal_path + "/" + r_name

                        save_sample_any(epoch, r_name, warped_list_val[y], sample_r_path)
                        # save_nii_any(epoch,r_name,warped_list_val[y],sample_r_path)

                    torch.save(model.state_dict(),
                               os.path.join(model_save_path, modal_name + "_" + str(epoch) + ".pth"))

    return
