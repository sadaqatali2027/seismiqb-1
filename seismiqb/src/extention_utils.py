FILL_VALUE = -999999

def update_horizon(src, dst):
    """ Join one horizon to another overwriting non empty points from src.
    Alternative to merge horizons when two horizons with zero overlap are merged.
    """
    dst_matrix = dst.full_matrix 
    dst_matrix[src.full_matrix != FILL_VALUE] = src.full_matrix[src.full_matrix != FILL_VALUE]
    return Horizon(dst_matrix, dst.geometry)

def correction_condition(predicted_horizon, true_horizon):
    """ Create filter matrix by thresholding on the metric value (Now it is l1 but is to be expanded on
    support corr metrics) """
    metric, _ = HorizonMetrics([predicted_horizon, true_horizon]).compare(printer=None, hist=False)
    metric[np.isnan(metric)] = 0
    metric[metric < 2.5] = 0
    metric[metric > 2.5] = 1
    return metric.astype(np.bool) 

def subset_from_true(condition_matrix, src, dst=None):
    """ Subsest horizon on condition matrix """
    FILL_VALUE = -999999
    if not dst:
        full_matrix = np.full(src.full_matrix.shape, FILL_VALUE, np.float32)
    else:
        full_matrix = dst.full_matrix
    full_matrix[condition_matrix] = src.full_matrix[condition_matrix]
    corrected_horizon = Horizon(full_matrix, src.geometry)
    return corrected_horizon


def extenf_cycle():
    internal_start_time = time.time()
    while(True):
        grid_start = time.time()
        test_ds.make_expand_grid(test_ds.indices[0], crop_shape=[1, 64, 64], stride=30,
                                 labels_img=test_ds.predicted_labels[test_ds.indices[0]][0].boundaries_matrix,
                                 labels_src='predicted_labels', batch_size=512)

        times['grid'].append(time.time() - grid_start)

        predict_ppl = create_predict_ppl(model_ppl, 'xline_crops_gen', test_ds.xline_crops_info['crop_shape'], (0, 1, 2))
        predict_ppl_i = create_predict_ppl(model_ppl, 'iline_crops_gen', test_ds.iline_crops_info['crop_shape'], (1, 0, 2))

        prediction_start = time.time()
        # predict xline
        for _ in range(test_ds.xline_crops_iters):
            predict_ppl.next_batch(1, n_epochs=None)

        for _ in range(test_ds.iline_crops_iters):
            predict_ppl_i.next_batch(1, n_epochs=None)

        times['prediction'].append(time.time() - prediction_start)

        assemble_start = time.time()
        assemble_ppl = Pipeline().assemble_crops(src=predict_ppl.v("result_preds"), dst='assembled_pred',
                                             grid_info=test_ds.xline_crops_info, order=(2, 0, 1)) << test_ds
        btch = assemble_ppl.next_batch(1)

        assemble_ppl_i = Pipeline().assemble_crops(src=predict_ppl_i.v("result_preds"), dst='assembled_pred',
                                             grid_info=test_ds.iline_crops_info, order=(0, 2, 1)) << test_ds
        btch_i = assemble_ppl_i.next_batch(1)
        times['assemble'].append(time.time() - assemble_start)


        mask_to_horizon_start = time.time()
        # etract horizons
        min_point = [rng[0] for rng in test_ds.xline_crops_info['range']]
        test_ds.mask_to_horizons(btch.assembled_pred, test_ds.indices[0], min_point=min_point,
                                 threshold=0.5, dst='xl_predict', minsize=5)

        # extract horizons
        min_point = [rng[0] for rng in test_ds.iline_crops_info['range']]
        test_ds.mask_to_horizons(btch_i.assembled_pred, test_ds.indices[0], min_point=min_point,
                                 threshold=0.5, dst='il_predict', minsize=5)
        times['mask_to_horizon'].append(time.time() - mask_to_horizon_start)

        merge_start = time.time()
        # merge il and xl prediction
        merge_candidates = [*test_ds.xl_predict[test_ds.indices[0]],
                            *test_ds.il_predict[test_ds.indices[0]]]
        test_ds.merge_horizons(merge_candidates, mean_threshold=16.5, q_threshold=50.0, adjacency=5)

        max_len_arg = 0
        for i in range(len(merge_candidates)):
            if len(merge_candidates[i]) > len(merge_candidates[max_len_arg]):
                max_len_arg = i

        for i in range(len(merge_candidates)):
            update_horizon(merge_candidates[i], dst=merge_candidates[max_len_arg])

        times['merge'].append(time.time() - merge_start)

        # correction condition
        correction_start = time.time()
        cond = correction_condition(merge_candidates[max_len_arg], test_ds.labels[test_ds.indices[0]][0])

        old_priors = test_ds.prior_mask[test_ds.indices[0]][0]

        # update priors
        test_ds.prior_mask[test_ds.indices[0]][0] = subset_from_true(cond, src=test_ds.labels[test_ds.indices[0]][0],
                                                                     dst=test_ds.prior_mask[test_ds.indices[0]][0])


        # add correction
        corrected = subset_from_true(cond, src=test_ds.labels[test_ds.indices[0]][0], dst=merge_candidates[max_len_arg])
        times['correction'].append(time.time() - correction_start)

        update_start = time.time()
        test_ds.predicted_labels[test_ds.indices[0]][0] = update_horizon(corrected, test_ds.predicted_labels[test_ds.indices[0]][0])
        times['update'].append(time.time() - update_start)

        print('predicted_labels', len(test_ds.predicted_labels[test_ds.indices[0]][0]))

        show_start = time.time()
    #     plt.imshow(test_ds.predicted_labels[test_ds.indices[0]][0].boundaries_matrix)
    #     plt.show()
        times['show'].append(time.time() - show_start)

    int_time = time.time() - internal_start_time    
    print('Whole time', int_time)
