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