import numpy as np
import yaml
import pickle


def print_latex_metrics(metrics, ranks=[1, 2, 3, 5, 10], keys=None, m2m=True, m2t=True, MedR=True, fid=True):
    rank_vals = [str(x).zfill(2) for x in ranks]

    if keys is None:
        m2m_keys = [f"m2m/R{i}" for i in rank_vals]
        if MedR:
            m2m_keys += ["m2m/MedR"]
        m2t_keys = [f"m2t/R{i}" for i in rank_vals]
        if MedR:
            m2t_keys += ["m2t/MedR"]

        keys = []
        if m2m:
            keys += m2m_keys
        if m2t:
            keys += m2t_keys

        keys += ["vocab/len"]

        if fid:
            keys += ["FID"]

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val
    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    if "m2m/len" in metrics:
        print("Number of samples: {}".format(int(metrics["m2m/len"])))
    else:
        print("Number of samples: {}".format(int(metrics["m2t/len"])))
    print(str_)

    return str_


def save_metric(path, metrics, metric_str=None):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    if metric_str is not None:
        strings += f"\n{metric_str}"
    with open(path, "w") as f:
        f.write(strings)


def all_contrastive_metrics_motion_to_gt_motions(
    sims, texts, rounding=2, return_cols=False
):

    text_arr = np.array([texts])
    text_equality_matrix = (text_arr == text_arr.T)

    m2m, m2m_cols, m2m_gt_cols = contrastive_metrics(
        sims, text_equality_matrix, 0.99, return_cols=True, rounding=rounding, return_gt_cols=True
    )

    all_m = {}
    for key in m2m:
        all_m[f"m2m/{key}"] = m2m[key]
    all_m["m2m/len"] = float(len(sims))
    if return_cols:
        return all_m, m2m_cols, m2m_gt_cols
    return all_m


def all_contrastive_metrics_motion_to_text(
    sims, texts, unique_texts, rounding=2, return_cols=False
):
    text_arr= np.array([texts])
    unique_texts_arr = np.array([unique_texts])

    text_gt_matrix = 1*(unique_texts_arr == text_arr.T)

    m2t, m2t_cols, m2t_gt_cols = contrastive_metrics(
            sims, text_gt_matrix, 0.99, return_cols=True, rounding=rounding, return_gt_cols=True
            )

    all_m = {}
    for key in m2t:
        all_m[f"m2t/{key}"] = m2t[key]
    all_m["m2t/len"] = float(len(sims))
    if return_cols:
        return all_m, m2t_cols, m2t_gt_cols
    return all_m


def all_contrastive_metrics(
    sims, emb=None, threshold=None, rounding=2, return_cols=False
):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T
    
    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    
    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    return_gt_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    #assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    
    if text_selfsim is not None and threshold is not None:
        #real_threshold = 2 * threshold - 1
        real_threshold = threshold
        idx = np.argwhere(text_selfsim >= real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]
    else:
        gt_dists = np.diag(dists)[:, None]
    
    gt_cols = np.where(dists == gt_dists)[1]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT
    
    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    try:
        assert cols.size == num_queries, msg
    except:
        print("cols.size : ", cols.size)
        print("num_queries: ", num_queries)
        with open("/linkhome/rech/genlgm01/ujv31bi/TMR_bobsl/slurm/logs/train/cols.pickle", "wb") as f:
            pickle.dump(cols, f)
        with open("/linkhome/rech/genlgm01/ujv31bi/TMR_bobsl/slurm/logs/train/num_queries.pickle", "wb") as f:
            pickle.dump(num_queries, f)
    
    if return_cols:
        if return_gt_cols:
            return cols2metrics(cols, num_queries, rounding=rounding), cols, gt_cols
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def contrastive_metrics_m2t_action_retrieval(
    sims,
    motion_cat_idx,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
    norm_metrics=True
):
    n, m = sims.shape
    num_queries = n
    
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx==i]
            cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))

        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(float(np.mean([elt[k] for elt in cat_metrics])), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)
    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval(
    sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):

    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m


def contrastive_metrics_m2t_action_retrieval_multi_labels(
    sims,
    motion_cat_idx,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
    norm_metrics=True
):
    #motion_cat_idx = [[idx1, ..], [idx1], [..], ...]
    n, m = sims.shape
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    # gt_dists = dists[range(n), motion_cat_idx]
    #gt_dists = gt_dists[:, None]

    motion_cat_idx = [cat_idx[np.argmin([dists[i, elt] for elt in cat_idx])] for i, cat_idx in enumerate(motion_cat_idx)]
    #for i, cat_idx in enumerate(motion_cat_idx):
    #    i1, i2 = cat_idx
    #    j = np.argmin([dists[i, i1], dists[i, i2]])
    #    cat = cat_idx[j]
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    # DEBUG THIS PATH ##START HERE
    #idx = np.array([[i, ind] for i, ind_l in enumerate(motion_cat_idx) for ind in ind_l])
    #partition = np.cumsum([len(elt) for elt in motion_cat_idx])
    #partition = np.concatenate(([0], partition))

    #gt_dists = np.minimum.reduceat(np.concatenate([dists[tuple(idx.T)], [-1]]), partition)[:-1]
    #gt_dists = gt_dists[:, None]
    ##END HERE

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg
    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx==i]
            if len(cols_cat) > 0:
                cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))

        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(float(np.mean([elt[k] for elt in cat_metrics])), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)

    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval_multi_labels(
    sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):

    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval_multi_labels(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols

def cols2metrics(cols, num_queries=None, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]

    if num_queries is None:
        num_queries = len(cols)
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics


# from action2motion

# from https://github.com/Mathux/ACTOR/blob/master/src/evaluate/action2motion/evaluate.py
def nearest_psd(matrix):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 0
    
    # Reconstruct the matrix with the modified eigenvalues
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return psd_matrix

def calculate_activation_statistics(activations, normalize=False):
    activations = activations.cpu().numpy()
    if normalize:
        activations = activations / np.linalg.norm(activations, axis=-1)[:, None]
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    #if len(activations) == 1:
    #    sigma = np.zeros(activations.shape[1])
    #    return mu, sigma
    return mu, nearest_psd(sigma)


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    from scipy import linalg
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    #covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    tr_covmean = np.trace(covmean)

    # Numerical error might give slight imaginary component
    #if np.iscomplexobj(covmean):
    if np.iscomplexobj(tr_covmean):
        #if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #if not np.allclose(tr_covmean.imag, 0, atol=1e-3):
        #    #m = np.max(np.abs(covmean.imag))
        #    m = tr_covmean.imag
        #    raise ValueError('Imaginary component {}'.format(m))
        #covmean = covmean.real
        tr_covmean = tr_covmean.real
    #tr_covmean = np.trace(covmean)
    #ev = np.linalg.eig(nearest_psd(sigma1) @ nearest_psd(sigma2))[0]
    #ev_real = ev.real
    #ev_real[abs(ev_real) < 0.000001] = 0
    #tr_covmean = np.sum(np.sqrt(ev_real))

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def retrieval(sims, n=1, keyids=None):
    dists = - sims
    ranked_indices = np.argsort(dists)
    ranked_indices = ranked_indices[:, :n]
    if keyids is not None:
        ranked_indices  = np.array(keyids)[ranked_indices]
    return ranked_indices
