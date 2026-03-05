import numpy as np
import datetime
test_config = {
    "period":[
        ["2023-01-01-00-00-00", "2023-09-28-00-00-00"],
        ["2025-01-01-00-00-00","2025-11-01-00-00-00"]
      ],
    #  "period":[ # for debugging
    #      ["2023-01-01-00-00-00", "2023-01-08-00-00-00"],
    #      ["2025-01-01-00-00-00","2025-01-10-00-00-00"]
    #    ],
    "logits_dir": "/home/dzm/cloud/logits",
    "mask_dir": "/home/dzm/PycharmProjects/conv_lstm_precipitation/simplified_VideoGPT/aux_data/mask_mat",
    "mapping_txt": "/home/dzm/PycharmProjects/conv_lstm_precipitation/simplified_VideoGPT/aux_data/bkg_map.txt",
    "class_map":{
        0:"sky",
        1:"cloud",
        2:"contamination"
    }
}



# Example of mapping_txt:

# 2018-05-01-00-02-44_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-07-56_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-13-08_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-18-20_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-23-32_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-28-45_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-33-57_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-39-09_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-44-21_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-49-33_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-54-45_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-00-59-58_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-05-10_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-10-22_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-15-34_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-20-46_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-25-58_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-31-11_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-36-23_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-41-35_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-46-47_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-51-59_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-01-57-11_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-02-24_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-07-36_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-12-48_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-18-00_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-23-12_logits.npz,mask_2018-05-01-00-02-44.npy
# 2018-05-01-02-28-24_logits.npz,mask_2018-05-01-00-02-44.npy
import tqdm
def get_time_list(config):
    res_list = []
    mask_data = {}
    
    with open(config["mapping_txt"], "r") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            try:
                logit_file, mask_file = line.strip().split(",")
                date_str = logit_file.replace("_logits.npz", "").split("_")[0]
                date_time = resolve_str_to_datetime(date_str)
                if mask_file not in mask_data:
                    mask_data[mask_file] = np.load(f"{config['mask_dir']}/{mask_file}")
                for period in config["period"]:
                    start_dt = resolve_str_to_datetime(period[0])
                    end_dt = resolve_str_to_datetime(period[1])

                    if start_dt <= date_time <= end_dt:
                        data = np.load(f"{config['logits_dir']}/{logit_file}")['logits_grid']
                    # 64 64 3 -> 3 64 64
                        data = np.transpose(data, (2,0,1))
                        res_list.append((date_str, logit_file, mask_file,data))
                        break
            except Exception as e:
                print(f"Skipping line due to error: {line.strip()} | Error: {e}")
    return res_list,mask_data

def resolve_str_to_datetime(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
    except Exception as e:
        print(f"Error parsing date string '{date_str}': {e}")
        return None

def yield_logits_and_mask(all_res_list, mask_data, config, n_step=2, overlap=False):
    i = 0
    print(len(all_res_list))
    while 1 :
        print(len(all_res_list),i)
        if i + n_step + 1 > len(all_res_list):
            break
        res = {"input_logits": [], "target_logits": None,"target_mask": None, "input_mask": []}
        for ti in range(i,i + n_step):
            date_str, logit_file, mask_file, data = all_res_list[ti]
            res["input_logits"].append(data)
            res["input_mask"].append(mask_data[mask_file])
        date_str, logit_file, mask_file, data = all_res_list[i + n_step]
        res["target_logits"] = data
        res["target_mask"] = mask_data[mask_file]
        yield res
        if not overlap:
            i += n_step + 1
        else:
            i += 1

def segmentation_metrics(mask, target_logits, pred_logits, class_map):
    """
    Compute segmentation metrics over a masked region.

    Args:
        mask: np.ndarray of shape (B, H, W). >0 means valid pixel.
        target_logits: np.ndarray of shape (B, C, H, W)
        pred_logits:   np.ndarray of shape (B, C, H, W)
        class_map: dict {class_idx: "class_name"}

    Returns:
        dict with:
          - overall_accuracy
          - mean_cross_entropy
          - mIoU
          - precision: {"macro": float, "per_class": {idx: {"name": str, "value": float}}}
          - recall:    {"macro": float, "per_class": {idx: {"name": str, "value": float}}}
          - f1:        {"macro": float, "per_class": {idx: {"name": str, "value": float}}}
    Notes:
      * Solid labels are obtained via argmax on logits.
      * Macro averages ignore classes where the metric is undefined (nanmean).
      * mean_cross_entropy uses hard target labels (argmax of target_logits).
    """
    # Ensure float for numeric stability
    target_logits = np.asarray(target_logits, dtype=np.float64)
    pred_logits   = np.asarray(pred_logits,   dtype=np.float64)
    mask          = np.asarray(mask)

    if target_logits.shape != pred_logits.shape:
        raise ValueError("target_logits and pred_logits must have the same shape (B, C, H, W).")
    if mask.shape != target_logits.shape[0:1] + target_logits.shape[2:4]:
        raise ValueError("mask must have shape (B, H, W) matching logits.")

    B, C, H, W = pred_logits.shape

    # Boolean mask
    mask_bool = mask.astype(bool)
    num_valid = int(mask_bool.sum())

    # Solid labels from logits (argmax over classes)
    y_true = target_logits.argmax(axis=1)  # (B, H, W)
    y_pred = pred_logits.argmax(axis=1)    # (B, H, W)

    # Flatten masked positions
    y_true_flat = y_true[mask_bool].ravel()
    y_pred_flat = y_pred[mask_bool].ravel()

    # Helper to convert numpy floats to plain Python floats; nan -> None
    def _to_float(x):
        xf = float(x)
        return None if np.isnan(xf) else xf

    # If no valid pixels, return N/A metrics
    if num_valid == 0:
        per_class_empty = {str(i): {"name": class_map.get(i, f"class_{i}"), "value": None} for i in range(C)}
        return {
            "overall_accuracy": None,
            "mean_cross_entropy": None,
            "mIoU": None,
            "precision": {"macro": None, "per_class": per_class_empty},
            "recall":    {"macro": None, "per_class": per_class_empty},
            "f1":        {"macro": None, "per_class": per_class_empty},
        }

    # Confusion matrix over masked pixels
    cm = np.bincount(C * y_true_flat + y_pred_flat, minlength=C * C).reshape(C, C)

    # Per-class counts
    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    denom_prec = TP + FP
    denom_rec  = TP + FN
    denom_iou  = TP + FP + FN

    # Safe divisions with NaN when undefined
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_k = TP / denom_prec
        rec_k  = TP / denom_rec
        f1_k   = 2 * prec_k * rec_k / (prec_k + rec_k)
        iou_k  = TP / denom_iou

    # Macro averages: ignore undefined (nanmean)
    macro_prec = np.nanmean(prec_k)
    macro_rec  = np.nanmean(rec_k)
    macro_f1   = np.nanmean(f1_k)
    mIoU       = np.nanmean(iou_k)

    # Overall accuracy over masked region
    overall_acc = float((y_true_flat == y_pred_flat).mean())

    # Mean cross-entropy over masked region (hard labels from target argmax)
    # Compute log-softmax in a numerically stable way
    pred_move = np.moveaxis(pred_logits, 1, -1)       # (B, H, W, C)
    pred_flat = pred_move.reshape(-1, C)              # (B*H*W, C)
    mask_flat = mask_bool.reshape(-1)
    pred_masked = pred_flat[mask_flat]                # (N, C)
    y_masked = y_true_flat                            # (N,)

    # logsumexp
    m = pred_masked.max(axis=1)
    lse = m + np.log(np.exp(pred_masked - m[:, None]).sum(axis=1))
    logp_y = pred_masked[np.arange(pred_masked.shape[0]), y_masked] - lse
    mean_ce = float((-logp_y).mean())

    # Build per-class sections
    def _per_class_dict(values):
        return {
            str(i): {"name": class_map.get(i, f"class_{i}"), "value": _to_float(values[i])}
            for i in range(C)
        }

    result = {
        "overall_accuracy": _to_float(overall_acc),
        "mean_cross_entropy": _to_float(mean_ce),
        "mIoU": _to_float(mIoU),
        "precision": {
            "macro": _to_float(macro_prec),
            "per_class": _per_class_dict(prec_k),
        },
        "recall": {
            "macro": _to_float(macro_rec),
            "per_class": _per_class_dict(rec_k),
        },
        "f1": {
            "macro": _to_float(macro_f1),
            "per_class": _per_class_dict(f1_k),
        },
    }
    return result


def benchmark_prediction(f,config = test_config, n_step = 2):
    data, mask_data = get_time_list(config)
    y_logits_list = []
    pred_logits_list = []
    mask_list = []
    ct = 0
    all_length = len(data) // (n_step + 1)
    for res in yield_logits_and_mask(data, mask_data, config, n_step=n_step, overlap=False):
        ct += 1
        #print(f"Processing sample {ct}/{all_length}")
        
        input_logits = np.array(res["input_logits"])  # (n_step, C, H, W)
        #print(input_logits.shape)
        target_logits = res["target_logits"]          # (C, H, W)
        #print(target_logits.shape)
        input_mask = np.array(res["input_mask"])      # (n_step, H, W)
        #print(input_mask.shape)
        target_mask = res["target_mask"]              # (H, W)

        pred_logits = f(input_logits, input_mask)    # (C, H, W)

        y_logits_list.append(target_logits)
        pred_logits_list.append(pred_logits)
        mask_list.append(target_mask)
    # calculate metrics
    y_logits_array =  np.array(y_logits_list)        # (N, C, H, W)
    pred_logits_array = np.array(pred_logits_list)   # (N, C, H, W)
    mask_array = np.array(mask_list)                 # (N, H, W)
    print(mask_array.shape, y_logits_array.shape, pred_logits_array.shape)
    metrics = segmentation_metrics(mask_array, y_logits_array, pred_logits_array, config["class_map"])
    return metrics
