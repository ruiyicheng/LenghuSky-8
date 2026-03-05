"""
inference to obtain the results on benchmark dataset
"""

from predictor import CloudNowcaster

import numpy as np

nowcaster = CloudNowcaster(
    config_json = "config_predict.json",
    vqvae_ckpt="./checkpoints/vqvae/vqvae-epoch=00-val_loss=0.0036.ckpt",
    gpt_ckpt="./checkpoints/gpt/gpt-epoch=00-val_loss=1.0759.ckpt"
)
out_path = "/home/dzm/PycharmProjects/conv_lstm_precipitation/benchmark_results/"

def _load_frame_npz(path_npz: str) -> np.ndarray:
    """
    Loads the first 2D/3D array found in an .npz and converts to HxWxC.
    Normalizes to [0,1] then shifts to [-0.5,0.5]
    """
    arr = None
    with np.load(path_npz) as data:
        for k in data.files:
            a = data[k]
            if a.ndim >= 2:
                arr = a
                break


    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)  # H,W -> H,W,3
    elif arr.ndim == 3:
        # unify to H,W,C
        if arr.shape[0] in (1, 3, 4):  # C,H,W -> H,W,C
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]

    # resize to 64x64
    # print(arr)
    # # arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    # arr = arr.astype(np.float32)
    # if arr.max() > 1.0 or arr.min() < 0.0:
    #     arr = np.clip(arr, 0, 255) / 255.0
    # arr = arr - 0.5
    # max_arr = np.max(arr)
    # min_arr = np.min(arr)
    # delta_arr = max_arr - min_arr
    # arr = (arr - min_arr) / delta_arr - 0.5
    m = np.mean(arr)
    s = np.std(arr)
    arr = (arr - m) / s / 6
    return arr  # H,W,3 in [-0.5,0.5]



logits_home = "/home/dzm/cloud/logits/"
map_dir = "/home/dzm/PycharmProjects/conv_lstm_precipitation/simplified_VideoGPT/aux_data/bkg_map.txt"

samples = 30
target_npz_list = []
with open(map_dir,"r") as f:
    lines = f.readlines()
    ct = 0
    for l in lines:
        ct +=1
        if ct < 400000:
            continue
        target_npz_list.append(logits_home + l.split(',')[0])
        if len(target_npz_list) >= samples:
            break


print(target_npz_list)

dataset = []
for npz_file in target_npz_list:
    res = _load_frame_npz(npz_file)
    dataset.append(res)

forward_reading = 7
dataset = np.array(dataset)
pd_list = []
for i in range(len(dataset)-forward_reading-1):
    past = dataset[i:i+forward_reading,...]
    pred = nowcaster.predict_next(past_frames=past,m_future = 1,temperature = 1, top_k = None, top_p = None)
    pd_list.append(np.squeeze(pred))
pd_list = np.array(pd_list)
ground_truth = dataset[forward_reading+1:]
naive_baseline = dataset[forward_reading:-1]
print(pd_list.shape, dataset[forward_reading+1:].shape)



# /home/dzm/PycharmProjects/conv_lstm_precipitation/benchmark_prediction.py use segmentation_metrics to calculate metrics
import sys
sys.path.append("/home/dzm/PycharmProjects/conv_lstm_precipitation/")

from code.baseline_and_benchmark.nowcast_baseline.benchmark_prediction import benchmark_prediction

# metrics = segmentation_metrics(ground_truth, ground_truth, pd_list, config["class_map"])
# print(metrics)

def wrapper_VideoGPT(data,mask):
    mask = mask.reshape(-1,1,64,64)
    data = data * mask
    m = np.mean(data)
    s = np.std(data)
    data = (data - m) / s / 6
    # data: [T,C,H,W] -> past: [C,T_past,H,W]
    past = np.transpose(data,(1,0,2,3))
    pred = nowcaster.predict_next(past_frames=past,m_future = 1,temperature = 1, top_k = None, top_p = None)
    pred = np.squeeze(pred)
    pred = pred * s * 6 + m
    # reshape 64 64 3 -> 3 64 64
    pred = np.transpose(pred,(2,0,1))
    return pred



# def wrapper_Naive(data,mask):
#     return data[-1,...]
# metric_naive = benchmark_prediction(wrapper_Naive,n_step = 7)
# print(metric_naive)
# outpath = out_path + "Trivial_metrics.json"
# with open(outpath,"w") as f:
#     import json
#     json.dump(metric_naive,f)

metric_GPT = benchmark_prediction(wrapper_VideoGPT,n_step = 1)
print(metric_GPT)
outpath = out_path + "VideoGPT-1_metrics.json"
with open(outpath,"w") as f:
    import json
    json.dump(metric_GPT,f)





import matplotlib.pyplot as plt

# Define color mapping for indices 0, 1, 2
color_map = {
    0: [0, 114, 178],  # RGB values
    1: [230, 159, 0],
    2: [204, 121, 167]
}

# Normalize colors to [0, 1] range for matplotlib
color_map_normalized = {k: [c / 255.0 for c in v] for k, v in color_map.items()}

# Convert logits to class indices by taking argmax along channel dimension
ground_truth_indices = np.argmax(ground_truth, axis=-1)
pd_indices = np.argmax(pd_list, axis=-1)
naive_baseline_indices = np.argmax(naive_baseline, axis=-1)
# Create visualization
for i in range(len(pd_list)):
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # Create RGB images from indices
    gt_rgb = np.zeros(ground_truth_indices[i].shape + (3,))
    pd_rgb = np.zeros(pd_indices[i].shape + (3,))
    nv_rgb = np.zeros(naive_baseline_indices[i].shape + (3,))
    for idx, color in color_map_normalized.items():
        gt_rgb[ground_truth_indices[i] == idx] = color
        pd_rgb[pd_indices[i] == idx] = color
        nv_rgb[naive_baseline_indices[i] == idx] = color

    # Plot ground truth
    ax1.imshow(gt_rgb)
    ax1.set_title(f'Ground Truth - Frame {i + 3}')
    ax1.axis('off')

    # Plot prediction
    ax2.imshow(pd_rgb)
    ax2.set_title(f'Prediction - Frame {i + 3}')
    ax2.axis('off')

    ax3.imshow(nv_rgb)
    ax3.set_title(f'Naive - Frame {i + 3}')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()