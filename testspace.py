import cv2
from dragon import dragonV
import numpy as np

nogada_excel_path = '/Users/ivory/Documents/github/Philadelphia/jointAndGt/pp086/pp086_omc_walk_fast_rear.mp4/nogada.xlsx'
gt_path = '/Users/ivory/Documents/github/Philadelphia/jointAndGt/pp086/pp086_omc_walk_fast_rear.mp4/contact_omc_walk_fast_30fps.npy'
video_path = '/Users/ivory/Documents/lab/verybigdata/pp086/30/pp086_omc_walk_fast_rear.mp4'
'''
gt = [ ltoe, lheel, rtoe, rheel ], [ 19, 21, 22, 24 ]
'''
all_frame_data = dragonV.xlsx2data(nogada_excel_path)
frame_len = len(all_frame_data)

gt_np = np.load(gt_path)
gt_list = gt_np.tolist()

data_pair_list = dragonV.make_dataAndGtPair(all_frame_data, gt_list)

dragonV.render_result_on_video(video_path, data_pair_list, 'gait1_render')