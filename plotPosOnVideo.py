'excel file 형태로 저장된 Joint pos를 비디오에 plotting'

import cv2
from dragon import dragonV

excel_root = 'C:/Users/admin/Desktop/Philadelphia/excel/pp08/pp008_omc_gait2_front.mp4/'
excel_idx0 = excel_root + 'idx0.xlsx'
excel_idx1 = excel_root + 'idx1.xlsx'

video_path = 'C:/Users/admin/Desktop/project/video/pp08/pp008_omc_gait2_front.mp4'

frame_data_list = dragonV.xlsx2data(excel_idx0)

dragonV.play_marked_position_from_video(frame_data_list, video_path, 'pp008_omc_gait2_off_front.mp4')