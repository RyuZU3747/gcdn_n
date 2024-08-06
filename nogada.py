from dragon import dragonV

root_path = '/Users/ivory/Documents/github/Philadelphia/jointAndGt/pp085/'

idx0 = root_path + 'pp085_omc_walk_turn_rear.mp4/idx0.xlsx'
idx1 = root_path + 'pp085_omc_walk_turn_rear.mp4/idx1.xlsx'
#idx2 = root_path + 'pp086_omc_walk_turn_front.mp4/idx2.xlsx'
nogada = root_path + 'pp085_omc_walk_turn_rear.mp4/nogada.xlsx'  

video_path = '/Users/ivory/Documents/lab/verybigdata/30/pp085_omc_walk_turn_rear.mp4'

idx0_frame_list = dragonV.xlsx2data(idx0)      
idx1_frame_list = dragonV.xlsx2data(idx1)         
#idx2_frame_list = dragonV.xlsx2data(idx2)
nogada_frame_list = dragonV.xlsx2data(nogada)    

dragonV.mark_pos_on_video(video_path, nogada_frame_list, 'gait1' ,speed=1)

print(len(idx0_frame_list))
print(len(idx1_frame_list))
