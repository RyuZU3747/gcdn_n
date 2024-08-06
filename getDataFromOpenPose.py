'''
윈도우 기준으로 작성되었습니다.(경로 표기방식이 운영체제마다 차이가 있기 때문.)
video -> openpose -> json file
sepcial thanks : Jingeun Lee
'''
import os

openpose_path = 'C:\\Users\\admin\\Desktop\\project\\openpose-1.7.0\\openpose\\bin\\OpenPoseDemo.exe'
json_dir_path = 'C:\\Users\\admin\\Desktop\\project\\openpose-1.7.0\\openpose\\json\\'
json_dir_name = 'walk_turn_rear\\'

video_dir_path = 'C:\\Users\\admin\\Desktop\\project\\video\\'
video_name = 'walk_turn_rear.mp4'

#동영상 이름 목록 가져오
video_list = os.listdir(video_dir_path)
print("파일 목록:")
for file_name in video_list:
    print("done : " + file_name)


cnt = 0
length = len(video_list)
for video_name in video_list:
    cnt += 1
    json_dir_name = video_name + '\\'
    os.mkdir(json_dir_path + video_name)
    command = openpose_path + ' --video ' + video_dir_path + video_name + ' --write_json ' + json_dir_path + json_dir_name +  ' --display 0 --render_pose 0'
    os.system(command)
    print(f'[{cnt}/{length}]save done')