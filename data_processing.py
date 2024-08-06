'''
윈도우 기준(경로 표기 문제 때문에, 다른 운영체제에서는 동작하지 않을 수 있음.)
video -> openpose -> json file
vs code의 터미널 위치는 openpose 에 있어야 함.
sepcial thanks : Jingeun Lee
'''

import os

# 자신의 환경에 맞게 수정해야 하는 부분
json_dir_path = 'C:/Users/admin/Desktop/project/json/pp087/'
openpose_path = 'C:/Users/admin/Desktop/project/openpose-1.7.0/openpose/bin/OpenPoseDemo.exe'
video_dir_path = 'C:/Users/admin/Downloads/30-20240305T021420Z-001/30/'


'''
아래 코드 내용은 수정 안해도 괜찮을 거예요.
'''
video_list = os.listdir(video_dir_path)
print("<video list>")
for video_name in video_list:
    print("Done : " + video_name)

cnt = 0
length = len(video_list)
for video_name in video_list:
    cnt += 1
    os.mkdir(json_dir_path + video_name)
    json_file_path = json_dir_path + video_name + '/'
    command = openpose_path + ' --video ' + video_dir_path + video_name + ' --number_people_max 2 --write_json ' + json_file_path + ' --display 0 --render_pose 0'
    os.system(command)
    print(f'[{cnt}/{length}] processed/n')

