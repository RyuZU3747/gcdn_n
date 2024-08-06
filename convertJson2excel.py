'''
json_dir에서 joint pos 추출 후, 엑셀 형태롤 저장
joint_pos idx가 2라고 가정
'''
from dragon import dragonV
import os
json_dir_path = '/Users/admin/Desktop/project/json/'
json_sub_path = ['pp087']
xlsx_dir_path = '/Users/admin/Desktop/Philadelphia/excel/'

for each_sub_path in json_sub_path:
    # each_sub_path : pp01, pp08, pp09,...
    # json_dir_path/ppxx/
    all_video_json_path = json_dir_path + each_sub_path + '/'
    # json_dir_path/ppxx/video_list
    video_list = [folder for folder in os.listdir(all_video_json_path)]

    for each_video in video_list:
        save_xlsx_path = xlsx_dir_path + each_sub_path + '/' + each_video
        #json/ppxx/video_name/
        json_path_for_each_video = json_dir_path + each_sub_path + '/' + each_video + '/'

        #xlsx/ppxx/each_video_name 
        os.mkdir(save_xlsx_path)
        
        #os.mkdir(xlsl_dir_path + each_sub_path + each_video)

        dragonV.jsons2excel(json_path_for_each_video, [], 0, 'idx0' + '.xlsx', save_xlsx_path + '/')
        dragonV.jsons2excel(json_path_for_each_video, [], 1, 'idx1' + '.xlsx', save_xlsx_path + '/')
        print(f'Done : {each_sub_path}->{each_video}')


'''
json_folder_list = [folder for folder in os.listdir(json_dir_path) if os.path.isdir(os.path.join(json_dir_path ,folder))]
xlsx_path = '/Users/ivory/Desktop/WorkSpace/Philadelphia/excel/'

#json file들을 excel 로 저장, --number_people_max 2 이기 때문에 0, 1로 저장
for each_json_folder in json_folder_list:
    json_file_list = dragonV.get_jsons_list(json_dir_path + each_json_folder + '/')

    dragonV.jsons2excel(json_dir_path + each_json_folder + '/', [], 0, each_json_folder + '+_0' +'.xlsx', xlsx_path)
    dragonV.jsons2excel(json_dir_path + each_json_folder + '/', [], 1, each_json_folder + '_1' +'.xlsx', xlsx_path)

'''
