from dragon import dragonV
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from dragon import dragonV

#define network structure
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        out = self.sigmoid(out)
        return out
    

#data_path
root_path = 'C:/Users/admin/Desktop/Philadelphia/inputs/'
npz_file_name = '1_B.npz'
video_file_name = 'openpose_render_video.mp4'

#constant
seleted_openpose_joint_idx_list = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]

#load
data = np.load(root_path + npz_file_name, encoding='latin1', allow_pickle=True)
keypoints = data['keypoints']
keypoints = np.array(keypoints, dtype=np.float32)

#convert to data properly
keypoints_list = keypoints.tolist()

total_frame_data_list = []
for each_frame in keypoints_list:
    tmp = []
    for joint_idx in range(0, len(each_frame)):
        x = each_frame[joint_idx][0]
        y = each_frame[joint_idx][1]

        tmp.append(x)
        tmp.append(y)
    
    total_frame_data_list.append(tmp)

#dragonV.mark_pos_on_video(root_path + video_file_name, total_frame_data_list, 'result')

norm_total_frame_data_list = dragonV.nomalize_data(total_frame_data_list)
norm_total_frame_lower_joint_list = dragonV.get_selected_joint_pos_frame_list(norm_total_frame_data_list, seleted_openpose_joint_idx_list)
#dragonV.mark_pos_on_video(root_path + video_file_name, total_frame_lower_joint_list, 'result')

#data formatting for the network input shape
norm_data_len = len(norm_total_frame_lower_joint_list)
trimmed_input_data = []
for idx in range(0, norm_data_len - 8):
    trimmed_input_data.append(norm_total_frame_lower_joint_list[idx:idx+9])

input_data_np = np.array(trimmed_input_data)
input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32)

#data validation check
has_nan = torch.isnan(input_data_tensor).any().item()
print("NaN 값이 있는지 여부:", has_nan)

#model load
input_size = 26
output_size = 4
hidden_size = 128


model_weights_path = 'C:/Users/admin/Desktop/Philadelphia/models/1000_epochs.pth'

model = LSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

outputs = model(input_data_tensor)

#threshold
threshold = 0.6

predicted = (outputs > threshold).int()
predicted_np = np.array(predicted)

padding_np = np.array([0, 0, 0, 0])

# 앞 4개의 프레임에 대해서 패딩
padded_predicted_np = np.vstack([padding_np] * 4 + [predicted_np])

padded_predicted_list = padded_predicted_np.tolist()

paired_frame_list = dragonV.make_dataAndGtPair(total_frame_data_list, padded_predicted_list)
dragonV.render_result_on_video(root_path + video_file_name, paired_frame_list, 'rendered video')