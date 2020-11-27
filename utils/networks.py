import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_relu(nn.Module):
    def __init__(self, mono_feature, non_mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num = 5, non_mono_hidden_num = 5, compress_non_mono=False, normalize_regression=False):
        super(MLP_relu, self).__init__()
        self.normalize_regression = normalize_regression 
        self.compress_non_mono = compress_non_mono
        if compress_non_mono:
            self.non_mono_feature_extractor = nn.Linear(non_mono_feature, 10, bias=True)
            self.mono_fc_in = nn.Linear(mono_feature+10, mono_hidden_num, bias=True)
        else:
            self.mono_fc_in = nn.Linear(mono_feature+non_mono_feature, mono_hidden_num, bias=True)
        
        bottleneck=10
        self.non_mono_fc_in = nn.Linear(non_mono_feature, non_mono_hidden_num, bias=True)
        self.mono_submods_out = nn.ModuleList([nn.Linear(mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)]) 
        self.mono_submods_in = nn.ModuleList([nn.Linear(2*bottleneck, mono_hidden_num, bias=True) for i in range(mono_sub_num)])
        self.non_mono_submods_out = nn.ModuleList([nn.Linear(non_mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)]) 
        self.non_mono_submods_in = nn.ModuleList([nn.Linear(bottleneck, non_mono_hidden_num, bias=True) for i in range(mono_sub_num)])

        self.mono_fc_last = nn.Linear(mono_hidden_num, 1, bias=True)
        self.non_mono_fc_last = nn.Linear(non_mono_hidden_num, 1, bias=True)

    def forward(self, mono_feature, non_mono_feature):
        y = self.non_mono_fc_in(non_mono_feature)
        y = F.relu(y)
        
        if self.compress_non_mono:
            non_mono_feature = self.non_mono_feature_extractor(non_mono_feature) 
            non_mono_feature = F.hardtanh(non_mono_feature, min_val=0.0, max_val=1.0)

        x = self.mono_fc_in(torch.cat([mono_feature, non_mono_feature], dim=1))
        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            x = F.hardtanh(x, min_val=0.0, max_val=1.0)
            
            y = self.non_mono_submods_out[i](y)
            y = F.hardtanh(y, min_val=0.0, max_val=1.0)
            
            x = self.mono_submods_in[i](torch.cat([x, y], dim=1))
            x = F.relu(x)
            
            y = self.non_mono_submods_in[i](y)
            y = F.relu(y)

        x = self.mono_fc_last(x)
                        
        y = self.non_mono_fc_last(y)
         
        out = x+y 
        if self.normalize_regression:
            out = F.sigmoid(out)
        return out
    
    def reg_forward(self, feature_num, mono_num, bottleneck=10, num=512): 
        in_list = []
        out_list = []
        if self.compress_non_mono:
            input_feature = torch.rand(num, mono_num+10).cuda()
        else:
            input_feature = torch.rand(num, feature_num).cuda()
        input_mono = input_feature[:, :mono_num]
        input_non_mono = input_feature[:, mono_num:]
        input_mono.requires_grad = True

        x = self.mono_fc_in(torch.cat([input_mono, input_non_mono],dim=1)) 
        in_list.append(input_mono)
        
        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            out_list.append(x)
            
            input_feature = torch.rand(num, 2*bottleneck).cuda()
            input_mono = input_feature[:, :bottleneck]
            input_non_mono = input_feature[:, bottleneck:]
            in_list.append(input_mono)
            in_list[-1].requires_grad = True

            x = self.mono_submods_in[i](torch.cat([input_mono, input_non_mono], dim=1))
            x = F.relu(x)

        x = self.mono_fc_last(x)
        out_list.append(x) 
        
        return in_list, out_list

class InterConv(nn.Module):
    def __init__(self, layer_num=1, hidden_num = 100, class_num=10):
        super(InterConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 20, 3, 1)
        self.conv3 = nn.Conv2d(20, 50, 3, 1)
        self.img_to_vector = nn.Linear(800, class_num, bias=True)
        
        self.submods_in = nn.ModuleList([nn.Linear(class_num, hidden_num, bias=True) for i in range(layer_num)]) 
        self.submods_out = nn.ModuleList([nn.Linear(hidden_num, class_num, bias=True) for i in range(layer_num)])
        self.class_num = class_num 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 5, 5)
        
        x = x.view(-1, 4*4*50) 
        x = self.img_to_vector(x)
        for i in range(int(len(self.submods_out))):
            x = F.hardtanh(x, min_val =0.0, max_val = 1.0)
            x = self.submods_in[i](x)
            x = F.relu(x)
            x = self.submods_out[i](x)
            
        return x
    
    def inter_forward(self, x):
        out_list = []
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 5, 5)
        
        x = x.view(-1, 4*4*50) 
        
        x = self.img_to_vector(x)
        for i in range(int(len(self.submods_out))):
            x = F.hardtanh(x, min_val =0.0, max_val = 1.0)
            out_list.append(x)
            x = self.submods_in[i](x)
            x = F.relu(x)
            x = self.submods_out[i](x)

        return x, out_list

    def reg_forward(self, num=512): 
        in_list = []
        out_list = []
        
        for i in range(int(len(self.submods_out))):
            input_feature = torch.rand(num, self.class_num).cuda()
            in_list.append(input_feature)
            in_list[-1].requires_grad = True

            x = self.submods_in[i](input_feature)
            x = F.relu(x)
            x = self.submods_out[i](x)
            out_list.append(x)

        return in_list, out_list

class InterConv_Robust(nn.Module):
    def __init__(self, layer_num=1, hidden_num = 100, class_num=10):
        super(InterConv_Robust, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 20, 3, 1)
        self.conv3 = nn.Conv2d(20, 50, 3, 1)
        self.img_to_vector = nn.Linear(800, class_num, bias=True)
        
        self.submods_in = nn.ModuleList([nn.Linear(class_num, hidden_num, bias=True) for i in range(layer_num)]) 
        self.submods_out = nn.ModuleList([nn.Linear(hidden_num, class_num, bias=True) for i in range(layer_num)])
        self.class_num = class_num 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 5, 5)
        
        x = x.view(-1, 4*4*50) 
        x = self.img_to_vector(x)
        for i in range(int(len(self.submods_out))):
            x = (F.tanh(x)+1.)*0.5
            x = self.submods_in[i](x)
            x = F.relu(x)
            x = self.submods_out[i](x)
            
        return x
 
