import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


import numpy as np
import torch


from src.resnet import ResNet, BasicBlock
class PreResNet(ResNet):
    def forward(self, x):
        x = self.stem(x)
        for l in self.layers:
            x = l(x)
        return x
from src.Yolo.models.common import DetectMultiBackend
from src.Yolo.utils.general import non_max_suppression
from src.Yolo.utils.segment.general import process_mask


class MLP_position(nn.Module):
    def __init__(self, in_c, out_c):
        super(MLP_position, self).__init__()
        self.pos = nn.Linear(in_c, out_c)

    def forward(self, x):
        return self.pos(x)

class ImageGraph(object):
    def __init__(self, max_nodes, model_name='faster_resnet9'):
        self.graph = {}
        self.model = self.load_model(model_name)
        for param_shuffle in self.model.parameters():
            param_shuffle.requires_grad = False
        self.Yolo_seg = DetectMultiBackend(weights='./yolov5s-seg.pt', device=torch.device('cuda:0'), dnn=False, data='./coco128.yaml',
                                           fp16=False)
        self.Yolo_seg.model.cuda()
        self.Yolo_seg.eval()
        for param_Yolo in self.model.parameters():
            param_Yolo.requires_grad = False
        self.position_emb = MLP_position(6, 32).cuda()
        self.image_features = {}
        self.max_nodes = max_nodes

    def load_model(self, model_name):
        """load model"""
        model = PreResNet(3,32,16,BasicBlock,[1,1,1,1]).cuda()
        model.load_state_dict(torch.load('faster_resnet9.pth'))
        model.eval()
        return model
    def Yolo_object_feature(self, in_image):
        pred, proto = self.Yolo_seg(in_image, augment=False, visualize=False)[:2]
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000, nm=32)
        det = pred[0]
        if len(det):
            masks = process_mask(proto[0], det[:, 6:], det[:, :4] // 4, proto.shape[2:], upsample=True)  # HWC
            det[:, :4] = det[:, :4] / 128.
            det[:, 5:6] = torch.log(det[:, 5:6] + 2.)
            out_mask = masks.chunk(masks.unsqueeze(1).shape[0], dim=0)
            result = [x * proto for x in out_mask]
            result = torch.cat(result, dim=0).mean([2, 3])
            pos = self.position_emb(det[:, :6])
            out = result + pos
        else:
            out = proto.mean([2, 3])
        return torch.mean(out, dim=0, keepdim=False)
    def extract_feature(self, image):
        """extract feature from image"""
        with torch.no_grad():
            feature = self.model(image).mean([2, 3]).squeeze()
            object_feature = self.Yolo_object_feature(image)
        return torch.cat([feature, object_feature], dim=0)

    def add_image(self, image_id, image_path):
        """add new image to graph"""
        if len(self.image_features) >= self.max_nodes:
            current2all_similarity_id, current2all_similarity_value, out_feature = self.calculate_input_similarity(image_path)
            max_value = max(current2all_similarity_value)
            max_value_index = current2all_similarity_value.index(max_value)
            self.replace_node_features(current2all_similarity_id[max_value_index], out_feature)
        else:
            if image_id not in self.graph:
                self.graph[image_id] = {}
            feature = self.extract_feature(image_path)
            self.image_features[image_id] = feature
            self.update_connections(image_id)

    def update_connections(self, new_image_id):
        """update graph connect relation"""
        for existing_image_id in self.graph.keys():
            if existing_image_id != new_image_id:
                similarity = self.calculate_similarity(new_image_id, existing_image_id)
                self.graph[new_image_id][existing_image_id] = similarity
                self.graph[existing_image_id][new_image_id] = similarity

    def calculate_similarity(self, image_id1, image_id2):
        """calculate the similarity for two images"""
        feature1 = self.image_features[image_id1]
        feature2 = self.image_features[image_id2]

        return torch.mean(torch.cosine_similarity(feature1, feature2, dim=0), dim=0)


    def get_adjacency_matrix(self):
        """get adjacency matrix"""
        num_nodes = len(self.graph)
        adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        I_G = torch.eye(num_nodes)
        nodes = list(self.graph.keys())
        for i, node in enumerate(nodes):
            for neighbor in self.graph[node].keys():
                j = nodes.index(neighbor)
                # print(self.graph)
                adjacency_matrix[i][j] = self.graph[node][neighbor]
                adjacency_matrix[j][i] = self.graph[node][neighbor]  # 无向图对称填充
        return adjacency_matrix+I_G

    def get_global_img_feature(self):
        concatenated_features = torch.stack(list(self.image_features.values()), dim=0).squeeze()
        # print(concatenated_features.shape)
        return concatenated_features

    def calculate_input_similarity(self, input_image_path):
        """calculate the similarity between image and other node"""
        input_feature = self.extract_feature(input_image_path)
        similarities_id = []
        similarities_value = []

        for image_id, feature in self.image_features.items():
            similarity = torch.mean(torch.cosine_similarity(input_feature, feature, dim=0), dim=0)
            similarities_id.append(image_id)
            similarities_value.append(similarity)
        return similarities_id, similarities_value, input_feature
    def replace_node_features(self, image_id, new_feature):
        if image_id in self.image_features:
            self.image_features[image_id] = new_feature
            for neighbor in self.graph[image_id]:
                similarity = self.calculate_similarity(image_id, neighbor)
                self.graph[image_id][neighbor] = similarity
                self.graph[neighbor][image_id] = similarity
        else:
            raise ValueError("Image ID not found in the graph")


if __name__ == '__main__':
    # test demo
    image_graph = ImageGraph()
    image_graph.add_image('img1', '1.png')
    image_graph.add_image('img2', '2.png')
    image_graph.add_image('img3', '3.png')
    # add image to graph
    # ...

    adjacency_matrix = image_graph.get_adjacency_matrix()
    print("adjacency matrix:\n", adjacency_matrix.shape, adjacency_matrix)
    img_feature = image_graph.get_global_img_feature()
    print("image feature:\n", img_feature.shape, img_feature)

    from GCN_layer import GCN
    gcn_net = GCN(1024, 1024)
    img_feature_input = torch.tensor(img_feature).unsqueeze(0)
    adjacency_matrix_input = torch.tensor(adjacency_matrix).unsqueeze(0)
    out = gcn_net(img_feature_input, adjacency_matrix_input)
    print(out.shape, out)
