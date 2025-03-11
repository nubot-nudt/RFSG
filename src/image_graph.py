from torch.nn.modules.module import Module
from src.image_object_graph_moudle import ImageGraph
from PIL import Image
import torchvision.transforms as transforms

"""define test demo"""
class Image_Graph_Net(Module):
    def __init__(self, max_nodes):
        super(Image_Graph_Net, self).__init__()
        self.I_Graph = ImageGraph(max_nodes)
        self.img_num = 0
    def forward(self, x, conds):
        if self.img_num>= 16:
            x_name = 'img_17'
            conds_name = 'img_17'
        else:
            x_name = 'img_'+str(self.img_num)
            self.img_num += 1
            conds_name = 'img_'+str(self.img_num)
            self.img_num += 1
        self.I_Graph.add_image(x_name, x)
        self.I_Graph.add_image(conds_name, conds)

        adjacency_matrix = self.I_Graph.get_adjacency_matrix()
        img_feature = self.I_Graph.get_global_img_feature()

        img_feature = img_feature.unsqueeze(0)
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
        #
        # out = self.gcn(img_feature,adjacency_matrix)

        return img_feature, adjacency_matrix
def get_transform_pro():
    """define test demo"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
if __name__ == '__main__':
    get_transform = get_transform_pro()
    image1 = Image.open('1.png').convert('RGB')
    image1 = get_transform(image1).unsqueeze(0)

    image2 = Image.open('3.png').convert('RGB')
    image2 = get_transform(image2).unsqueeze(0)

    My_graph = Image_Graph_Net(1024, 1024)
    out = My_graph(image1, image2)

    print(out.shape, My_graph.I_Graph.get_global_img_feature())

