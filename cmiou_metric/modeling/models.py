import torch
import torchvision.models as models
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Conv2d(2048, num_classes, 1)

    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        return self.classifier(features)

def load_pretrained_model(model_path, num_classes):
    model = SegmentationModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        return torch.argmax(output, dim=1).squeeze().numpy()