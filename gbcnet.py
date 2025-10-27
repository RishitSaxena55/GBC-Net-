from imports import *
from mssop_classifier import *

class GBCNet(nn.Module):
    """
    GBCNet for Thyroid Nodule Classification
    Binary classification: Benign (0) vs Malignant (1)
    """
    def __init__(self, roi_size=7, out_ch=128, Ho=64, Wo=64,
                 in_ch=256, num_classes=3):
        super().__init__()

        # Frozen Faster R-CNN for nodule localization
        self.roi_model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        for param in self.roi_model.parameters():
            param.requires_grad = False
        self.roi_model.eval()

        # Trainable MS-SoP Classifier
        self.mssop_classifier = MSSOPClassifier(
            in_ch=in_ch,
            out_ch=out_ch,
            H=roi_size,
            W=roi_size,
            Ho=Ho,
            Wo=Wo
        )

        # Classification head for thyroid
        self.cls_head = nn.Sequential(
            nn.Linear(out_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.roi_size = roi_size
        self.num_classes = num_classes

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            # Training mode
            with torch.no_grad():
                if isinstance(x, list):
                    x_stacked = torch.stack(x)
                else:
                    x_stacked = x

                features = self.roi_model.backbone(x_stacked)
                if isinstance(features, dict):
                    features = features['0']

            roi_list = [t['boxes'] for t in targets]
            labels = torch.cat([t['labels'] for t in targets])

            pooled = roi_pool(
                features, roi_list,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=0.25
            )

            roi_features = self.mssop_classifier(pooled)
            logits = self.cls_head(roi_features)
            loss = nn.CrossEntropyLoss()(logits, labels)

            return {'loss_classifier': loss}

        else:
            # Inference mode
            self.roi_model.eval()

            with torch.no_grad():
                detections = self.roi_model(x)

                if isinstance(x, list):
                    x_stacked = torch.stack(x)
                else:
                    x_stacked = x

                features = self.roi_model.backbone(x_stacked)
                if isinstance(features, dict):
                    features = features['0']

            roi_list = [d['boxes'] for d in detections]
            total_rois = sum(len(boxes) for boxes in roi_list)

            if total_rois == 0:
                return detections

            pooled = roi_pool(
                features, roi_list,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=0.25
            )

            roi_features = self.mssop_classifier(pooled)
            logits = self.cls_head(roi_features)
            probs = torch.softmax(logits, dim=1)

            start_idx = 0
            for det in detections:
                num_boxes = len(det['boxes'])
                if num_boxes > 0:
                    det['thyroid_classes'] = torch.argmax(
                        probs[start_idx:start_idx+num_boxes], dim=1
                    )
                    det['thyroid_probs'] = probs[start_idx:start_idx+num_boxes]
                    start_idx += num_boxes

            return detections