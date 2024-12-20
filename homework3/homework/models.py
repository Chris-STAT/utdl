from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class RegressionLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Regression loss using Mean Squared Error (MSE)

        Args:
            predictions: tensor (b,) or (b, 1) predictions
            target: tensor (b,) or (b, 1) target values

        Returns:
            tensor, scalar loss
        """
        return nn.functional.mse_loss(predictions, target)


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        #raise NotImplementedError("ClassificationLoss.forward() is not implemented")
        return nn.functional.cross_entropy(logits, target)





class ClassificationLoss_detection(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        #raise NotImplementedError("ClassificationLoss.forward() is not implemented")
        return nn.functional.cross_entropy(logits, target, weight=weight)
    





class ResBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cnn_layer_1 = torch.nn.Conv2d(in_c, out_c, 3, stride = 1, padding = 1)
        self.bn_layer_1 = torch.nn.BatchNorm2d(out_c)
        self.cnn_layer_2 = torch.nn.Conv2d(in_c, out_c, 3, stride = 1, padding = 1)
        self.bn_layer_2 = torch.nn.BatchNorm2d(out_c)
        self.act_layer= torch.nn.ReLU()

    def forward(self, x):
        x_dup = x
        x_new = self.cnn_layer_1(x)
        x_new = self.bn_layer_1(x_new)
        x_new = self.act_layer(x_new)
        x_new = self.cnn_layer_2(x_new)
        x_new = x_new + x_dup
        x_new = self.bn_layer_2(x_new)
        x_new = self.act_layer(x_new)
        return x_new


'''
class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        size = 128
        in_c = size
        out_c = size

        self.layer_1 = torch.nn.Conv2d(in_channels, out_c,3,stride=1,padding=1)
        self.layer_2 = ResBlock(in_c, out_c)
        self.layer_3 = ResBlock(in_c, out_c)
        self.layer_4 = ResBlock(in_c, out_c)
        self.layer_5 = ResBlock(in_c, out_c)
        self.layer_6 = ResBlock(in_c, out_c)
        self.dropout = torch.nn.Dropout(0.15)
        self.output = torch.nn.Linear(in_c,num_classes)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        #logits = torch.randn(x.size(0), 6)
        z = self.layer_1(z)
        z = self.layer_2(z)
        z = self.layer_3(z)
        z = self.layer_4(z)
        z = self.layer_5(z)
        z = self.layer_6(z)
        z = self.dropout(z)
        z =z.mean((2,3))
        logits = self.output(z)
        return logits



    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self.forward (x).argmax(dim=1)
'''
class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.as_tensor([0.2064, 0.1944, 0.2252]))

        size = 64
        in_c = size
        out_c = size

        # Define the layers with batch normalization
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.layer_2 = ResBlock(in_c, out_c)
        #self.layer_3 = ResBlock(in_c, out_c)
        #self.layer_4 = ResBlock(in_c, out_c)
        #self.layer_5 = ResBlock(in_c, out_c)
        #self.layer_6 = ResBlock(in_c, out_c)
        self.dropout = nn.Dropout(0.15)
        self.output = nn.Linear(in_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through the layers with batch normalization
        z = self.layer_1(z)
        z = self.layer_2(z)
        #z = self.layer_3(z)
        #z = self.layer_4(z)
        #z = self.layer_5(z)
        #z = self.layer_6(z)
        z = self.dropout(z)
        
        # Global average pooling and final fully connected layer
        z = z.mean((2, 3))  # Global average pooling over spatial dimensions
        logits = self.output(z)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self.forward(x).argmax(dim=1)



'''
class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        self.down1 = nn.Conv2d(in_channels, 16, kernel_size = 3, stride=2,  padding=1)
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.logits_layer = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        if isinstance(x, tuple):
            x = x[0]
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        #logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        #raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))

        z = self.down1(z)
        z = self.relu(z)
        z = self.down2(z)
        z = self.relu(z)

        z = self.up1(z)
        z = self.relu(z)
        z = self.up2(z)
        z = self.relu(z)

        logits = self.logits_layer(z)
        raw_depth = self.relu(self.depth_layer(z))

        return logits, raw_depth.squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
           

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self.forward(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth.squeeze(1)

'''


'''
class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        # Registering mean and std as buffers for normalization
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))

        # Down-sampling layers with ResBlock
        self.down1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.resblock1 = ResBlock(16, 16)

        self.down2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.resblock2 = ResBlock(32, 32)

        # Up-sampling layers with ResBlock
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock3 = ResBlock(16, 16)

        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock4 = ResBlock(16, 16)

        # Output layers for segmentation and depth
        self.logits_layer = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_layer = nn.Conv2d(16, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, 1, h, w)
        """
        # Normalize the input
        if isinstance(x, tuple):
            x = x[0]
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Down-sampling with ResBlocks
        z = self.relu(self.down1(z))
        z = self.resblock1(z)
        
        z = self.relu(self.down2(z))
        z = self.resblock2(z)

        # Up-sampling with ResBlocks
        z = self.relu(self.up1(z))
        z = self.resblock3(z)
        
        z = self.relu(self.up2(z))
        z = self.resblock4(z)

        # Outputs
        logits = self.logits_layer(z)
        raw_depth = self.relu(self.depth_layer(z))

        return logits, raw_depth.squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self.forward(x)
        pred = logits.argmax(dim=1)

        depth = raw_depth

        return pred, depth
'''

'''
class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        # Registering mean and std as buffers for normalization
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))

        # Down-sampling layers with batch normalization and ResBlock
        self.down1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.resblock1 = ResBlock(16, 16)

        self.down2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.resblock2 = ResBlock(32, 32)

        # Up-sampling layers with batch normalization and ResBlock
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.resblock3 = ResBlock(16, 16)

        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.resblock4 = ResBlock(16, 16)

        # Output layers for segmentation and depth
        self.logits_layer = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_layer = nn.Conv2d(16, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, 1, h, w)
        """
        # Normalize the input
        if isinstance(x, tuple):
            x = x[0]
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Down-sampling with ResBlocks and BatchNorm
        z = self.relu(self.bn1(self.down1(z)))
        z = self.resblock1(z)
        
        z = self.relu(self.bn2(self.down2(z)))
        z = self.resblock2(z)

        # Up-sampling with ResBlocks and BatchNorm
        z = self.relu(self.bn3(self.up1(z)))
        z = self.resblock3(z)
        
        z = self.relu(self.bn4(self.up2(z)))
        z = self.resblock4(z)

        # Outputs
        logits = self.logits_layer(z)
        raw_depth = self.depth_layer(z)

        return logits, raw_depth.squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self.forward(x)
        pred = logits.argmax(dim=1)

        depth = raw_depth.clamp(0, 1)  # Optional normalization of depth

        return pred, depth
'''


class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        super(Detector, self).__init__()

        # Registering mean and std as buffers for normalization
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))

        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16, 16),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 32),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, 64),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128, 128),
        )

        # Decoder (Upsampling path)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, 64),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 32),
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16, 16),
        )

        # Output layers for segmentation and depth
        self.logits_layer = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize the input
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder path with downsampling
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder path with upsampling and skip connections
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Output layers for segmentation and depth maps
        logits = self.logits_layer(dec1)
        raw_depth = self.depth_layer(dec1)

        return logits, raw_depth.squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self.forward(x)
        pred = logits.argmax(dim=1)

        # Optional normalization of depth
        depth = raw_depth.clamp(0, 1)

        return pred, depth








MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
