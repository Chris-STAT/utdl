from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_size = 2 * n_track *2
        self.output_size = n_waypoints * 2

        self.mlp = nn.Sequential(
             nn.Linear(self.input_size, 128), # Input layer
             nn.ReLU(),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Linear(64, 32),
             nn.ReLU(),
             nn.Linear(32, self.output_size)
             )



    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #raise NotImplementedError
        
        batch_size = track_left.shape[0]
        
        # Faltten the left and right track boundaries
        track_left_flat = track_left.view(batch_size, -1) # shape (b, self.n_track * 2)
        track_right_flat = track_right.view(batch_size, -1) # shape (b, self.n_track * 2)

        # Concatenate the flattened left and right boundaries
        inputs = torch.cat([track_left_flat, track_right_flat], dim=1)  # shape (b, 2 * n_track * 2)

        # Pass through the MLP
        outputs = self.mlp(inputs)

        # Reshape to (b, n_waypoints, 2)
        return outputs.view(batch_size, -1, 2)
    


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_track (int): Number of points in each side of the track.
            n_waypoints (int): Number of waypoints to predict.
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            dim_feedforward (int): Dimension of feedforward layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Embeddings for queries (waypoints)
        self.query_embed = nn.Embedding(self.n_waypoints, self.d_model)

        # Linear projection for track input
        self.input_proj = nn.Linear(2, self.d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
        )

        # Output projection to (x, y) coordinates
        self.output_proj = nn.Linear(self.d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (B, n_track, 2)
            track_right (torch.Tensor): shape (B, n_track, 2)

        Returns:
            torch.Tensor: Predicted waypoints with shape (B, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Concatenate left and right tracks (B, n_track * 2, 2)
        track = torch.cat([track_left, track_right], dim=1)

        # Linear projection to embed track points (B, n_track * 2, d_model)
        track_embed = self.input_proj(track)

        # Positional encoding (optional, you can implement it for better results)
        position_ids = torch.arange(track_embed.shape[1], device=track.device).unsqueeze(0)
        position_embed = F.one_hot(position_ids, num_classes=self.d_model).float()
        track_embed += position_embed

        # Query embeddings (n_waypoints, d_model)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Transformer decoder
        waypoints_embed = self.transformer_decoder(
            query_embed,  # Queries
            track_embed.permute(1, 0, 2),  # Keys/Values (shape: seq_len, batch_size, d_model)
        ).permute(1, 0, 2)  # Output shape: (B, n_waypoints, d_model)

        # Project to (x, y) coordinates (B, n_waypoints, 2)
        waypoints = self.output_proj(waypoints_embed)

        return waypoints

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)


        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, H/16, W/16)
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*8, 512), # Assuming input size (3, 96, 128)
            nn.ReLU(),
            nn.Linear(512, self.n_waypoints*2), # Predict (x,y) for each waypoint
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x = self.conv_block(x)
        x = self.output(x)

        # Reshape output to (B, n_waypoints, 2)
        waypoints = x.view(-1, self.n_waypoints, 2)

        return waypoints

        #raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
