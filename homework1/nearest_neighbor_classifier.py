import torch


class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        You do not have to modify this function, but you will need to implement the functions it calls.

        Args:
            x: list of lists of floats, data
            y: list of floats, labels
        """
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)

    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warmup: Convert the data into PyTorch tensors.
        Assumptions:
        - len(x) == len(y)

        Args:
            x: list of lists of floats, data
            y: list of floats, labels

        Returns:
            tuple of x and y both torch.Tensor's.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the data.
        Each row denotes a single data point.

        Args:
            x: 2D tensor data shape = [N, D]

        Returns:
            tuple of mean and standard deviation of the data.
            Both should have a shape [1, D]
        """
        row_means = x.mean(dim=0).unsqueeze(0)
        row_stds = x.std(dim=0).unsqueeze(0)
        return row_means, row_stds

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input x using the mean and std computed from the data in __init__

        Args:
            x: 1D or 2D tensor shape = [D] or [N, D]

        Returns:
            normalized 2D tensor shape = x.shape
        """
        return (x - self.data_mean) / self.data_std

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the input x's nearest neighbor and the corresponding label.

        Args:
            x: 1D tensor shape = [D]

        Returns:
            tuple of the nearest neighbor data point [D] and its label [1]
        """
        x = self.input_normalization(x)
        dists = torch.cdist(self.data_normalized,x.unsqueeze(0))
        idx = dists.argmin()
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors of input x from the data.

        Args:
            x: 1D tensor shape = [D]
            k: int, number of neighbors

        Returns:
            tuple of the k-nearest neighbors data points and their labels
            data points will be size (k, D)
            labels will be size (k,)
        """
        x = self.input_normalization(x)
        dists = torch.cdist(self.data_normalized,x.unsqueeze(0)).squeeze(0,-1)
        _, idx = (-dists).topk(k)
        return self.data[idx], self.label[idx]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use the k-nearest neighbors of the input x to predict its regression label.
        The prediction will be the average value of the labels from the k neighbors.

        Args:
            x: 1D tensor [D]
            k: int, number of neighbors

        Returns:
            average value of labels from the k neighbors. Tensor of shape [1]
        """
        _, k_labels = self.get_k_nearest_neighbor(x,k)
        return k_labels.mean()

