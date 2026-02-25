

from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.utils import BranchConstructor, CNN1DBranchConstructor, MLPConstructor


def model_1(latent_dim: int = 64, output_dim: int = 12) -> MIONet:
    """Create MIONet with default architecture.

    Args:
        latent_dim: Dimension of latent space shared by branches and trunk.
        output_dim: Number of output features.

    Returns:
        Uninitialized MIONet model.
    """
    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[3, 100, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="surge_force",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="sway_force",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="yaw_moment",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 250, 250, latent_dim],
        activation="gelu"
    )
    return MIONet(branches, trunk, output_dim)



def model_2() -> MIONet:
    """Create MIONet with default architecture.

    Args:
        latent_dim: Dimension of latent space shared by branches and trunk.
        output_dim: Number of output features.

    Returns:
        Uninitialized MIONet model.
    """
    latent_dim: int = 250
    output_dim: int = 12
    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[3, 100, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="surge_force",
            layer_sizes=[1000, 750, 500, 250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="sway_force",
            layer_sizes=[1000, 750, 500,  250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="yaw_moment",
            layer_sizes=[1000, 750, 500, 250, 250, latent_dim],
            activation="gelu"
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu"
    )

    return MIONet(branches, trunk, output_dim)

def model_cnn_1() -> MIONet:
    latent_dim = 250
    # Number of features to predict.
    output_dim = 12


    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[12, 100, latent_dim],
            activation="gelu"
        ),
        CNN1DBranchConstructor(
            name="surge_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu"
        ),
        CNN1DBranchConstructor(
            name="sway_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu"
        ),
        CNN1DBranchConstructor(
            name="yaw_moment",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu"
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu"
    )
    
    return MIONet(branches, trunk, output_dim)

def model_cnn_2() -> MIONet:
    latent_dim = 128
    # Number of features to predict.
    output_dim = 12


    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[12, 100, latent_dim],
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="surge_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="sway_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="yaw_moment",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu",
        dropout=0.1
    )
        
    return MIONet(branches, trunk, output_dim)


def model_1dof() -> MIONet:
    latent_dim = 128
    output_dim = 1
    input_dim = 1

    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[input_dim, 100, latent_dim],
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="surge_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
    ]

    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu",
        dropout=0.1
    )

    return MIONet(branches, trunk, output_dim)

def model_1dof_2() -> MIONet:
    latent_dim = 256
    output_dim = 1
    input_dim = 1

    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[input_dim, 256, 256, latent_dim],
            activation="sin",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="surge_force",
            in_channels=1,
            channels     =[128, 128,  96,  64,  32,  32],
            kernel_sizes =[  7,  15,  31,  63, 127, 255],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
    ]

    trunk = MLPConstructor(
        layer_sizes=[1, 128, 256, 256, latent_dim],
        activation="gelu",
        dropout=0.1
    )

    return MIONet(branches, trunk, output_dim)