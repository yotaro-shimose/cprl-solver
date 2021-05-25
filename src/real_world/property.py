class RealDQNArgs:
    def __init__(
        self,
        n_city: int,
        grid_size: int,
        tw_size: float,
        max_travel_time: float,
        tw_ratio: float,
        seed: int,
        batch_size: int,
        learning_rate: float,
        hidden_layer: int,
        latent_dim: int,
        max_softmax_beta: float,
        n_step: int,
        mode: str,
        plot_training: bool,
    ):
        self.n_city = n_city
        self.grid_size = grid_size
        self.tw_size = tw_size
        self.max_travel_time = max_travel_time
        self.tw_ratio = tw_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layer = hidden_layer
        self.latent_dim = latent_dim
        self.max_softmax_beta = max_softmax_beta
        self.n_step = n_step
        self.mode = mode
        self.plot_training = bool(plot_training)
