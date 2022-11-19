import torch
import time
from pathlib import Path
import pickle

try:
    from .modules import ReLU, Sigmoid, TransposeConv2d, Sequential, MSE, Conv2d, Upsampling
    from .optim import SGD
except:
    from modules import ReLU, Sigmoid, TransposeConv2d, Sequential, MSE, Conv2d, Upsampling
    from optim import SGD


class Model:
    def __init__(self, learning_rate: float = 1e-3, hidden_dim: int = 16, no_padding: bool = False) -> None:
        """Initialize denoising model.

        Args:
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            hidden_dim (int, optional): Hidden channel size. Defaults to 16.
            no_padding (bool, optional): Whether to not apply padding. Defaults to False.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if no_padding:
            self.model = Sequential(
                Conv2d(3, hidden_dim, 3, stride=2),
                ReLU(),
                Conv2d(hidden_dim, hidden_dim, 3, stride=2),
                ReLU(),
                Upsampling(hidden_dim, hidden_dim, 3, stride=2),
                ReLU(),
                Upsampling(hidden_dim, 3, 4, stride=2),
                Sigmoid()).to(self.device)  
        else:
            # our best model arch
            self.model = Sequential(
                Conv2d(3, hidden_dim, 3, stride=2, padding=1),
                ReLU(),
                Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                ReLU(),
                Upsampling(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                ReLU(),
                Upsampling(hidden_dim, 3, 4, stride=2, padding=1),
                Sigmoid()).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = MSE()
        self.validate_every = 0
        self.batch_size = 64

    def save_pretrained_model(self, ckpt_name: str = Path(__file__).parent / 'bestmodel.pth') -> None:
        """Saves pretrained model to specified path.

        Args:
            ckpt_name (str, optional): Checkpoint path to save to. Defaults to Path(__file__).parent/'bestmodel.pth'.
        """
        model = self.model.to(torch.device("cpu"))
        with open(ckpt_name, "wb") as ckpt_f:
            pickle.dump(model.state_dict(), ckpt_f)

    def load_pretrained_model(self, ckpt_name: str = Path(__file__).parent / 'bestmodel.pth') -> None:
        """Loads pretrained model from specified path.

        Args:
            ckpt_name (str, optional): Checkpoint path to load from. Defaults to Path(__file__).parent/'bestmodel.pth'.
        """
        print(f'Loading pretrained model from {ckpt_name}')
        with open(ckpt_name, "rb") as ckpt_f:
            ckpt = pickle.load(ckpt_f)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.optimizer.load_parameters(self.model.parameters())

    def train(self, train_input, train_target, num_epochs: int = 25, use_wandb: bool = False) -> None:
        """Trains a model and validates if necessary.

        Args:
            train_input (_type_): Train set
            train_target (_type_): Train target
            num_epochs (int, optional): Number of epochs. Defaults to 25.
            use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.
        """
        # Need to do this for memory optimization purposes.
        # Otherwise torch also builds the computation graph and this quickly fills up memory
        torch.set_grad_enabled(False)
        print('Training...')

        # Set model in training mode
        self.model.train()

        if use_wandb:
            import wandb

        # If input is ByteTensor, convert to FloatTensor
        train_input = self.__check_input_type(train_input)
        train_target = self.__check_input_type(train_target)

        start_time = time.time()
        num_batches = len(train_input) / self.batch_size

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            print(f'Epoch {epoch + 1} / {num_epochs}')
            # Minibatch loop
            for batch_idx in range(0, len(train_input), self.batch_size):
                # Get minibatch
                batch_input = train_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                batch_target = train_target[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model(batch_input)
                # Compute loss
                loss = self.criterion(output, batch_target)
                running_loss += loss.item()
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()

            # Compute average batch loss
            train_loss = running_loss / num_batches

            print(f'\tLoss: {train_loss:.6f}')

            if use_wandb:
                wandb.log({'train_loss': train_loss})

            # Validate if validation frequency is set, which requires a validation set
            if self.validate_every:
                if epoch % self.validate_every == self.validate_every - 1:
                    val_loss = self.validate(self.val_input, self.val_target)
                    print(f'\tValidation loss: {val_loss:.6f}')

                    if use_wandb:
                        wandb.log({'val_loss': val_loss})

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

        # We set this back for consistency
        torch.set_grad_enabled(True)

    def validate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> float:
        print('Validating...')
        torch.set_grad_enabled(False)

        # Set model in evaluation mode
        self.model.eval()

        # If input is ByteTensor, convert to FloatTensor
        test_input = self.__check_input_type(test_input)
        test_target = self.__check_input_type(test_target)
        running_loss = 0.0

        # Validation loop
        for batch_idx in range(0, len(test_input), self.batch_size):
            # Get minibatch
            batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                self.device)
            batch_target = test_target[batch_idx:batch_idx + self.batch_size].to(
                self.device)
            # Forward pass
            output = self.model(batch_input)
            # Compute loss
            loss = self.criterion(output, batch_target)
            running_loss += loss.item()

        # Set model back in training mode
        self.model.train()

        torch.set_grad_enabled(True)

        # Return loss
        return running_loss / (len(test_input) / self.batch_size)

    def predict(self, test_input) -> torch.Tensor:
        torch.set_grad_enabled(False)
        # Set model in evaluation mode
        self.model.eval()

        # Predict on minibatches
        test_input = self.__check_input_type(test_input)
        denoised_output = torch.empty(test_input.shape, dtype=torch.uint8).to(self.device)

        for batch_idx in range(0, len(test_input), self.batch_size):
            # Get minibatch
            batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                self.device)
            # Forward pass
            output = self.model(batch_input) * 255
            # Clip output to [0, 255]
            output = output.clamp(0, 255)
            # Save output
            denoised_output[batch_idx:batch_idx + self.batch_size] = output

        self.model.train()

        torch.set_grad_enabled(True)
        return denoised_output

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_val_data(self, val_input: torch.Tensor, val_target: torch.Tensor,
                     validation_frequency: int = 10) -> None:
        self.val_input = val_input
        self.val_target = val_target
        self.validate_every = validation_frequency

    @staticmethod
    def __check_input_type(tensor_input: torch.Tensor) -> torch.Tensor:
        # Convert Byte tensors to float if not already
        if isinstance(tensor_input, (torch.ByteTensor, torch.cuda.ByteTensor)):
            return tensor_input.float() / 255.0
        return tensor_input