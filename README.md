# GAN- Understanding GAN Architecture

## Generator: 
This network generates fake data that resembles real data. It takes random noise as input and transforms it into a data sample.

## Discriminator: 
This network evaluates the data, distinguishing between real and fake samples. It outputs a probability indicating whether the input is real or generated.

### Steps to Build a GAN

- Choose a programming language (Python)
- Install necessary libraries (e.g., TensorFlow, PyTorch, NumPy, Matplotlib).
- #### Prepare Your Dataset:

  Select a dataset relevant to your task (e.g., MNIST for handwritten digits).
  Preprocess the data (normalization, reshaping).
  
- #### Define the Generator Model:

  Create a neural network that takes random noise as input and outputs a generated sample.
  Use layers like Dense, BatchNorm, and activation functions like ReLU or Sigmoid.
- #### Define the Discriminator Model:

  Create a neural network that takes an image as input and outputs a probability of it being real or fake.
  Use layers like Dense and activation functions like LeakyReLU.
- #### Set Up the Training Loop:

  Alternate between training the discriminator and the generator.
  Use a loss function like binary cross-entropy to evaluate performance.
  Update the weights of both networks based on their respective losses.
- #### Train the GAN:

  Run the training loop for a specified number of epochs.
  Monitor the losses of both the generator and discriminator to ensure balanced training.
- #### Generate Samples:

  After training, use the generator to create new samples by feeding it random noise.
