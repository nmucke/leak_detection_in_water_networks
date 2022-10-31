import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import load
from skopt.plots import plot_objective, plot_histogram, plot_convergence

# Define the search space
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
dim_activation = Categorical(categories=['relu', 'elu','softplus','selu'], name='activation')
dim_batch_size = Integer(low=128, high=512, name='batch_size')
dim_lambda = Real(low=1, high=100, name="lambda")

dimensions = [dim_learning_rate,
              dim_activation,
              dim_batch_size,
              dim_lambda
             ]
default_parameters = [1e-3, 'relu', 256, 10]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, activation, batch_size, lambda):

    # Set up GAN based on hyperparameters
    generator = ...
    critic = ...

    # Train GAN
    train_loss = training(train_data, generator, critic)

    # Compute validation loss
    val_loss = validation(val_data, generator, critic)

    return val_loss


# Run the hyperparameter optimization
gp_result = gp_minimize(
    func=fitness,
    x0=default_parameters,             
    dimensions=dimensions,
    n_calls=50
    )

# Plot convergence of fitness
plot_convergence(gp_result)

# Plot the objective function
plot_objective(gp_result)

# Print best hyperparameters
print(gp_result.x)