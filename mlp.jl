# http://www.diegoacuna.me/multi-layer-perceptron-for-regression-in-julia-using-the-mocha-framework/

using Mocha
using Distributions

#Parameter Definition for the dataset generation
media_x1=0.0
media_x2=0.0
mean=[media_x1;media_x2]
var_x1=1.0
var_x2=1.0
var=[var_x1 0.0;0.0 var_x2]

# by fixing the random seed we can replicate our results
srand(500)

# we're going to use 5000 points to estimate our function
tam = 5000

# generate inputs, those are the x and y points on our function
# we generate them using the normal distribution
generate_dataset(media,var,tam) = rand(MvNormal(media, var), tam)

# generate outputs
f1(x1, x2) = sin(x1).*sin(x2)./(x1.*x2)

datasetinput = generate_dataset(mean, var, tam)
datasetoutput = f1(datasetinput[1,:], datasetinput[2,:])

# i don't have a gpu :(
backend = CPUBackend()
init(backend)

# first layer, is a data layer and receives as input our dataset input (the 5000 points)
# and also we pass the outputs from our function, those are needed to train our network
data_layer = MemoryDataLayer(name="data", data=Array[datasetinput, datasetoutput], batch_size=100)
# then we have our fully connected hidden layer, here I use 35 hidden neurons
ip_layer = InnerProductLayer(name="ip", output_dim=2, bottoms=[:data], tops=[:ip], neuron=Neurons.Tanh())
# the final layer is also a fully connected layer but with only one neuron, the output one
aggregator = InnerProductLayer(name="aggregator", output_dim=1, tops=[:aggregator], bottoms=[:ip] )

layer_loss = SquareLossLayer(name="loss", bottoms=[:aggregator, :label])

# first with the layers we construct our final neural network
common_layers = [ip_layer, aggregator]
net = Net("MLP", backend, [data_layer, common_layers..., layer_loss])

# when we train our network, also we perform validation of the training
# for this, we define a twin neural network where the only difference is
# the input layer (because we pass validation inputs and labels)
input_test = generate_dataset(mean, var, 5000)
output_test = f1(input_test[1,:], input_test[2,:])

data_test = MemoryDataLayer(data = Array[input_test, output_test], batch_size = 100)
accuracy = SquareLossLayer(name="acc", bottoms=[:aggregator, :label])

net_test = Net("test", backend, [data_test, common_layers..., accuracy])

# we tell Mocha that this "twin" network is for validation purposes
test_performance = ValidationPerformance(net_test)

# to train the network we use stochastic gradient descent
method = SGD() # stochastic gradient descent

# the max. number of iterations of SGD is 1000
params = make_solver_parameters(method, max_iter=1000)
solver = Solver(method, params)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=10)
add_coffee_break(solver, Snapshot("snapshots"), every_n_iter=1000)
add_coffee_break(solver, test_performance, every_n_iter=10)

setup_coffee_lounge(solver, save_into="statistics.jld", every_n_iter=10)

# train the network
solve(solver, net)

# dump some useful statistics
Mocha.dump_statistics(solver.coffee_lounge, get_layer_state(net, "loss"), true)

data = a["obj_val"]
dfs = [x = collect(keys(data)); y = [data[xx] for xx in x]


finetune = load("statistics.jld")["statistics"]
randinit = load("statistics.jld")["statistics"]
stats = [finetune, randinit]
category = ["pre-train", "random init"]

plot_vars = (("acc-square-loss", "Accuracy (Test set)"),
             ("obj_val", "Objective Function (Training set)"))

for (var, title) in plot_vars
  dfs = [begin
    data = stats[i][var]
    x = collect(keys(data)); y = [data[xx] for xx in x]
    DataFrame(x=x, y=y, category=category[i])
  end for i = 1:length(stats)]

  df = vcat(dfs...)
  the_plot = plot(df, x="x", y="y", color="category",
      Geom.line, Guide.xlabel("Iteration"), Guide.title(title))

  draw(SVG("$var.svg", 18cm, 9cm), the_plot)
end

# free resources and shutdown the backend
destroy(net)
destroy(net_test)
shutdown(backend)
