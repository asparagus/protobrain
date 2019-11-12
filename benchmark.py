from protobrain import brain
from protobrain import neuron
from protobrain import sensor
from protobrain import computation
from protobrain.encoders import numerical
from protobrain.metrics import benchmark
from protobrain.metrics import spike_density


if __name__ == '__main__':
    metrics = [
        spike_density.SpikeDensity(),
    ]

    sens = sensor.Sensor(numerical.CyclicEncoder(0, 100, 1024))
    sample_inputs = [i for i in range(100)]

    std_comp = computation.SparseComputation(0.02)
    layers = [neuron.Neurons(i, std_comp) for i in [10, 10, 10]]
    cortex = neuron.FeedForward(layers)
    brains = [
        brain.Brain(cortex, sens)
    ]

    b = benchmark.Benchmark(metrics)
    results = b.run(brains, sample_inputs, verbose=True)
