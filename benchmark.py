from protobrain import brain
from protobrain import neuron
from protobrain import sensor
from protobrain import computation
from protobrain import learning
from protobrain.encoders import numerical
from protobrain.metrics import benchmark
from protobrain.metrics import spike_count
from protobrain.metrics import spike_density


if __name__ == '__main__':
    metrics = [
        spike_density.SpikeDensity(),
        spike_count.SpikeCount(),
    ]

    max_val = 1000
    sample_inputs = [i for i in range(max_val)]

    def create_sensor(dim):
        return sensor.Sensor(numerical.CyclicEncoder(0, max_val, dim))

    std_comp = computation.SparseComputation(0.02)
    hb_learning = learning.HebbianLearning()

    layers_1 = [neuron.Neurons(i, computation=std_comp, learning=hb_learning)
                for i in [10, 10, 10]]
    cortex_1 = neuron.FeedForward(layers_1)

    layers_2 = [neuron.Neurons(i, computation=std_comp, learning=None)
                for i in [10, 10, 10]]
    cortex_2 = neuron.FeedForward(layers_2)

    brains = [
        brain.Brain(cortex_1, create_sensor(1024)),
        brain.Brain(cortex_2, create_sensor(1024)),
    ]

    b = benchmark.Benchmark(metrics)
    results = b.run(brains, sample_inputs, verbose=True)
