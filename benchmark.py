import numpy as np
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
        # spike_density.SpikeDensity(),
        spike_count.SpikeCount(),
    ]

    cycles = 5
    max_val = 100
    sample_inputs = [i % max_val for i in range(max_val * cycles)]

    def create_sensor(dim):
        return sensor.Sensor(numerical.CyclicEncoder(0, max_val, dim))

    def create_brain(architecture,
                     sensor_dim,
                     computation=None,
                     learning=None,
                     random_seed=0):
        np.random.seed(random_seed)
        cortex = neuron.FeedForward([neuron.Neurons(i) for i in architecture])
        return brain.Brain(cortex, create_sensor(sensor_dim),
                           computation=computation, learning=learning)

    std_comp = computation.SparseComputation(0.05)
    hb_learning = learning.HebbianLearning()

    brains = [
        create_brain([100, 100, 100], 1024, std_comp),
        create_brain([100, 100, 100], 1024, std_comp, hb_learning),
        create_brain([100, 100, 100], 1024, std_comp, learning.HebbianLearning(0.005, 0.001)),
        create_brain([100, 100, 100], 1024, std_comp, learning.HebbianLearning(0.002, 0.0002)),
        create_brain([100, 100, 100], 1024, std_comp, learning.HebbianLearning(0.001, 0.0001)),
    ]

    b = benchmark.Benchmark(metrics)
    results = b.run(brains, sample_inputs, verbose=True)
