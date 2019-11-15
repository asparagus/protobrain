# protobrain

## Benchmark
Benchmarks are required for validating these techniques.

In this context, a `Benchmark` consists of a group of metrics, which are then evaluated on one or more brain instances with possibly different configurations.

### Metrics
Metrics provide insight into the model's behavior.

Metrics are called on every iteration to take a snapshot of the neurons' status that might be needed for later computations. At the end of an experiment, they return an object which might have any of the following:

- `global_result`: A value that describes the whole experiment
- `per_layer_result`: A value aggregated over steps that describes layers' behavior over the whole experiment.
- `per_step_result`: A value aggregated over layers that describes the system's behavior on each step.
- `per_layer_per_step_result`: A raw value obtained from each step and each layer.

In order to provide this functionality, the base class `Metric` defines three main functions that any metric needs to implement:

- `reset()`:
    This function resets all internal accumulators and should be run before any experiment.

- `next(neurons)`
    This function takes a snapshot of the neurons' state that might be needed for doing the computation later.

- `compute()`
    This function does all the require computation to produce the output.

The current metrics are:
- `metrics.spike_count.SpikeCount`
    Histogram data of the number of times that different neurons spike during the experiment. As such, there's no `per_step_result`.

- `metrics.spike_density.SpikeDensity`
    Fraction of neurons that spiked during the experiment.

### Future goals
- Metric to measure predictive performance of the system and this is the next focus of this module.
- Integration with [OpenAI Gym](https://gym.openai.com/)
