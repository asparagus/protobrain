# protobrain
This is an experimental project for biologically-inspired machine learning techniques.

Loosely based on Hierarchical Temporal Memory by [Numenta](http://numenta.org/).

## Use
The purpose of this library is to enable quick experimentation with different computation and learning models.

There are three steps to building any model:
1. Define an architecture
    This involves building a nested-layer architecture and setting up connections between them.

    You can build groups of neurons and individually manage the connections between them.
    ```python
    n1 = neuron.Neurons(5)  # [x, x, x, x, x]
    n2 = neuron.Neurons(3)  #    [x, x, x]

    n2.input = n1  # Sets the 'MAIN' input of n2 to be n1
    ```

    Inputs have names, which can be referred to in the computation functions. You can access these through the `set` and `get` functions.
    ```python
    n1.set('feedback', n2)
    ```

    You can treat neurons as layers and connect them with either `neuron.FeedForward` for forward connections, `neuron.FeedBackward` for backward connections, or `neuron.LoopBack` for connections between the same layer.
    ```python
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]
    # Not specifying an input name leads to setting the 'MAIN' input
    neuron.FeedForward(layers)
    neuron.FeedBackward(layers, 'feedback')
    neuron.LoopBack(layers, 'neighbors')
    ```

2. Define a computation function:
    Every neuron has to be able to compute an output based on its inputs.

    Some available computations are `computation.StandardComputation` and `computation.SparseComputation`.

    ```python
    std = computation.StandardComputation(threshold=0.5)
    sparse = computation.SparseComputation(n=0.05)
    ```

    You can also implement your own function by creating a child class of `computation.Computation`.

3. Define a learning function:
    Synapses might be adjusted based on some logic.

    The only currently available learning function is `learning.HebbianLearning`.

    ```python
    learn = learning.HebbianLearning(increase=0.005, decrease=0.001)
    ```

4. Define a sensor object that feeds inputs into the brain.
    ```python
    encoder = numerical.CyclicEncoder(min_value, max_value, encoding_length)
    sensor_ = sensor.Sensor(encoder=encoder)
    ```

5. Define a brain object that combines all of these:

    ```python
    brain_ = brain.Brain(neurons, sensor_, computation=computation, learning=learning)
    ```

Once you've built a model, you want to run a benchmark to verify that it's doing what it's supposed to, and compare against different setups.

An example of this can be seen in the [`benchmark.py`](benchmark.py) script.

## Building the protobuf files
Some functionality of this project depends on protobuf files. To build them, use the following command:
```
protoc -I=. --python_out=. protobrain/proto/*.proto
```

For more information on Protocol buffers, read the [official documentation](https://developers.google.com/protocol-buffers/).

## Testing
This project uses pytest and the tests can be run from the repository root.
```
python -m pytest
```

## License
protobrain is a project licensed under GNU GPLv3.
