"""Module for implementation of a benchmark to evaluate architectures."""

import logging


log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark objects evaluate metrics on given architectures."""

    def __init__(self, metrics):
        """Initialize the benchmark with the given metrics.

        Args:
            metrics: List of metric objects to use
        """
        self.metrics = metrics

    def run(self, brains, inputs, learning=True, verbose=False):
        """Run the benchmark on the given brains with the given inputs.

        Args:
            brains: Brain architectures to evaluate
            inputs: Input sequence to use
            learning: Whether to use learning
            verbose: Whether to log progress as evaluations are run

        Returns:
            A list with results for each brain architecture given.
            Results consist of {'metric_name': protobrain.metric.MetricResults}
            entries for each metric.
        """
        log.setLevel(logging.DEBUG)
        metric_results = []
        for i, brain in enumerate(brains):
            for metric in self.metrics:
                metric.reset()

            brain_results = {}
            for inp in inputs:
                brain.sensor.feed(inp)
                brain.compute()
                if learning:
                    brain.learn()

                for metric in self.metrics:
                    metric.next(brain.neurons)

            for metric in self.metrics:
                result = metric.compute()
                brain_results[metric.name] = result
                log.debug("Brain #%i - %s", i, result)

            metric_results.append(brain_results)
        return metric_results
