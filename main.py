#!/usr/bin/python
# -*- coding: utf-8 -*-
import subprocess
import tempfile
from protobrain.proto import experiment_pb2
from protobrain.proto import encoder_pb2
from protobrain.proto import snapshot_pb2
from protobrain.util import proto_io


if __name__ == '__main__':
    exp = experiment_pb2.Experiment()
    exp.encoder.type = encoder_pb2.Encoder.NUMERICAL_CYCLIC
    exp.encoder.shape.extend([98])

    ext = encoder_pb2.CyclicEncoder.cyclic_encoder
    exp.encoder.Extensions[ext].min_value = 0
    exp.encoder.Extensions[ext].max_value = 100

    exp.cortex.layer.extend([40, 40, 30, 20])

    inp = exp.input.add()
    inp.int = 1
    inp = exp.input.add()
    inp.int = 2
    inp = exp.input.add()
    inp.int = 3
    inp = exp.input.add()
    inp.int = 4

    with tempfile.NamedTemporaryFile('wb') as input_file:
        with tempfile.NamedTemporaryFile('rb') as output_file:
            input_file.write(exp.SerializeToString())
            input_file.seek(0)

            subprocess.run(
                ['python', 'protobrain/cli/experiment.py', input_file.name, output_file.name]
            )

            output_file.seek(0)
            reader = proto_io.ProtoReader(output_file, snapshot_pb2.Snapshot)
            for proto in reader:
                print(proto)

    print('Finished')
