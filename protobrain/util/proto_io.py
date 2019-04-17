#!/usr/bin/python
# -*- coding: utf-8 -*-


class ProtoWriter:
    def __init__(self, open_file):
        self.open_file = open_file

    def write(self, proto):
        proto_bytes = proto.SerializeToString()
        length_bytes = len(proto_bytes).to_bytes(4, 'big')
        self.open_file.write(length_bytes)
        self.open_file.write(proto_bytes)


class ProtoReader:
    def __init__(self, open_file, proto_class):
        self.open_file = open_file
        self.proto_class = proto_class

    def __iter__(self):
        while True:
            length_bytes = self.open_file.read(4)
            if not length_bytes:
                break

            length = int.from_bytes(
                length_bytes,
                byteorder='big',
                signed=False
            )

            proto_bytes = self.open_file.read(length)
            yield self.proto_class.FromString(proto_bytes)




