# protobrain-proto
Protobuf objects for the [protobrain](https://github.com/Ariel-Perez/protobrain) project.

For more information on Protocol buffers, read the [official documentation](https://developers.google.com/protocol-buffers/).

## Compiling changes
The compiled protobufs for C++, JavaScript and Python are checked in with the code, but if you make any changes or need any other languages, you'll have to recompile them.

If you haven't installed the compiler, [download the package](https://developers.google.com/protocol-buffers/docs/downloads) and follow the instructions in the README.

####Python:

```
protoc -I=proto --python_out=python proto/*.proto
```
####JavaScript
```
protoc -I=proto --js_out=import_style=commonjs,binary:js proto/*.proto
```
####C++
```
protoc -I=proto --cpp_out=cpp proto/*.proto
```
