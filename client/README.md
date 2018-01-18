## requirements

- [protobuf](https://github.com/google/protobuf)
- [protoc-gen-go](https://github.com/golang/protobuf)

## setup

```bash
# for pb
protoc -I=serving -I serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow_serving/apis/*.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/framework/*.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/protobuf/{saver,meta_graph}.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/example/*.proto
# for dependent libraries
dep ensure
```

## run

Firstly, run tensorflow model server, then

```bash
go run main.go -addr=host:9000 -log=log.csv
```