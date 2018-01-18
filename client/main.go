package main

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"strings"
	"time"

	protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"
	tf_core_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"
)

var (
	addr          = flag.String("addr", "localhost:9000", "The tensorflow serving address")
	req           = flag.Int("req", 1000, "number of requests")
	features      = flag.Int("features", 20, "number of features")
	modelName     = flag.String("model_name", "lr", "model name")
	signatureName = flag.String("signature_name", "predict", "signature name")
	modelVersion  = flag.Int("model_version", 1, "model version")
	log           = flag.String("log", "log.csv", "path of log csv")
)

func genDummyData(size int) []float32 {
	features := make([]float32, size)
	for i := 0; i < size; i++ {
		features[i] = rand.Float32()
	}
	return features
}

func genRequest(modelName string, signatureName string, modelVersion int64, values []float32) *pb.PredictRequest {
	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          modelName,
			SignatureName: signatureName,
			Version:       &protobuf.Int64Value{Value: modelVersion},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			"inputs": &tf_core_framework.TensorProto{
				Dtype: tf_core_framework.DataType_DT_FLOAT,
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{
						&tf_core_framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&tf_core_framework.TensorShapeProto_Dim{
							Size: int64(len(values)),
						},
					},
				},
				FloatVal: values,
			},
		},
	}
	return request
}

func writeLog(lines []string, fname string) {
	header := "id,elapsed_time"
	content := strings.Join(lines, "\n")
	ioutil.WriteFile(fname, []byte(header+"\n"+content+"\n"), 0644)
}

func main() {
	flag.Parse()

	data := genDummyData(*features)
	conn, err := grpc.Dial(*addr, grpc.WithInsecure())
	if err != nil {
		fmt.Println("cannot connect to server")
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)
	lines := make([]string, *req)

	fmt.Println("start prediction")
	for i := 0; i < *req; i++ {
		start := time.Now()
		request := genRequest(*modelName, *signatureName, int64(*modelVersion), data)
		_, err := client.Predict(context.Background(), request)
		if err != nil {
			fmt.Println("cannot predict")
			fmt.Println(err)
		}
		end := time.Now()
		elapsed_time := end.Sub(start).Seconds()
		lines[i] = fmt.Sprintf("%d,%f", i, elapsed_time)
		//fmt.Println(resp.Outputs["outputs"].FloatVal)
	}
	fmt.Println("end prediction")

	writeLog(lines, *log)
	fmt.Println("write log to " + *log)
}
