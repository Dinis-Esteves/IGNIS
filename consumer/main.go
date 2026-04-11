package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"ignis/common/processor"
	pb "ignis/common/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ConsumerClient struct {
	id     string
	jobId  string
	client pb.IgnisServiceClient
}

func (c *ConsumerClient) registerCommands(cp *processor.CommandProcessor) {
	cp.Register(processor.Command{
		Name:        "G",
		Description: "Send a greeting to the server",
		Execute: func(args []string) error {
			resp, err := c.client.Greet(context.Background(), &pb.GreetRequest{
				Sender:  "consumer",
				Message: "Hello from consumer!",
			})
			if err != nil {
				return err
			}
			fmt.Println("Server replied:", resp.Reply)
			return nil
		},
	})

	cp.Register(processor.Command{
		Name:        "J",
		Description: "Submit a job (e.g. J AI_INFERENCE facebook/opt-125m Hello world)",
		Execute: func(args []string) error {
			if len(args) < 3 {
				return fmt.Errorf("usage: J <task_type> <model_name> <input_data>")
			}
			taskVal, ok := pb.TaskType_value[args[0]]
			if !ok {
				return fmt.Errorf("unknown task type: %s", args[0])
			}
			modelName := args[1]
			remaining := args[2:]
			var inputData, config string
			splitAt := -1
			for i, a := range remaining {
				if strings.HasPrefix(a, "{") {
					splitAt = i
					break
				}
			}
			if splitAt >= 0 {
				inputData = strings.Join(remaining[:splitAt], " ")
				config = strings.Join(remaining[splitAt:], " ")
			} else {
				inputData = strings.Join(remaining, " ")
			}

			req := &pb.JobRequest{
				Tasktype:  pb.TaskType(taskVal),
				ModelName: modelName,
				InputData: inputData,
				Config:    config,
			}

			// if inputData is a file path, send bytes instead
			if fileBytes, err := os.ReadFile(inputData); err == nil {
				req.InputBytes = fileBytes
				req.InputData = ""
			}

			resp, err := c.client.Job(context.Background(), req)
			if err != nil {
				return err
			}
			c.jobId = resp.JobId
			fmt.Println("Job submitted, ID:", c.jobId)
			return nil
		},
	})

	cp.Register(processor.Command{
		Name:        "R",
		Description: "Check result of the last submitted job",
		Execute: func(args []string) error {
			if c.jobId == "" {
				return fmt.Errorf("no job submitted yet")
			}
			resp, err := c.client.GetResult(context.Background(), &pb.GetResultRequest{
				JobId: c.jobId,
			})
			if err != nil {
				return err
			}
			if !resp.Ready {
				fmt.Println("Result not ready yet")
				return nil
			}
			fmt.Println("Result:", resp.Output)
			return nil
		},
	})
}


func main() {
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	c := &ConsumerClient{client: pb.NewIgnisServiceClient(conn)}

	cp := processor.NewCommandProcessor()
	c.registerCommands(cp)
	cp.Run()
}
