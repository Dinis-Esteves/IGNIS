package main

import (
	"context"
	"fmt"

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
		Description: "Submit an inference job (e.g. J openai/whisper-base Hello world)",
		Execute: func(args []string) error {
			if len(args) < 2 {
				return fmt.Errorf("usage: J <model_name> <input_text>")
			}
			modelName := args[0]
			inputText := fmt.Sprintf("%s", joinArgs(args[1:]))

			resp, err := c.client.Job(context.Background(), &pb.JobRequest{
				Tasktype:  pb.TaskType_AI_INFERENCE,
				ModelName: modelName,
				InputText: inputText,
			})
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

func joinArgs(args []string) string {
	result := ""
	for i, a := range args {
		if i > 0 {
			result += " "
		}
		result += a
	}
	return result
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
