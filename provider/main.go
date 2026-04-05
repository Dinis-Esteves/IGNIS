package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"ignis/common/processor"
	pb "ignis/common/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ProviderClient struct {
	id     string
	client pb.IgnisServiceClient
}

func (p *ProviderClient) registerCommands(cp *processor.CommandProcessor) {
	cp.Register(processor.Command{
		Name:        "G",
		Description: "Send a greeting to the server",
		Execute: func(args []string) error {
			resp, err := p.client.Greet(context.Background(), &pb.GreetRequest{
				Sender:  "provider",
				Message: "Hello from provider!",
			})
			if err != nil {
				return err
			}
			fmt.Println("Server replied:", resp.Reply)
			return nil
		},
	})

	cp.Register(processor.Command{
		Name:        "W",
		Description: "Register as a worker and start receiving jobs",
		Execute: func(args []string) error {
			stream, err := p.client.Subscribe(context.Background(), &pb.SubscribeRequest{
				Sender: "provider",
			})
			if err != nil {
				return err
			}

			// Read the worker ID from the first message header isn't available in
			// unary-style, so we generate the ID server-side and log it there.
			// For now we just mark ourselves as active.
			p.id = "subscribed"
			fmt.Println("Registered — waiting for jobs...")

			go func() {
				for {
					job, err := stream.Recv()
					if err != nil {
						fmt.Println("Stream closed:", err)
						p.id = ""
						return
					}
					fmt.Printf("Job received [%s] model:%s\n", job.JobId, job.ModelName)
					go p.handleJob(job)
				}
			}()

			return nil
		},
	})

	cp.Register(processor.Command{
		Name:        "S",
		Description: "Unregister from the server",
		Execute: func(args []string) error {
			if p.id == "" {
				return fmt.Errorf("not registered")
			}
			resp, err := p.client.Unsubscribe(context.Background(), &pb.UnsubscribeRequest{
				WorkerId: p.id,
			})
			if err != nil {
				return err
			}
			fmt.Println("Server replied:", resp.Ack)
			p.id = ""
			return nil
		},
	})
}

func (p *ProviderClient) handleJob(job *pb.JobAssignment) {
	output, err := runInference(job.ModelName, job.InputText)
	if err != nil {
		fmt.Printf("Inference failed [%s]: %v\n", job.JobId, err)
		output = fmt.Sprintf("ERROR: %v", err)
	}

	_, err = p.client.SubmitResult(context.Background(), &pb.SubmitResultRequest{
		JobId:  job.JobId,
		Output: output,
	})
	if err != nil {
		fmt.Printf("Failed to submit result [%s]: %v\n", job.JobId, err)
		return
	}

	fmt.Printf("Result submitted [%s]\n", job.JobId)
}

func runInference(modelName, inputText string) (string, error) {
	dir, _ := os.Getwd()
	if filepath.Base(dir) == "provider" {
		dir = filepath.Dir(dir)
	}
	python := filepath.Join(dir, "provider", ".venv", "bin", "python3")
	script := filepath.Join(dir, "provider", "inference.py")
	cmd := exec.Command(python, script, modelName, inputText)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			fmt.Fprintf(os.Stderr, "[inference stderr] %s\n", string(exitErr.Stderr))
		}
		return "", fmt.Errorf("inference process failed")
	}
	return strings.TrimSpace(string(out)), nil
}

func main() {
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	p := &ProviderClient{client: pb.NewIgnisServiceClient(conn)}

	cp := processor.NewCommandProcessor()
	p.registerCommands(cp)
	cp.Run()
}
