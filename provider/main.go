package main

import (
	"context"
	"fmt"

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
		Description: "Send a work request to the server",
		Execute: func(args []string) error {
			resp, err := p.client.Work(context.Background(), &pb.WorkRequest{
				Sender: "provider",
			})
			if err != nil {
				return err
			}
			fmt.Println("Server replied:", resp.Ack, resp.WorkerId)
			p.id = resp.WorkerId
			return nil
		},
	})

	cp.Register(processor.Command{
		Name:        "S",
		Description: "Send a stop work request to the server",
		Execute: func(args []string) error {
			resp, err := p.client.Stop(context.Background(), &pb.StopRequest{
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
