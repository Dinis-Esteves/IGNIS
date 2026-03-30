package main

import (
	"context"
	"fmt"

	pb "ignis/common/proto"
	"ignis/common/processor"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	client := pb.NewGreetingServiceClient(conn)

	cp := processor.NewCommandProcessor()

	cp.Register(processor.Command{
		Name:        "G",
		Description: "Send a greeting to the server",
		Execute: func(args []string) error {
			resp, err := client.Greet(context.Background(), &pb.GreetRequest{
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

	cp.Run()
}
