package main

import (
	"context"
	"fmt"
	"net"

	pb "ignis/common/proto"

	"google.golang.org/grpc"
)

type greetingServer struct {
	pb.UnimplementedGreetingServiceServer
}

func (s *greetingServer) Greet(_ context.Context, req *pb.GreetRequest) (*pb.GreetResponse, error) {
	fmt.Printf("[%s]: %s\n", req.Sender, req.Message)
	return &pb.GreetResponse{Reply: "Hello, " + req.Sender + "!"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		panic(err)
	}

	s := grpc.NewServer()
	pb.RegisterGreetingServiceServer(s, &greetingServer{})

	fmt.Println("Server listening on :50051")
	if err := s.Serve(lis); err != nil {
		panic(err)
	}
}
