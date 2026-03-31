package main

import (
	"context"
	"fmt"
	"net"

	pb "ignis/common/proto"

	"google.golang.org/grpc"

	"github.com/google/uuid"
)

type IgnisServer struct {
	pb.UnimplementedIgnisServiceServer
	availableWorkers map[string]struct{}
}

func NewIgnisServer() *IgnisServer {
	return &IgnisServer{
		availableWorkers: make(map[string]struct{}),
	}
}

func (s *IgnisServer) Greet(_ context.Context, req *pb.GreetRequest) (*pb.GreetResponse, error) {
	fmt.Printf("[%s]: %s\n", req.Sender, req.Message)
	return &pb.GreetResponse{Reply: "Hello, " + req.Sender + "!"}, nil
}

func (s *IgnisServer) Work(_ context.Context, req *pb.WorkRequest) (*pb.WorkResponse, error) {
	fmt.Printf("[%s]: Work Request\n", req.Sender)
	var newWorkerId string = uuid.New().String()
	s.availableWorkers[newWorkerId] = struct{}{}
	fmt.Printf("[%s]: Added New Worker\n", newWorkerId)
	return &pb.WorkResponse{Ack: "ACK " + req.Sender + " Work Request", WorkerId: newWorkerId}, nil
}

func (s *IgnisServer) Stop(_ context.Context, req *pb.StopRequest) (*pb.StopResponse, error) {
	fmt.Printf("[%s]: Work Request\n", req.WorkerId)
	_, exists := s.availableWorkers[req.WorkerId]
	if exists {
		delete(s.availableWorkers, req.WorkerId)
		fmt.Printf("[%s]: Removed Worker\n", req.WorkerId)
		return &pb.StopResponse{Ack: "ACK " + req.WorkerId + " Stop Request"}, nil
	}
	return nil, fmt.Errorf("worker not found")
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		panic(err)
	}

	s := grpc.NewServer()
	pb.RegisterIgnisServiceServer(s, NewIgnisServer())

	fmt.Println("Server listening on :50051")
	if err := s.Serve(lis); err != nil {
		panic(err)
	}
}
