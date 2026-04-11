package main

import (
	"context"
	"fmt"
	"net"
	"sync"

	pb "ignis/common/proto"

	"github.com/google/uuid"
	"google.golang.org/grpc"
)

type providerConn struct {
	workerId string
	jobChan  chan *pb.JobAssignment
}

type IgnisServer struct {
	pb.UnimplementedIgnisServiceServer

	mu         sync.Mutex
	providers  map[string]*providerConn
	jobResults map[string]string
}

func NewIgnisServer() *IgnisServer {
	return &IgnisServer{
		providers:  make(map[string]*providerConn),
		jobResults: make(map[string]string),
	}
}

func (s *IgnisServer) Greet(_ context.Context, req *pb.GreetRequest) (*pb.GreetResponse, error) {
	fmt.Printf("[%s]: %s\n", req.Sender, req.Message)
	return &pb.GreetResponse{Reply: "Hello, " + req.Sender + "!"}, nil
}

func (s *IgnisServer) Subscribe(req *pb.SubscribeRequest, stream pb.IgnisService_SubscribeServer) error {
	workerId := uuid.New().String()
	conn := &providerConn{
		workerId: workerId,
		jobChan:  make(chan *pb.JobAssignment, 10),
	}

	s.mu.Lock()
	s.providers[workerId] = conn
	s.mu.Unlock()
	fmt.Printf("[%s]: Provider subscribed\n", workerId)

	for {
		select {
		case job := <-conn.jobChan:
			if err := stream.Send(job); err != nil {
				s.removeProvider(workerId)
				return err
			}
		case <-stream.Context().Done():
			s.removeProvider(workerId)
			fmt.Printf("[%s]: Provider disconnected\n", workerId)
			return nil
		}
	}
}

func (s *IgnisServer) Unsubscribe(_ context.Context, req *pb.UnsubscribeRequest) (*pb.UnsubscribeResponse, error) {
	s.removeProvider(req.WorkerId)
	fmt.Printf("[%s]: Provider unsubscribed\n", req.WorkerId)
	return &pb.UnsubscribeResponse{Ack: "ACK " + req.WorkerId}, nil
}

func (s *IgnisServer) removeProvider(workerId string) {
	s.mu.Lock()
	delete(s.providers, workerId)
	s.mu.Unlock()
}

func (s *IgnisServer) Job(_ context.Context, req *pb.JobRequest) (*pb.JobResponse, error) {
	s.mu.Lock()
	var target *providerConn
	for _, p := range s.providers {
		target = p
		break
	}
	s.mu.Unlock()

	if target == nil {
		return nil, fmt.Errorf("no providers available")
	}

	jobId := uuid.New().String()
	target.jobChan <- &pb.JobAssignment{
		JobId:      jobId,
		ModelName:  req.ModelName,
		InputData:  req.InputData,
		TaskType:   req.Tasktype,
		Config:     req.Config,
		InputBytes: req.InputBytes,
	}
	fmt.Printf("[job:%s] Dispatched to provider:%s\n", jobId, target.workerId)
	return &pb.JobResponse{JobId: jobId}, nil
}

func (s *IgnisServer) SubmitResult(_ context.Context, req *pb.SubmitResultRequest) (*pb.SubmitResultResponse, error) {
	s.mu.Lock()
	s.jobResults[req.JobId] = req.Output
	s.mu.Unlock()
	fmt.Printf("[job:%s] Result received\n", req.JobId)
	return &pb.SubmitResultResponse{Ack: "ACK result for " + req.JobId}, nil
}

func (s *IgnisServer) GetResult(_ context.Context, req *pb.GetResultRequest) (*pb.GetResultResponse, error) {
	s.mu.Lock()
	output, ready := s.jobResults[req.JobId]
	s.mu.Unlock()
	return &pb.GetResultResponse{Ready: ready, Output: output}, nil
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
