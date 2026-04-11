package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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
		Name:        "T",
		Description: "Get job template for a model (e.g. T openai/whisper-large-v3-turbo)",
		Execute: func(args []string) error {
			if len(args) < 1 {
				return fmt.Errorf("usage: T <model_name>")
			}
			taskType, inputType, err := fetchModelTemplate(args[0])
			if err != nil {
				return err
			}
			fmt.Printf("Model:      %s\n", args[0])
			fmt.Printf("Task type:  %s\n", taskType)
			fmt.Printf("Input:      %s\n", inputType)
			fmt.Printf("Output:     text\n")
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


var pipelineToTask = map[string][2]string{
	// Audio
	"automatic-speech-recognition": {"SPEECH_RECOGNITION",   "bytes"},
	"audio-classification":         {"AUDIO_CLASSIFICATION",  "bytes"},
	"text-to-speech":               {"TEXT_TO_SPEECH",        "text"},
	"text-to-audio":                {"TEXT_TO_AUDIO",         "text"},
	"keyword-spotting":             {"KEYWORD_SPOTTING",      "bytes"},

	// NLP
	"text-generation":              {"TEXT_INFERENCE",        "text"},
	"text2text-generation":         {"TEXT_INFERENCE",        "text"},
	"summarization":                {"SUMMARIZATION",         "text"},
	"translation":                  {"TRANSLATION",           "text"},
	"question-answering":           {"QUESTION_ANSWERING",    "text"},
	"text-classification":          {"TEXT_CLASSIFICATION",   "text"},
	"fill-mask":                    {"FILL_MASK",             "text"},

	// Computer Vision
	"image-classification":         {"IMAGE_CLASSIFICATION",  "bytes"},
	"object-detection":             {"OBJECT_DETECTION",      "bytes"},
	"image-to-text":                {"IMAGE_CAPTIONING",      "bytes"},
	"depth-estimation":             {"DEPTH_ESTIMATION",      "bytes"},
	"video-classification":         {"VIDEO_CLASSIFICATION",  "bytes"},
	"image-segmentation":           {"IMAGE_SEGMENTATION",    "bytes"},

	// Multimodal
	"visual-question-answering":    {"VISUAL_QA",             "bytes"},
	"document-question-answering":  {"DOCUMENT_QA",           "bytes"},
}

func fetchModelTemplate(modelName string) (taskType, inputType string, err error) {
	resp, err := http.Get("https://huggingface.co/api/models/" + modelName)
	if err != nil {
		return "", "", fmt.Errorf("failed to reach HuggingFace")
	}
	defer resp.Body.Close()

	var result struct {
		PipelineTag string `json:"pipeline_tag"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", fmt.Errorf("failed to parse model info")
	}

	entry, ok := pipelineToTask[result.PipelineTag]
	if !ok {
		return "", "", fmt.Errorf("unsupported pipeline type: %s", result.PipelineTag)
	}
	return entry[0], entry[1], nil
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
