package processor

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type Command struct {
	Name        string
	Description string
	Execute     func(args []string) error
}

type CommandProcessor struct {
	commands map[string]Command
	running  bool
}

func NewCommandProcessor() *CommandProcessor {
	cp := &CommandProcessor{
		commands: make(map[string]Command),
	}

	// default commands
	cp.Register(Command{
		Name:        "W",
		Description: "Freeze for N milliseconds (e.g. W 4000)",
		Execute: func(args []string) error {
			if len(args) == 0 {
				return fmt.Errorf("usage: W <milliseconds>")
			}
			n, err := strconv.Atoi(args[0])
			if err != nil {
				return fmt.Errorf("invalid duration: %s", args[0])
			}
			time.Sleep(time.Duration(n) * time.Millisecond)
			return nil
		},
	})

	cp.Register(Command{
		Name:        "X",
		Description: "Shut down the command processor",
		Execute: func(args []string) error {
			cp.Stop()
			return nil
		},
	})

	return cp
}

func (cp *CommandProcessor) Register(cmd Command) {
	cp.commands[cmd.Name] = cmd
}

func (cp *CommandProcessor) Run() {
	cp.running = true
	scanner := bufio.NewScanner(os.Stdin)

	for cp.running {
		fmt.Print("> ")
		scanner.Scan()
		parts := strings.Fields(scanner.Text())
		if len(parts) == 0 {
			continue
		}

		name, args := parts[0], parts[1:]

		cmd, ok := cp.commands[name]
		if !ok {
			fmt.Println("Unknown command. Available commands:")
			for _, cmd := range cp.commands {
				fmt.Println(" ", cmd.Name, " - ", cmd.Description)
			}
			continue
		}

		if err := cmd.Execute(args); err != nil {
			fmt.Println("Error:", err)
		}
	}
}

func (cp *CommandProcessor) Stop() {
	cp.running = false
}
