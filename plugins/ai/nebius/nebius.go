package nebius

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/danielmiessler/fabric/common"
	"github.com/danielmiessler/fabric/plugins"
	"github.com/samber/lo"
	goopenai "github.com/sashabaranov/go-openai"
)

// NewClient creates a new Nebius client with default settings
func NewClient() (ret *Client) {
	return NewClientCompatible("Nebius", "https://api.studio.nebius.ai/v1", nil)
}

// NewClientCompatible creates a new Nebius client with custom settings
func NewClientCompatible(vendorName string, defaultBaseUrl string, configureCustom func() error) (ret *Client) {
	ret = &Client{}
	if configureCustom == nil {
		configureCustom = ret.configure
	}
	ret.PluginBase = &plugins.PluginBase{
		Name:            vendorName,
		EnvNamePrefix:   plugins.BuildEnvVariablePrefix(vendorName),
		ConfigureCustom: configureCustom,
	}
	ret.ApiKey = ret.AddSetupQuestion("API Key", true)
	ret.ApiBaseURL = ret.AddSetupQuestion("API Base URL", false)
	ret.ApiBaseURL.Value = defaultBaseUrl
	return
}

// Client represents a Nebius API client
type Client struct {
	*plugins.PluginBase
	ApiKey     *plugins.SetupQuestion
	ApiBaseURL *plugins.SetupQuestion
	ApiClient  *goopenai.Client
}

// configure sets up the Nebius client with the provided configuration
func (n *Client) configure() (ret error) {
	config := goopenai.DefaultConfig(n.ApiKey.Value)
	if n.ApiBaseURL.Value != "" {
		config.BaseURL = n.ApiBaseURL.Value
	}
	n.ApiClient = goopenai.NewClientWithConfig(config)
	return
}

// ListModels returns a list of available Nebius models
func (n *Client) ListModels() (ret []string, err error) {
	var models goopenai.ModelsList
	if models, err = n.ApiClient.ListModels(context.Background()); err != nil {
		return
	}
	// Nebius-specific model filtering could be added here
	for _, mod := range models.Models {
		// You might want to filter only for Nebius-specific models
		if isNebiusModel(mod.ID) {
			ret = append(ret, mod.ID)
		}
	}
	return
}

// isNebiusModel checks if a model ID belongs to Nebius
func isNebiusModel(modelID string) bool {
	// Add logic to identify Nebius models
	// Based on the documentation, models start with "meta-llama/" or "mistralai/"
	nebiusPrefixes := []string{
		"meta-llama/",
		"mistralai/",
		"deepseek-ai/",
		"microsoft/",
		"allenai/",
	}
	
	for _, prefix := range nebiusPrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return true
		}
	}
	return false
}

// SendStream sends a streaming request to the Nebius API
func (n *Client) SendStream(
	msgs []*common.Message, opts *common.ChatOptions, channel chan string,
) (err error) {
	req := n.buildChatCompletionRequest(msgs, opts)
	req.Stream = true
	var stream *goopenai.ChatCompletionStream
	if stream, err = n.ApiClient.CreateChatCompletionStream(context.Background(), req); err != nil {
		fmt.Printf("ChatCompletionStream error: %v\n", err)
		return
	}
	defer stream.Close()

	for {
		var response goopenai.ChatCompletionStreamResponse
		if response, err = stream.Recv(); err == nil {
			if len(response.Choices) > 0 {
				channel <- response.Choices[0].Delta.Content
			} else {
				channel <- "\n"
				close(channel)
				break
			}
		} else if errors.Is(err, io.EOF) {
			channel <- "\n"
			close(channel)
			err = nil
			break
		} else if err != nil {
			fmt.Printf("\nStream error: %v\n", err)
			break
		}
	}
	return
}

// Send sends a non-streaming request to the Nebius API
func (n *Client) Send(ctx context.Context, msgs []*common.Message, opts *common.ChatOptions) (ret string, err error) {
	req := n.buildChatCompletionRequest(msgs, opts)
	var resp goopenai.ChatCompletionResponse
	if resp, err = n.ApiClient.CreateChatCompletion(ctx, req); err != nil {
		return
	}
	if len(resp.Choices) > 0 {
		ret = resp.Choices[0].Message.Content
		slog.Debug("SystemFingerprint: " + resp.SystemFingerprint)
	}
	return
}

// buildChatCompletionRequest creates a chat completion request for Nebius
func (n *Client) buildChatCompletionRequest(
	msgs []*common.Message, opts *common.ChatOptions,
) (ret goopenai.ChatCompletionRequest) {
	messages := lo.Map(msgs, func(message *common.Message, _ int) goopenai.ChatCompletionMessage {
		return goopenai.ChatCompletionMessage{Role: message.Role, Content: message.Content}
	})

	if opts.Raw {
		ret = goopenai.ChatCompletionRequest{
			Model:    opts.Model,
			Messages: messages,
		}
	} else {
		// Nebius supports additional parameters like top_k
		ret = goopenai.ChatCompletionRequest{
			Model:            opts.Model,
			Temperature:      float32(opts.Temperature),
			TopP:            float32(opts.TopP),
			PresencePenalty:  float32(opts.PresencePenalty),
			FrequencyPenalty: float32(opts.FrequencyPenalty),
			Messages:         messages,
		}

		if opts.Seed != 0 {
			ret.Seed = &opts.Seed
		}
	}
	return
}