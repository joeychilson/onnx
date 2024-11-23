# onnx

A Go library for automatically downloading initializing the ONNX runtime.

## Installation

```bash
go get -u github.com/joeychilson/onnx
```

## Usage

```go
package main

import (
	"context"
	"log"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/joeychilson/onnx"
)

func main() {
	ctx := context.Background()

	runtime, err := onnx.New(ctx)
	if err != nil {
		log.Fatal(err)
	}
	runtime.Close()

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		log.Fatal(err)
	}
	defer sessionOptions.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		"model.onnx",
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		sessionOptions,
	)
	if err != nil {
		log.Fatal(err)
	}
}
```
