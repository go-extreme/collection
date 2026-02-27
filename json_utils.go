package collection

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// JSONError represents a JSON-related error with formatting capabilities.
type JSONError struct {
	Err error  `json:"-"`
	Msg string `json:"error"`
}

func (e *JSONError) Error() string {
	return e.Msg
}

// Formatted returns the JSON-formatted string of the error.
func (e *JSONError) Formatted() string {
	data, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, e.Msg)
	}
	return string(data)
}

// IsJSON checks if the string is valid JSON.
func IsJSON(input string) bool {
	if input == "" {
		return false
	}
	input = strings.TrimSpace(input)
	if !strings.HasPrefix(input, "{") && !strings.HasPrefix(input, "[") {
		return false
	}

	var js json.RawMessage
	return json.Unmarshal([]byte(input), &js) == nil
}

// JSONDecode decodes JSON recursively up to maxDepth.
// Returns the decoded interface{} and any error encountered.
func JSONDecode(input string, maxDepth int) (interface{}, error) {
	if maxDepth <= 0 {
		maxDepth = 1
	}

	var result interface{}
	current := strings.TrimSpace(input)

	if current == "" {
		return nil, &JSONError{Err: fmt.Errorf("empty input"), Msg: "empty input"}
	}

	for depth := 0; depth < maxDepth; depth++ {
		var decoded interface{}
		err := json.Unmarshal([]byte(current), &decoded)
		if err != nil {
			if depth == 0 {
				return current, nil
			}
			return result, nil
		}

		result = decoded

		if str, ok := decoded.(string); ok {
			str = strings.TrimSpace(str)
			if str == "" {
				return SafeToJSONString(result), nil
			}
			current = str
		} else {
			break
		}
	}

	return result, nil
}

func formatJsonError(err error) *JSONError {
	var syntaxErr *json.SyntaxError
	var typeErr *json.UnmarshalTypeError

	switch {
	case errors.As(err, &syntaxErr):
		return &JSONError{
			Err: err,
			Msg: fmt.Sprintf("syntax error at byte offset: %d", syntaxErr.Offset),
		}
	case errors.As(err, &typeErr):
		return &JSONError{
			Err: err,
			Msg: fmt.Sprintf("type error: expected=%v, got=%v, field=%s",
				typeErr.Type, typeErr.Value, typeErr.Field),
		}
	default:
		return &JSONError{
			Err: err,
			Msg: err.Error(),
		}
	}
}

func normalizeForJSON(v any) any {
	switch val := v.(type) {
	case map[any]any:
		m := make(map[string]any)
		for k, v := range val {
			m[fmt.Sprintf("%v", k)] = normalizeForJSON(v)
		}
		return m
	case []any:
		for i := range val {
			val[i] = normalizeForJSON(val[i])
		}
		return val
	default:
		return val
	}
}

// SafeToJSON converts any value to JSON safely, handling errors and wrapping them as JSONError.
func SafeToJSON(v interface{}) ([]byte, error) {
	if v == nil || v == "" {
		return []byte("{}"), nil
	}

	switch val := v.(type) {
	case map[string]interface{}:
		if len(val) == 0 {
			return []byte("{}"), nil
		}
		converted := make(map[string]any)
		for k, v := range val {
			keyStr := fmt.Sprintf("%v", k)
			converted[keyStr] = normalizeForJSON(v)
		}

		b, err := json.Marshal(converted)
		if err != nil {
			return nil, &JSONError{
				Err: err,
				Msg: fmt.Sprintf("failed to marshal map[any]any: %s", err.Error()),
			}
		}
		return b, nil
	case []byte:
		if json.Valid(val) {
			return val, nil
		}
		b, err := json.Marshal(string(val))
		if err != nil {
			return nil, &JSONError{
				Err: err,
				Msg: fmt.Sprintf("failed to marshal []byte: %s", err.Error()),
			}
		}
		return b, nil

	case string:
		if v == "" {
			return []byte("{}"), nil
		}
		if json.Valid([]byte(val)) {
			return []byte(val), nil
		}
		b, err := json.Marshal(val)
		if err != nil {
			return nil, &JSONError{
				Err: err,
				Msg: fmt.Sprintf("failed to marshal string: %s", err.Error()),
			}
		}
		return b, nil

	case json.RawMessage:
		if len(val) == 0 || string(val) == "" {
			return []byte("{}"), nil
		}
		return val, nil

	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, &JSONError{
				Err: err,
				Msg: fmt.Sprintf("%s failed to marshal to JSON: %s", reflect.TypeOf(v).String(), err.Error()),
			}
		}
		return b, nil
	}
}

// SafeToJSONString marshals the value to JSON, returning a string.
// It handles errors by returning an empty JSON object.
func SafeToJSONString(v interface{}) string {
	b, err := SafeToJSON(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// ParseJSON safely parses a JSON string into the target interface.
func ParseJSON(data string, target interface{}) error {
	if data == "" {
		return &JSONError{
			Err: fmt.Errorf("empty JSON string"),
			Msg: "empty JSON string",
		}
	}

	decoder := json.NewDecoder(bytes.NewReader([]byte(data)))
	decoder.UseNumber()
	if err := decoder.Decode(target); err != nil {
		return formatJsonError(err)
	}
	return nil
}

// MergeJSON merges multiple JSON objects into one.
func MergeJSON(jsons ...string) ([]byte, error) {
	result := make(map[string]interface{})

	for _, j := range jsons {
		if j == "" {
			continue
		}

		var m map[string]interface{}
		if err := ParseJSON(j, &m); err != nil {
			var je *JSONError
			if errors.As(err, &je) {
				return nil, errors.New(je.Formatted())
			}
			return nil, err
		}

		for k, v := range m {
			result[k] = v
		}
	}

	return SafeToJSON(result)
}

// ToJSON marshals a value with indentation.
func ToJSON(v interface{}) ([]byte, error) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, &JSONError{
			Err: err,
			Msg: fmt.Sprintf("%s failed to marshal to JSON: %s", reflect.TypeOf(v).String(), err.Error()),
		}
	}
	return data, nil
}

// PrettyPrintJSON prints a value as formatted JSON.
func PrettyPrintJSON(data interface{}) {
	json1, _ := ToJSON(data)
	fmt.Println(string(json1))
}

// RemoveKeysFromJSON removes specified keys from a JSON byte array.
func RemoveKeysFromJSON(body []byte, keys ...string) ([]byte, error) {
	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	for _, key := range keys {
		delete(data, key)
	}

	return json.Marshal(data)
}

// MustToJSON marshals a value with indentation, returning nil on error.
func MustToJSON(v interface{}) []byte {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil
	}
	return data
}

// ToNormalJSON marshals a value without indentation.
func ToNormalJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}

	return data
}

// FromJSON unmarshal data into a given target.
func FromJSON(data []byte, v interface{}) error {
	err := json.Unmarshal(data, v)
	if err != nil {
		return &JSONError{
			Err: err,
			Msg: fmt.Sprintf("%s failed to unmarshal from JSON: %s", reflect.TypeOf(v).String(), err.Error()),
		}
	}
	return nil
}
