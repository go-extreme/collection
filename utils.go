package collection

import (
	"encoding/json"
	"fmt"
	"reflect"
	"runtime/debug"
	"strconv"
	"strings"
	"unicode"
)

// GetValueByTagKey retrieves the value of a struct field by its tag key.
// It recursively searches through nested structs and pointers.
func GetValueByTagKey(data interface{}, tagKey string) (string, bool) {
	v := reflect.ValueOf(data)
	t := reflect.TypeOf(data)

	if t.Kind() == reflect.Ptr {
		v = v.Elem()
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return "", false
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		fieldValue := v.Field(i)

		// Check if the field has the tag key
		if _, ok := field.Tag.Lookup(tagKey); ok {
			if fieldValue.Kind() == reflect.String {
				return fieldValue.String(), true
			}
			return fmt.Sprintf("%v", fieldValue.Interface()), true
		}

		// If it's a struct, recurse
		if fieldValue.Kind() == reflect.Struct {
			if val, found := GetValueByTagKey(fieldValue.Interface(), tagKey); found {
				return val, true
			}
		}

		// If it's a pointer to a struct, recurse
		if fieldValue.Kind() == reflect.Ptr && !fieldValue.IsNil() && fieldValue.Elem().Kind() == reflect.Struct {
			if val, found := GetValueByTagKey(fieldValue.Interface(), tagKey); found {
				return val, true
			}
		}
	}

	return "", false
}

// MergeMaps takes any number of maps and merges them into one.
// Later maps override earlier ones for duplicate keys.
func MergeMaps(maps ...map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{})

	for _, m := range maps {
		for k, v := range m {
			merged[k] = v
		}
	}

	return merged
}

// DeepClone creates a deep copy of any value using JSON marshaling.
func DeepClone[T any](src T) (T, error) {
	var dst T
	data, err := json.Marshal(src)
	if err != nil {
		return dst, err
	}
	err = json.Unmarshal(data, &dst)
	return dst, err
}

// UppercaseFirst converts the first character of a string to uppercase.
func UppercaseFirst(s string) string {
	if s == "" {
		return ""
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// Is checks if two values are equal.
func Is(v any, t any) bool {
	return v == t
}

// IsTypeEqual checks if two values have the same type name.
func IsTypeEqual(v any, t any) bool {
	return reflect.TypeOf(v).Name() == reflect.TypeOf(t).Name()
}

// ToStructFromMap converts a map to a struct using JSON marshaling.
func ToStructFromMap[T any](m map[string]interface{}) (*T, error) {
	jsonBytes, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("error marshalling map to JSON: %w", err)
	}
	var target T
	if err := json.Unmarshal(jsonBytes, &target); err != nil {
		return nil, fmt.Errorf("error unmarshalling JSON to struct: %w", err)
	}

	return &target, nil
}

// ToStruct converts a map or JSON string to a struct.
func ToStruct[T any](data interface{}) (T, error) {
	var info T

	switch v := data.(type) {
	case map[string]interface{}:
		jsonData, err := json.Marshal(v)
		if err != nil {
			return info, err
		}

		err = json.Unmarshal(jsonData, &info)
		if err != nil {
			return info, err
		}

		return info, nil

	case string:
		err := json.Unmarshal([]byte(v), &info)
		if err != nil {
			return info, err
		}

		return info, nil
	default:
		return info, fmt.Errorf("unsupported type for data: %T", v)
	}
}

// Atoi converts a string to an integer, returning 0 on error.
func Atoi(s string) int {
	i, err := strconv.Atoi(s)
	if err != nil {
		fmt.Printf("Error converting string '%s' to int: %v", s, err)
		return 0
	}
	return i
}

// Atof converts a string to a float64, returning 0 on error.
func Atof(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		fmt.Printf("Error converting string '%s' to float64: %v", s, err)
		return 0
	}
	return f
}

// Itoa converts an integer to a string.
func Itoa(i int) string {
	return strconv.Itoa(i)
}

// GetMapValue retrieves a nested value from a map using dot notation path.
func GetMapValue(data map[string]interface{}, path string) (interface{}, bool) {
	keys := strings.Split(path, ".")
	var current interface{} = data

	for _, key := range keys {
		switch currMap := current.(type) {
		case map[string]interface{}:
			val, ok := currMap[key]
			if !ok {
				return nil, false
			}
			current = val
		case map[string]string:
			val, ok := currMap[key]
			if !ok {
				return nil, false
			}
			current = val
		default:
			return nil, false
		}
	}

	return current, true
}

// StackFrame represents a single frame in a stack trace.
type StackFrame struct {
	File     string `json:"file"`
	Line     int    `json:"line"`
	Function string `json:"function"`
}

// BuildStructuredStack builds a structured JSON representation of the current stack trace.
func BuildStructuredStack() json.RawMessage {
	raw := debug.Stack()
	lines := strings.Split(string(raw), "\n")

	var frames []StackFrame

	for i := 1; i < len(lines)-1; i += 2 {
		fn := strings.TrimSpace(lines[i])
		loc := strings.TrimSpace(lines[i+1])

		parts := strings.Split(loc, ":")
		if len(parts) < 2 {
			continue
		}

		file := parts[0]
		lineStr := parts[1]
		lineNum := 0
		if idx := strings.Index(lineStr, " "); idx != -1 {
			lineStr = lineStr[:idx]
		}
		fmt.Sscanf(lineStr, "%d", &lineNum)

		frame := StackFrame{
			File:     file,
			Line:     lineNum,
			Function: fn,
		}
		frames = append(frames, frame)
	}

	encoded, err := json.Marshal(frames)
	if err != nil {
		fallback, _ := json.Marshal(string(raw))
		return fallback
	}
	return encoded
}

// ToNullString converts an empty string pointer to nil.
func ToNullString(s *string) *string {
	if s != nil && *s == "" {
		return nil
	}
	return s
}
