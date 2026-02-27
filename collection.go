package collection

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"sync"
	_ "unicode"
)

// LazyOp is a function that takes an input slice and produces an output slice.
// This allows us to chain operations without applying them immediately.
type LazyOp[T any] func([]T) []T

// Collection represents a generic, lazy-evaluated collection.
type Collection[T any] struct {
	source []T         // The original data
	ops    []LazyOp[T] // List of operations to apply lazily
	kind   reflect.Kind
	shared bool
	t      T
}

// WithShared Careful with this one it will make the collection shared and affect all instances
func (c *Collection[T]) WithShared(shared bool) *Collection[T] {
	c.shared = shared
	return c
}

// NewCollection creates a new Collection from slices, arrays, maps, or structs.
//   - Slice/Array: each element is added as-is
//   - Map: converts key/value pairs into map[string]any
//   - Struct: converts fields into map[string]any
func NewCollection[T any](data any) *Collection[T] {
	var t T
	c := &Collection[T]{source: []T{}}
	c.t = t
	val := reflect.ValueOf(data)
	kind := val.Kind()

	switch kind {
	case reflect.Slice, reflect.Array:
		c.kind = kind
		for i := 0; i < val.Len(); i++ {
			c.source = append(c.source, val.Index(i).Interface().(T))
		}

	case reflect.Map:
		c.kind = reflect.Map
		for _, key := range val.MapKeys() {
			entry := any(map[string]any{
				"key":   key.Interface(),
				"value": val.MapIndex(key).Interface(),
			}).(T)
			c.source = append(c.source, entry)
		}

	case reflect.Struct, reflect.Ptr:
		if obj, ok := data.(T); ok {
			c.source = append(c.source, obj)
		} else {
			// Handle pointer to struct if T is the struct type
			if kind == reflect.Ptr && val.Elem().Kind() == reflect.Struct {
				if obj, ok := val.Elem().Interface().(T); ok {
					c.source = append(c.source, obj)
				} else {
					panic(fmt.Sprintf("collection: pointer type %T does not match collection type %T", data, *new(T)))
				}
			} else {
				panic(fmt.Sprintf("collection: struct type %T does not match collection type %T", data, *new(T)))
			}
		}
		c.kind = reflect.Struct

	default:
		panic(fmt.Sprintf("collection: unsupported type %s", kind))
	}

	return c
}

// ToSlice triggers execution of all lazy ops and returns a fully processed []T.
func (c *Collection[T]) ToSlice() []T {
	result := make([]T, len(c.source))
	copy(result, c.source)
	for i := range result {
		result[i], _ = DeepClone(result[i])
	}

	for _, op := range c.ops {
		result = op(result)
	}
	return result
}

// ForEach applies a function immediately (used for side effects).
func (c *Collection[T]) ForEach(fn func(T)) {
	for _, v := range c.ToSlice() {
		fn(v)
	}
}

// ForEachMap applies fn(key, value) for each map entry (only if original type was a map).
func (c *Collection[T]) ForEachMap(fn func(key any, value any)) {
	for _, entry := range c.source {
		if kv, ok := any(entry).(map[string]any); ok {
			fn(kv["key"], kv["value"])
		}
	}
}

// ToJson marshals the processed slice into JSON.
func (c *Collection[T]) ToJson() ([]byte, error) {
	items := c.ToSlice()
	if c.kind == reflect.Struct && len(items) == 1 {
		return json.Marshal(items[0])
	}
	return json.Marshal(items)
}

// ToPrettyJson marshals the collection into an indented (pretty-printed) JSON string.
func (c *Collection[T]) ToPrettyJson() ([]byte, error) {
	items := c.ToSlice()
	if c.kind == reflect.Struct && len(items) == 1 {
		return json.MarshalIndent(items[0], "", "  ")
	}
	if c.kind == reflect.Map {
		if len(items) == 1 {
			return json.MarshalIndent(items[0], "", "  ")
		}
	}
	return json.MarshalIndent(items, "", "  ")
}

// Count returns the length after applying all lazy operations.
func (c *Collection[T]) Count() int {
	return len(c.ToSlice())
}

// IsEmpty checks if collection has no items after lazy ops.
func (c *Collection[T]) IsEmpty() bool {
	return c.Count() == 0
}

// ParallelMap applies fn concurrently to all items.
func (c *Collection[T]) ParallelMap(fn func(T) T, workers int) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		out := make([]T, len(input))
		var wg sync.WaitGroup
		jobs := make(chan int, len(input))

		if workers <= 0 {
			workers = 1
		}

		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for idx := range jobs {
					res, _ := DeepClone(input[idx])
					out[idx] = fn(res)
				}
			}()
		}

		for i := range input {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		return out
	})
	return c
}

// Pluck extracts a field value by name from each item.
func (c *Collection[T]) Pluck(field string) []any {
	var results []any
	uField := UppercaseFirst(field)
	for _, item := range c.ToSlice() {
		val := reflect.ValueOf(item)
		typ := reflect.TypeOf(item)

		switch val.Kind() {
		case reflect.Map:
			if m, ok := any(item).(map[string]any); ok {
				if v, exists := m[field]; exists {
					results = append(results, v)
				}
			}
		case reflect.Struct:
			if f, ok := typ.FieldByName(field); ok && f.IsExported() {
				results = append(results, val.FieldByName(field).Interface())
			} else if f, ok := typ.FieldByName(uField); ok && f.IsExported() {
				results = append(results, val.FieldByName(uField).Interface())
			}
		default:

		}
	}
	return results
}

func (c *Collection[T]) GroupBy(fn func(T) any) map[any][]T {
	groups := make(map[any][]T)
	for _, item := range c.ToSlice() {
		key := fn(item)
		groups[key] = append(groups[key], item)
	}
	return groups
}

// KeyBy maps items into a map using fn(item) as the key.
func (c *Collection[T]) KeyBy(fn func(T) any) map[any]T {
	result := make(map[any]T)
	for _, item := range c.ToSlice() {
		key := fn(item)
		result[key] = item
	}
	return result
}

// Map lazily applies fn to each item.
func (c *Collection[T]) Map(fn func(T) T) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		out := make([]T, len(input))
		for i, v := range input {
			out[i] = fn(v)
		}
		return out
	})
	return c
}

// FlatMap lazily applies fn to each item and flattens the result.
func (c *Collection[T]) FlatMap(fn func(T) []T) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		for _, v := range input {
			out = append(out, fn(v)...)
		}
		return out
	})
	return c
}

// Reduce reduces the collection to a single value using an accumulator function.
func (c *Collection[T]) Reduce(fn func(acc any, curr T) any, initial any) any {
	res := initial
	for _, v := range c.ToSlice() {
		res = fn(res, v)
	}
	return res
}

func (c *Collection[T]) FindWithError(
	predicate func(T) (bool, error),
	throwError error,
) (T, error) {
	var zero T
	result := c.ToSlice()
	for _, item := range result {
		match, err := predicate(item)
		if err != nil {
			return zero, throwError
		}
		if match {
			return item, nil
		}
	}
	return zero, throwError
}

func (c *Collection[T]) Find(fn func(T) bool) (T, bool) {
	var zero T
	result := c.ToSlice()
	for _, item := range result {
		if fn(item) {
			return item, true
		}
	}
	return zero, false
}

// Filter lazily keeps only items where fn(item) == true.
func (c *Collection[T]) Filter(fn func(T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		out := make([]T, 0, len(input))
		for _, v := range input {
			if fn(v) {
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

// Reject lazily removes items where fn(item) == true.
func (c *Collection[T]) Reject(fn func(T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		for _, v := range input {
			if !fn(v) {
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

// Take lazily keeps only the first n items.
func (c *Collection[T]) Take(n int) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		if n <= 0 {
			return []T{}
		}
		if n > len(input) {
			return input
		}
		return input[:n]
	})
	return c
}

// Skip lazily skips the first n items.
func (c *Collection[T]) Skip(n int) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		if n <= 0 {
			return input
		}
		if n > len(input) {
			return []T{}
		}
		return input[n:]
	})
	return c
}

// First returns the first item in the collection.
func (c *Collection[T]) First() (T, bool) {
	var zero T
	items := c.ToSlice()
	if len(items) == 0 {
		return zero, false
	}
	return items[0], true
}

// Last returns the last item in the collection.
func (c *Collection[T]) Last() (T, bool) {
	var zero T
	items := c.ToSlice()
	if len(items) == 0 {
		return zero, false
	}
	return items[len(items)-1], true
}

// Contains checks if the collection contains an item matching the predicate.
func (c *Collection[T]) Contains(fn func(T) bool) bool {
	_, found := c.Find(fn)
	return found
}

// Reverse lazily reverses the order.
func (c *Collection[T]) Reverse() *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		n := len(input)
		for i := 0; i < n/2; i++ {
			input[i], input[n-1-i] = input[n-1-i], input[i]
		}
		return input
	})
	return c
}

// SortBy lazily sorts using a comparator.
func (c *Collection[T]) SortBy(fn func(a, b T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		sort.Slice(input, func(i, j int) bool {
			return fn(input[i], input[j])
		})
		return input
	})
	return c
}

// Unique lazily removes duplicates.
func (c *Collection[T]) Unique() *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		seen := make(map[any]bool)
		var out []T
		for _, v := range input {
			// Using reflect.ValueOf to handle non-comparable types if needed,
			// but for Unique, we'll stick to basic comparability or JSON string representation for safety.
			key := any(v)
			if !isComparable(v) {
				data, _ := json.Marshal(v)
				key = string(data)
			}
			if !seen[key] {
				seen[key] = true
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

func isComparable(v any) bool {
	if v == nil {
		return true
	}
	return reflect.TypeOf(v).Comparable()
}

// Chunk splits the collection into chunks of given size.
func (c *Collection[T]) Chunk(size int) [][]T {
	final := c.ToSlice()
	if size <= 0 {
		return [][]T{final}
	}
	var chunks [][]T
	for i := 0; i < len(final); i += size {
		end := i + size
		if end > len(final) {
			end = len(final)
		}
		chunks = append(chunks, final[i:end])
	}
	return chunks
}

func (c *Collection[T]) Kind() reflect.Kind {
	return c.kind
}

// Union combines two collections, removing duplicates.
func (c *Collection[T]) Union(other *Collection[T]) *Collection[T] {
	combined := append(c.ToSlice(), other.ToSlice()...)
	return NewCollection[T](combined).Unique()
}

// Intersect returns elements present in both collections.
func (c *Collection[T]) Intersect(other *Collection[T]) *Collection[T] {
	otherSlice := other.ToSlice()
	return c.Filter(func(item T) bool {
		for _, o := range otherSlice {
			if equalItems(item, o) {
				return true
			}
		}
		return false
	})
}

// Diff returns elements in this collection but not in other.
func (c *Collection[T]) Diff(other *Collection[T]) *Collection[T] {
	otherSlice := other.ToSlice()
	return c.Filter(func(item T) bool {
		for _, o := range otherSlice {
			if equalItems(item, o) {
				return false
			}
		}
		return true
	})
}

// SymmetricDiff returns elements in either collection but not in both.
func (c *Collection[T]) SymmetricDiff(other *Collection[T]) *Collection[T] {
	left := c.Diff(other)
	right := other.Diff(c)
	return left.Concat(right)
}

// Sum returns the sum of numeric values using a selector function.
func (c *Collection[T]) Sum(fn func(T) float64) float64 {
	var sum float64
	for _, v := range c.ToSlice() {
		sum += fn(v)
	}
	return sum
}

// Average returns the average of numeric values using a selector function.
func (c *Collection[T]) Average(fn func(T) float64) float64 {
	items := c.ToSlice()
	if len(items) == 0 {
		return 0
	}
	return c.Sum(fn) / float64(len(items))
}

// Min returns the minimum item using a comparator.
func (c *Collection[T]) Min(fn func(a, b T) bool) (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	minimum := items[0]
	for _, v := range items[1:] {
		if fn(v, minimum) {
			minimum = v
		}
	}
	return minimum, true
}

// Max returns the maximum item using a comparator.
func (c *Collection[T]) Max(fn func(a, b T) bool) (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	maximum := items[0]
	for _, v := range items[1:] {
		if fn(v, maximum) {
			maximum = v
		}
	}
	return maximum, true
}

// MinBy returns the item with minimum value from selector.
func (c *Collection[T]) MinBy(fn func(T) float64) (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	minimum := items[0]
	minVal := fn(minimum)
	for _, v := range items[1:] {
		if val := fn(v); val < minVal {
			minimum = v
			minVal = val
		}
	}
	return minimum, true
}

// MaxBy returns the item with maximum value from selector.
func (c *Collection[T]) MaxBy(fn func(T) float64) (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	maximum := items[0]
	maxVal := fn(maximum)
	for _, v := range items[1:] {
		if val := fn(v); val > maxVal {
			maximum = v
			maxVal = val
		}
	}
	return maximum, true
}

// All checks if all items match the predicate.
func (c *Collection[T]) All(fn func(T) bool) bool {
	for _, v := range c.ToSlice() {
		if !fn(v) {
			return false
		}
	}
	return true
}

// Any checks if any item matches the predicate.
func (c *Collection[T]) Any(fn func(T) bool) bool {
	_, found := c.Find(fn)
	return found
}

// None checks if no items match the predicate.
func (c *Collection[T]) None(fn func(T) bool) bool {
	return !c.Any(fn)
}

// Partition splits collection into two based on predicate.
func (c *Collection[T]) Partition(fn func(T) bool) ([]T, []T) {
	var pass, fail []T
	for _, v := range c.ToSlice() {
		if fn(v) {
			pass = append(pass, v)
		} else {
			fail = append(fail, v)
		}
	}
	return pass, fail
}

// Zip combines two collections element-wise.
func Zip[T, U any](c1 *Collection[T], c2 *Collection[U]) []struct {
	First  T
	Second U
} {
	s1 := c1.ToSlice()
	s2 := c2.ToSlice()
	minLen := len(s1)
	if len(s2) < minLen {
		minLen = len(s2)
	}
	result := make([]struct {
		First  T
		Second U
	}, minLen)
	for i := 0; i < minLen; i++ {
		result[i] = struct {
			First  T
			Second U
		}{s1[i], s2[i]}
	}
	return result
}

// ZipWith combines two collections using a custom function.
func ZipWith[T, U, R any](c1 *Collection[T], c2 *Collection[U], fn func(T, U) R) []R {
	s1 := c1.ToSlice()
	s2 := c2.ToSlice()
	minLen := len(s1)
	if len(s2) < minLen {
		minLen = len(s2)
	}
	result := make([]R, minLen)
	for i := 0; i < minLen; i++ {
		result[i] = fn(s1[i], s2[i])
	}
	return result
}

// Flatten flattens a collection of slices.
func (c *Collection[T]) Flatten() *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		for _, v := range input {
			val := reflect.ValueOf(v)
			if val.Kind() == reflect.Slice {
				for i := 0; i < val.Len(); i++ {
					if item, ok := val.Index(i).Interface().(T); ok {
						out = append(out, item)
					}
				}
			} else {
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

// DistinctBy removes duplicates based on key selector.
func (c *Collection[T]) DistinctBy(fn func(T) any) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		seen := make(map[any]bool)
		var out []T
		for _, v := range input {
			key := fn(v)
			if !isComparable(key) {
				data, _ := json.Marshal(key)
				key = string(data)
			}
			if !seen[key] {
				seen[key] = true
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

// Random returns a random element from the collection.
func (c *Collection[T]) Random() (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	return items[rand.Intn(len(items))], true
}

// Sample returns n random elements from the collection.
func (c *Collection[T]) Sample(n int) []T {
	items := c.ToSlice()
	if n <= 0 || len(items) == 0 {
		return []T{}
	}
	if n >= len(items) {
		return items
	}
	indices := rand.Perm(len(items))[:n]
	result := make([]T, n)
	for i, idx := range indices {
		result[i] = items[idx]
	}
	return result
}

// Shuffle randomizes the order of elements.
func (c *Collection[T]) Shuffle() *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		rand.Shuffle(len(input), func(i, j int) {
			input[i], input[j] = input[j], input[i]
		})
		return input
	})
	return c
}

// Page returns a specific page of elements.
func (c *Collection[T]) Page(pageNum, pageSize int) *Collection[T] {
	if pageNum < 1 {
		pageNum = 1
	}
	skip := (pageNum - 1) * pageSize
	return c.Skip(skip).Take(pageSize)
}

// TakeWhile takes elements while predicate is true.
func (c *Collection[T]) TakeWhile(fn func(T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		for _, v := range input {
			if !fn(v) {
				break
			}
			out = append(out, v)
		}
		return out
	})
	return c
}

// SkipWhile skips elements while predicate is true.
func (c *Collection[T]) SkipWhile(fn func(T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		skipping := true
		for _, v := range input {
			if skipping && fn(v) {
				continue
			}
			skipping = false
			out = append(out, v)
		}
		return out
	})
	return c
}

// Join joins string elements with separator.
func (c *Collection[T]) Join(separator string) string {
	items := c.ToSlice()
	var parts []string
	for _, v := range items {
		parts = append(parts, fmt.Sprint(v))
	}
	return strings.Join(parts, separator)
}

// JoinBy joins elements using custom string converter.
func (c *Collection[T]) JoinBy(fn func(T) string, separator string) string {
	items := c.ToSlice()
	var parts []string
	for _, v := range items {
		parts = append(parts, fn(v))
	}
	return strings.Join(parts, separator)
}

// IndexOf returns the index of first item matching predicate.
func (c *Collection[T]) IndexOf(fn func(T) bool) int {
	for i, v := range c.ToSlice() {
		if fn(v) {
			return i
		}
	}
	return -1
}

// LastIndexOf returns the index of last item matching predicate.
func (c *Collection[T]) LastIndexOf(fn func(T) bool) int {
	items := c.ToSlice()
	for i := len(items) - 1; i >= 0; i-- {
		if fn(items[i]) {
			return i
		}
	}
	return -1
}

// ElementAt returns element at index with bounds checking.
func (c *Collection[T]) ElementAt(index int) (T, bool) {
	items := c.ToSlice()
	var zero T
	if index < 0 || index >= len(items) {
		return zero, false
	}
	return items[index], true
}

// Batch processes elements in batches.
func (c *Collection[T]) Batch(size int, fn func([]T)) {
	chunks := c.Chunk(size)
	for _, chunk := range chunks {
		fn(chunk)
	}
}

// Window returns sliding windows of given size.
func (c *Collection[T]) Window(size int) [][]T {
	items := c.ToSlice()
	if size <= 0 || size > len(items) {
		return [][]T{}
	}
	var windows [][]T
	for i := 0; i <= len(items)-size; i++ {
		windows = append(windows, items[i:i+size])
	}
	return windows
}

// MapWithError maps with error handling.
func (c *Collection[T]) MapWithError(fn func(T) (T, error)) ([]T, error) {
	items := c.ToSlice()
	result := make([]T, len(items))
	for i, v := range items {
		var err error
		result[i], err = fn(v)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// FilterWithError filters with error handling.
func (c *Collection[T]) FilterWithError(fn func(T) (bool, error)) ([]T, error) {
	items := c.ToSlice()
	var result []T
	for _, v := range items {
		match, err := fn(v)
		if err != nil {
			return nil, err
		}
		if match {
			result = append(result, v)
		}
	}
	return result, nil
}

// TryReduce reduces with error propagation.
func (c *Collection[T]) TryReduce(fn func(acc any, curr T) (any, error), initial any) (any, error) {
	res := initial
	for _, v := range c.ToSlice() {
		var err error
		res, err = fn(res, v)
		if err != nil {
			return nil, err
		}
	}
	return res, nil
}

// Tap inspects elements without modifying them.
func (c *Collection[T]) Tap(fn func(T)) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		for _, v := range input {
			fn(v)
		}
		return input
	})
	return c
}

// ToMap converts collection to map using key and value extractors.
func (c *Collection[T]) ToMap(keyFn func(T) any, valueFn func(T) any) map[any]any {
	result := make(map[any]any)
	for _, v := range c.ToSlice() {
		result[keyFn(v)] = valueFn(v)
	}
	return result
}

// ToSet converts collection to a map-based set.
func (c *Collection[T]) ToSet() map[any]bool {
	result := make(map[any]bool)
	for _, v := range c.ToSlice() {
		key := any(v)
		if !isComparable(v) {
			data, _ := json.Marshal(v)
			key = string(data)
		}
		result[key] = true
	}
	return result
}

// ToChannel converts collection to a channel.
func (c *Collection[T]) ToChannel() <-chan T {
	ch := make(chan T)
	go func() {
		defer close(ch)
		for _, v := range c.ToSlice() {
			ch <- v
		}
	}()
	return ch
}

// Median returns the median value using a selector.
func (c *Collection[T]) Median(fn func(T) float64) (float64, bool) {
	items := c.ToSlice()
	if len(items) == 0 {
		return 0, false
	}
	values := make([]float64, len(items))
	for i, v := range items {
		values[i] = fn(v)
	}
	sort.Float64s(values)
	n := len(values)
	if n%2 == 0 {
		return (values[n/2-1] + values[n/2]) / 2, true
	}
	return values[n/2], true
}

// Mode returns the most common element.
func (c *Collection[T]) Mode() (T, bool) {
	items := c.ToSlice()
	var zero T
	if len(items) == 0 {
		return zero, false
	}
	freq := make(map[any]int)
	for _, v := range items {
		key := any(v)
		if !isComparable(v) {
			data, _ := json.Marshal(v)
			key = string(data)
		}
		freq[key]++
	}
	var maxCount int
	var mode T
	for _, v := range items {
		key := any(v)
		if !isComparable(v) {
			data, _ := json.Marshal(v)
			key = string(data)
		}
		if freq[key] > maxCount {
			maxCount = freq[key]
			mode = v
		}
	}
	return mode, true
}

// Frequencies returns count of each element.
func (c *Collection[T]) Frequencies() map[any]int {
	freq := make(map[any]int)
	for _, v := range c.ToSlice() {
		key := any(v)
		if !isComparable(v) {
			data, _ := json.Marshal(v)
			key = string(data)
		}
		freq[key]++
	}
	return freq
}

// Concat concatenates with another collection.
func (c *Collection[T]) Concat(other *Collection[T]) *Collection[T] {
	combined := append(c.ToSlice(), other.ToSlice()...)
	return NewCollection[T](combined)
}

// Chain chains operations from another collection.
func (c *Collection[T]) Chain(other *Collection[T]) *Collection[T] {
	c.ops = append(c.ops, other.ops...)
	return c
}

// Sort sorts the collection using natural ordering (requires comparable types).
func (c *Collection[T]) Sort(less func(a, b T) bool) *Collection[T] {
	return c.SortBy(less)
}

// BinarySearch performs binary search on a sorted collection.
// Returns the index and true if found, -1 and false otherwise.
// Collection must be sorted before calling this method.
func (c *Collection[T]) BinarySearch(target T, compare func(a, b T) int) (int, bool) {
	items := c.ToSlice()
	left, right := 0, len(items)-1

	for left <= right {
		mid := left + (right-left)/2
		cmp := compare(items[mid], target)

		if cmp == 0 {
			return mid, true
		} else if cmp < 0 {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1, false
}

// BinarySearchBy performs binary search using a key extractor.
func (c *Collection[T]) BinarySearchBy(keyFn func(T) float64, target float64) (int, bool) {
	items := c.ToSlice()
	left, right := 0, len(items)-1

	for left <= right {
		mid := left + (right-left)/2
		midVal := keyFn(items[mid])

		if midVal == target {
			return mid, true
		} else if midVal < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1, false
}

// QuickSort sorts the collection using quicksort algorithm.
func (c *Collection[T]) QuickSort(less func(a, b T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		if len(input) <= 1 {
			return input
		}
		quickSort(input, 0, len(input)-1, less)
		return input
	})
	return c
}

func quickSort[T any](arr []T, low, high int, less func(a, b T) bool) {
	if low < high {
		pi := partition(arr, low, high, less)
		quickSort(arr, low, pi-1, less)
		quickSort(arr, pi+1, high, less)
	}
}

func partition[T any](arr []T, low, high int, less func(a, b T) bool) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j < high; j++ {
		if less(arr[j], pivot) {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// MergeSort sorts the collection using merge sort algorithm.
func (c *Collection[T]) MergeSort(less func(a, b T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		if len(input) <= 1 {
			return input
		}
		return mergeSort(input, less)
	})
	return c
}

func mergeSort[T any](arr []T, less func(a, b T) bool) []T {
	if len(arr) <= 1 {
		return arr
	}

	mid := len(arr) / 2
	left := mergeSort(arr[:mid], less)
	right := mergeSort(arr[mid:], less)

	return merge(left, right, less)
}

func merge[T any](left, right []T, less func(a, b T) bool) []T {
	result := make([]T, 0, len(left)+len(right))
	i, j := 0, 0

	for i < len(left) && j < len(right) {
		if less(left[i], right[j]) {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	result = append(result, left[i:]...)
	result = append(result, right[j:]...)
	return result
}

// IsSorted checks if the collection is sorted according to the comparator.
func (c *Collection[T]) IsSorted(less func(a, b T) bool) bool {
	items := c.ToSlice()
	for i := 1; i < len(items); i++ {
		if less(items[i], items[i-1]) {
			return false
		}
	}
	return true
}

// Nth returns the nth smallest element using quickselect algorithm.
func (c *Collection[T]) Nth(n int, less func(a, b T) bool) (T, bool) {
	items := c.ToSlice()
	var zero T
	if n < 0 || n >= len(items) {
		return zero, false
	}
	return quickSelect(items, 0, len(items)-1, n, less), true
}

func quickSelect[T any](arr []T, low, high, k int, less func(a, b T) bool) T {
	if low == high {
		return arr[low]
	}

	pi := partition(arr, low, high, less)

	if k == pi {
		return arr[k]
	} else if k < pi {
		return quickSelect(arr, low, pi-1, k, less)
	}
	return quickSelect(arr, pi+1, high, k, less)
}

// TopK returns the top k elements using heap-based selection.
func (c *Collection[T]) TopK(k int, less func(a, b T) bool) []T {
	items := c.ToSlice()
	if k <= 0 || len(items) == 0 {
		return []T{}
	}
	if k >= len(items) {
		sort.Slice(items, func(i, j int) bool { return !less(items[i], items[j]) })
		return items
	}

	// Use partial sort for top k
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(items); j++ {
			if !less(items[j], items[maxIdx]) {
				maxIdx = j
			}
		}
		items[i], items[maxIdx] = items[maxIdx], items[i]
	}
	return items[:k]
}

// BottomK returns the bottom k elements.
func (c *Collection[T]) BottomK(k int, less func(a, b T) bool) []T {
	items := c.ToSlice()
	if k <= 0 || len(items) == 0 {
		return []T{}
	}
	if k >= len(items) {
		sort.Slice(items, func(i, j int) bool { return less(items[i], items[j]) })
		return items
	}

	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(items); j++ {
			if less(items[j], items[minIdx]) {
				minIdx = j
			}
		}
		items[i], items[minIdx] = items[minIdx], items[i]
	}
	return items[:k]
}

// HeapSort sorts the collection using heap sort algorithm.
func (c *Collection[T]) HeapSort(less func(a, b T) bool) *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		if len(input) <= 1 {
			return input
		}
		heapSort(input, less)
		return input
	})
	return c
}

func heapSort[T any](arr []T, less func(a, b T) bool) {
	n := len(arr)
	for i := n/2 - 1; i >= 0; i-- {
		heapify(arr, n, i, less)
	}
	for i := n - 1; i > 0; i-- {
		arr[0], arr[i] = arr[i], arr[0]
		heapify(arr, i, 0, less)
	}
}

func heapify[T any](arr []T, n, i int, less func(a, b T) bool) {
	largest := i
	left := 2*i + 1
	right := 2*i + 2

	if left < n && !less(arr[left], arr[largest]) {
		largest = left
	}
	if right < n && !less(arr[right], arr[largest]) {
		largest = right
	}
	if largest != i {
		arr[i], arr[largest] = arr[largest], arr[i]
		heapify(arr, n, largest, less)
	}
}

// TopKHeap returns top k elements using min-heap (O(n log k)).
func (c *Collection[T]) TopKHeap(k int, less func(a, b T) bool) []T {
	items := c.ToSlice()
	if k <= 0 || len(items) == 0 {
		return []T{}
	}
	if k >= len(items) {
		sort.Slice(items, func(i, j int) bool { return !less(items[i], items[j]) })
		return items
	}

	heap := make([]T, 0, k)
	for _, item := range items {
		if len(heap) < k {
			heap = append(heap, item)
			if len(heap) == k {
				// Build min heap
				for i := k/2 - 1; i >= 0; i-- {
					minHeapify(heap, len(heap), i, less)
				}
			}
		} else if !less(item, heap[0]) {
			heap[0] = item
			minHeapify(heap, len(heap), 0, less)
		}
	}
	sort.Slice(heap, func(i, j int) bool { return !less(heap[i], heap[j]) })
	return heap
}

func minHeapify[T any](arr []T, n, i int, less func(a, b T) bool) {
	smallest := i
	left := 2*i + 1
	right := 2*i + 2

	if left < n && less(arr[left], arr[smallest]) {
		smallest = left
	}
	if right < n && less(arr[right], arr[smallest]) {
		smallest = right
	}
	if smallest != i {
		arr[i], arr[smallest] = arr[smallest], arr[i]
		minHeapify(arr, n, smallest, less)
	}
}

// CountBy counts occurrences by key.
func (c *Collection[T]) CountBy(fn func(T) any) map[any]int {
	result := make(map[any]int)
	for _, v := range c.ToSlice() {
		key := fn(v)
		if !isComparable(key) {
			data, _ := json.Marshal(key)
			key = string(data)
		}
		result[key]++
	}
	return result
}

// MaxN returns n maximum elements.
func (c *Collection[T]) MaxN(n int, fn func(a, b T) bool) []T {
	return c.TopK(n, fn)
}

// MinN returns n minimum elements.
func (c *Collection[T]) MinN(n int, fn func(a, b T) bool) []T {
	return c.BottomK(n, fn)
}

// Compact removes zero values from collection.
func (c *Collection[T]) Compact() *Collection[T] {
	c.ops = append(c.ops, func(input []T) []T {
		var out []T
		var zero T
		for _, v := range input {
			if !reflect.DeepEqual(v, zero) {
				out = append(out, v)
			}
		}
		return out
	})
	return c
}

// DifferenceAll returns elements not in any of the other collections.
func (c *Collection[T]) DifferenceAll(others ...*Collection[T]) *Collection[T] {
	result := c
	for _, other := range others {
		result = result.Diff(other)
	}
	return result
}

// ScanLeft performs cumulative reduce (running totals).
func (c *Collection[T]) ScanLeft(fn func(acc any, curr T) any, initial any) []any {
	items := c.ToSlice()
	result := make([]any, len(items)+1)
	result[0] = initial
	acc := initial
	for i, v := range items {
		acc = fn(acc, v)
		result[i+1] = acc
	}
	return result
}

// Interleave merges collections alternating elements.
func (c *Collection[T]) Interleave(other *Collection[T]) *Collection[T] {
	s1 := c.ToSlice()
	s2 := other.ToSlice()
	result := make([]T, 0, len(s1)+len(s2))
	maxLen := len(s1)
	if len(s2) > maxLen {
		maxLen = len(s2)
	}
	for i := 0; i < maxLen; i++ {
		if i < len(s1) {
			result = append(result, s1[i])
		}
		if i < len(s2) {
			result = append(result, s2[i])
		}
	}
	return NewCollection[T](result)
}

// Memoize caches the ToSlice() result for performance.
func (c *Collection[T]) Memoize() *Collection[T] {
	cached := c.ToSlice()
	return NewCollection[T](cached)
}

// Pagination represents paginated data with metadata.
type Pagination[T any] struct {
	Data        []T               `json:"data" xml:"Data"`
	CurrentPage int               `json:"current_page" xml:"CurrentPage"`
	PerPage     int               `json:"per_page" xml:"PerPage"`
	Total       int               `json:"total" xml:"Total"`
	TotalPages  int               `json:"total_pages" xml:"TotalPages"`
	HasNext     bool              `json:"has_next" xml:"HasNext"`
	HasPrev     bool              `json:"has_prev" xml:"HasPrev"`
	fieldNames  map[string]string // For custom field names
}

// WithFieldNames sets custom field names for serialization (chainable).
func (p *Pagination[T]) WithFieldNames(fieldNames map[string]string) *Pagination[T] {
	p.fieldNames = fieldNames
	return p
}

// ToJson converts pagination to JSON.
func (p *Pagination[T]) ToJson() ([]byte, error) {
	if p.fieldNames != nil {
		return json.Marshal(p.ToCustomMap(p.fieldNames))
	}
	return json.Marshal(p)
}

// ToPrettyJson converts pagination to pretty-printed JSON.
func (p *Pagination[T]) ToPrettyJson() ([]byte, error) {
	if p.fieldNames != nil {
		return json.MarshalIndent(p.ToCustomMap(p.fieldNames), "", "  ")
	}

	return json.MarshalIndent(p, "", "  ")
}

// ToMap converts Pagination to a map for custom serialization.
func (p *Pagination[T]) ToMap() map[string]any {
	if p.fieldNames != nil {
		return p.ToCustomMap(p.fieldNames)
	}
	return map[string]any{
		"data":         p.Data,
		"current_page": p.CurrentPage,
		"per_page":     p.PerPage,
		"total":        p.Total,
		"total_pages":  p.TotalPages,
		"has_next":     p.HasNext,
		"has_prev":     p.HasPrev,
	}
}

// ToCustomMap converts Pagination to a map with custom field names.
func (p *Pagination[T]) ToCustomMap(fieldNames map[string]string) map[string]any {
	defaultNames := map[string]string{
		"data":         "data",
		"current_page": "current_page",
		"per_page":     "per_page",
		"total":        "total",
		"total_pages":  "total_pages",
		"has_next":     "has_next",
		"has_prev":     "has_prev",
	}

	// Merge custom names with defaults
	for k, v := range fieldNames {
		defaultNames[k] = v
	}

	return map[string]any{
		defaultNames["data"]:         p.Data,
		defaultNames["current_page"]: p.CurrentPage,
		defaultNames["per_page"]:     p.PerPage,
		defaultNames["total"]:        p.Total,
		defaultNames["total_pages"]:  p.TotalPages,
		defaultNames["has_next"]:     p.HasNext,
		defaultNames["has_prev"]:     p.HasPrev,
	}
}

// Paginate returns a Pagination struct with metadata.
func (c *Collection[T]) Paginate(page, perPage int) *Pagination[T] {
	items := c.ToSlice()
	total := len(items)

	if page < 1 {
		page = 1
	}
	if perPage < 1 {
		perPage = 10
	}

	totalPages := (total + perPage - 1) / perPage
	if totalPages == 0 {
		totalPages = 1
	}

	if page > totalPages {
		page = totalPages
	}

	start := (page - 1) * perPage
	end := start + perPage

	if start < 0 {
		start = 0
	}
	if end > total {
		end = total
	}

	var data []T
	if start < total {
		data = items[start:end]
	} else {
		data = []T{}
	}

	return &Pagination[T]{
		Data:        data,
		CurrentPage: page,
		PerPage:     perPage,
		Total:       total,
		TotalPages:  totalPages,
		HasNext:     page < totalPages,
		HasPrev:     page > 1,
	}
}

func equalItems[T any](a, b T) bool {
	if isComparable(a) && isComparable(b) {
		return any(a) == any(b)
	}
	dataA, _ := json.Marshal(a)
	dataB, _ := json.Marshal(b)
	return string(dataA) == string(dataB)
}
