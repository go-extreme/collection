# Go Collection Package

A powerful, generic, lazy-evaluated collection library for Go, offering a fluent API for data manipulation similar to Java Streams, C# LINQ, or JavaScript lodash. With **95+ methods**, it provides comprehensive functionality for filtering, mapping, aggregating, and transforming data with type safety and performance.

## Features

- **Generic**: Works with any data type `T` using Go generics
- **Lazy Evaluation**: Operations are chained and executed only when a terminal operation is called
- **Fluent API**: Chain multiple operations for concise and readable code
- **Concurrency**: `ParallelMap` for concurrent processing of items
- **Flexible Initialization**: Create collections from slices, arrays, maps, or structs
- **Rich API**: 95+ methods covering transformations, aggregations, set operations, and more
- **Advanced Algorithms**: Multiple sorting algorithms, binary search, heap operations
- **Performance Optimized**: Heap-based TopK (O(n log k)), QuickSelect (O(n)), Memoization
- **Chainable Pagination**: Built-in pagination with customizable field names
- **Well-Tested**: Comprehensive test suite with 90+ tests

## Installation

To use this package, you need to have Go installed. Then, you can add it to your project:

```bash
go get github.com/go-extreme/collection
```

## Usage

### Creating a Collection

You can create a collection from various data types:

```go
package main

import (
	"fmt"
	"github.com/go-extreme/collection"
)

type User struct {
	ID   int
	Name string
	Age  int
}

func main() {
	// From a slice
	sliceData := []int{1, 2, 3, 4, 5}
	collectionFromSlice := collection.NewCollection[int](sliceData)
	fmt.Println("Collection from slice:", collectionFromSlice.ToSlice())

	// From a struct
	userData := User{ID: 1, Name: "Alice", Age: 30}
	collectionFromStruct := collection.NewCollection[User](userData)
	fmt.Println("Collection from struct:", collectionFromStruct.ToSlice())

	// From a map
	mapData := map[string]int{"one": 1, "two": 2}
	// For maps, the collection elements will be of type map[string]any with "key" and "value" fields
	collectionFromMap := collection.NewCollection[map[string]any](mapData)
	fmt.Println("Collection from map:", collectionFromMap.ToSlice())
}
```

### Chaining Operations

```go
package main

import (
	"fmt"
	"github.com/go-extreme/collection"
)

type Product struct {
	ID    int
	Name  string
	Price float64
	Stock int
}

func main() {
	products := []Product{
		{ID: 1, Name: "Laptop", Price: 1200.00, Stock: 10},
		{ID: 2, Name: "Mouse", Price: 25.00, Stock: 50},
		{ID: 3, Name: "Keyboard", Price: 75.00, Stock: 20},
		{ID: 4, Name: "Monitor", Price: 300.00, Stock: 5},
	}

	// Basic filtering and mapping
	expensiveProducts := collection.NewCollection[Product](products).
		Filter(func(p Product) bool { return p.Price > 50.00 }).
		SortBy(func(a, b Product) bool { return a.Price < b.Price }).
		ToSlice()
	fmt.Println("Expensive products:", expensiveProducts)

	// Aggregations
	c := collection.NewCollection[Product](products)
	avgPrice := c.Average(func(p Product) float64 { return p.Price })
	totalValue := c.Sum(func(p Product) float64 { return p.Price * float64(p.Stock) })
	fmt.Printf("Average price: $%.2f, Total inventory value: $%.2f\n", avgPrice, totalValue)

	// Statistical operations
	medianPrice, _ := c.Median(func(p Product) float64 { return p.Price })
	fmt.Printf("Median price: $%.2f\n", medianPrice)

	// Predicates
	allInStock := c.All(func(p Product) bool { return p.Stock > 0 })
	hasExpensive := c.Any(func(p Product) bool { return p.Price > 1000 })
	fmt.Printf("All in stock: %v, Has expensive items: %v\n", allInStock, hasExpensive)

	// Grouping
	byPriceRange := c.GroupBy(func(p Product) any {
		if p.Price < 50 {
			return "Budget"
		} else if p.Price < 500 {
			return "Mid-range"
		}
		return "Premium"
	})
	fmt.Println("Products by price range:", byPriceRange)

	// Pagination
	page1 := collection.NewCollection[Product](products).Page(1, 2).ToSlice()
	fmt.Println("Page 1:", page1)

	// Pagination with metadata
	paginated := collection.NewCollection[Product](products).Paginate(1, 2)
	fmt.Printf("Page %d of %d (Total: %d items)\n", 
		paginated.CurrentPage, paginated.TotalPages, paginated.Total)
	fmt.Println("Data:", paginated.Data)
	
	// Chainable pagination with custom field names
	jsonData, _ := collection.NewCollection[Product](products).
		Filter(func(p Product) bool { return p.Price > 50 }).
		Paginate(1, 2).
		WithFieldNames(map[string]string{
			"data": "items",
			"current_page": "page",
			"total": "count",
		}).
		ToPrettyJson()
	fmt.Println(string(jsonData))

	// Set operations
	c1 := collection.NewCollection[int]([]int{1, 2, 3, 4})
	c2 := collection.NewCollection[int]([]int{3, 4, 5, 6})
	fmt.Println("Union:", c1.Union(c2).ToSlice())
	fmt.Println("Intersection:", c1.Intersect(c2).ToSlice())
	fmt.Println("Difference:", c1.Diff(c2).ToSlice())

	// Parallel processing
	processed := collection.NewCollection[Product](products).
		ParallelMap(func(p Product) Product {
			p.Price = p.Price * 1.1 // 10% price increase
			return p
		}, 4).
		ToSlice()
	fmt.Println("Processed products:", processed)

	// String operations
	names := collection.NewCollection[Product](products).Pluck("Name")
	fmt.Println("Product names:", names)
	namesList := collection.NewCollection[Product](products).
		JoinBy(func(p Product) string { return p.Name }, ", ")
	fmt.Println("Names joined:", namesList)
}
```

## API Reference

### Core Operations
- `NewCollection[T](data)` - Create a new collection
- `ToSlice()` - Execute all operations and return slice
- `ToJson()` / `ToPrettyJson()` - Convert to JSON
- `Count()` / `IsEmpty()` - Get size information
- `ForEach(fn)` - Iterate with side effects

### Transformation
- `Map(fn)` - Transform each element
- `FlatMap(fn)` - Transform and flatten
- `Filter(fn)` / `Reject(fn)` - Filter elements
- `Unique()` / `DistinctBy(fn)` - Remove duplicates
- `Flatten()` - Flatten nested collections
- `Reverse()` - Reverse order
- `SortBy(fn)` - Sort with comparator
- `Shuffle()` - Randomize order

### Aggregation
- `Reduce(fn, initial)` - Reduce to single value
- `Sum(fn)` / `Average(fn)` - Numeric aggregations
- `Min(fn)` / `Max(fn)` - Find extremes with comparator
- `MinBy(fn)` / `MaxBy(fn)` - Find extremes by value
- `Median(fn)` - Calculate median
- `Mode()` - Find most common element
- `Frequencies()` - Count occurrences

### Set Operations
- `Union(other)` - Combine collections (unique)
- `Intersect(other)` - Common elements
- `Diff(other)` - Elements in first but not second
- `SymmetricDiff(other)` - Elements in either but not both
- `Concat(other)` - Concatenate collections

### Predicates
- `All(fn)` / `Any(fn)` / `None(fn)` - Test conditions
- `Contains(fn)` - Check if element exists
- `Find(fn)` / `FindWithError(fn)` - Find first match
- `Partition(fn)` - Split into two groups

### Selection
- `First()` / `Last()` - Get first/last element
- `Take(n)` / `Skip(n)` - Take/skip n elements
- `TakeWhile(fn)` / `SkipWhile(fn)` - Conditional take/skip
- `Page(pageNum, pageSize)` - Pagination
- `Paginate(page, perPage)` - Pagination with metadata struct
- `ElementAt(index)` - Get element at index
- `Random()` / `Sample(n)` - Random selection

### Grouping & Indexing
- `GroupBy(fn)` - Group by key
- `KeyBy(fn)` - Create map by key
- `Chunk(size)` - Split into chunks
- `Window(size)` - Sliding windows
- `Batch(size, fn)` - Process in batches
- `IndexOf(fn)` / `LastIndexOf(fn)` - Find index

### Combining
- `Zip(c1, c2)` - Combine element-wise
- `ZipWith(c1, c2, fn)` - Combine with function

### String Operations
- `Join(separator)` - Join to string
- `JoinBy(fn, separator)` - Join with converter
- `Pluck(field)` - Extract field values

### Conversion
- `ToMap(keyFn, valueFn)` - Convert to map
- `ToSet()` - Convert to set
- `ToChannel()` - Convert to channel

### Error Handling
- `MapWithError(fn)` - Map with errors
- `FilterWithError(fn)` - Filter with errors
- `TryReduce(fn, initial)` - Reduce with errors

### Debugging & Performance
- `Tap(fn)` - Inspect without modifying
- `ParallelMap(fn, workers)` - Concurrent mapping
- `Chain(other)` - Chain operations

### Algorithms
- `Sort(less)` - Sort with comparator (alias for SortBy)
- `QuickSort(less)` - Sort using quicksort algorithm
- `MergeSort(less)` - Sort using merge sort algorithm
- `HeapSort(less)` - Sort using heap sort algorithm (O(n log n))
- `BinarySearch(target, compare)` - Binary search on sorted collection
- `BinarySearchBy(keyFn, target)` - Binary search using key extractor
- `IsSorted(less)` - Check if collection is sorted
- `Nth(n, less)` - Find nth smallest element (quickselect)
- `TopK(k, less)` - Get top k elements
- `TopKHeap(k, less)` - Get top k elements using heap (O(n log k))
- `BottomK(k, less)` - Get bottom k elements

### Utility Methods
- `CountBy(fn)` - Count occurrences by key
- `MaxN(n, fn)` / `MinN(n, fn)` - Get multiple max/min values
- `Compact()` - Remove zero values
- `DifferenceAll(others...)` - Multiple set difference
- `ScanLeft(fn, initial)` - Cumulative reduce (running totals)
- `Interleave(other)` - Merge collections alternating elements
- `Memoize()` - Cache ToSlice() result for performance

### Pagination
- `Paginate(page, perPage)` - Returns Pagination struct with metadata
- `WithFieldNames(fieldNames)` - Customize field names (chainable)
- `ToJson()` / `ToPrettyJson()` - Convert to JSON with custom fields
- `ToMap()` - Convert to map with custom fields
- `ToCustomMap(fieldNames)` - Convert with custom field names

See `collection.go` for detailed API documentation.

## Contributing

Feel free to open issues or pull requests on the GitHub repository.

## Testing

Run the test suite:
```bash
go test -v
```

All 90+ tests should pass successfully.

Run benchmarks:
```bash
go test -bench=. -benchmem
```

## Pagination

The library includes powerful pagination with customizable field names:

```go
// Basic pagination
page := collection.NewCollection[Product](products).Paginate(1, 10)
fmt.Printf("Page %d of %d\n", page.CurrentPage, page.TotalPages)

// Chainable with custom field names
json, _ := collection.NewCollection[Product](products).
    Filter(func(p Product) bool { return p.Active }).
    Paginate(1, 10).
    WithFieldNames(map[string]string{
        "data": "items",
        "current_page": "page",
        "total": "count",
    }).
    ToPrettyJson()
```

**Pagination Struct:**
```go
type Pagination[T any] struct {
    Data        []T  // Paginated items
    CurrentPage int  // Current page number
    PerPage     int  // Items per page
    Total       int  // Total items
    TotalPages  int  // Total pages
    HasNext     bool // Has next page
    HasPrev     bool // Has previous page
}
```

**Customizable Fields:**
- `data` → `items`, `results`, `products`, etc.
- `current_page` → `page`, `page_number`, etc.
- `per_page` → `limit`, `page_size`, etc.
- `total` → `count`, `total_count`, etc.
- `total_pages` → `pages`, `last_page`, etc.
- `has_next` → `next`, `has_more`, etc.
- `has_prev` → `prev`, `previous`, etc.

**Common API Formats:**
```go
// Laravel style
json, _ := c.Paginate(1, 10).
    WithFieldNames(map[string]string{"total_pages": "last_page"}).
    ToJson()

// Simple items/count
json, _ := c.Paginate(1, 10).
    WithFieldNames(map[string]string{
        "data": "items",
        "total": "count",
    }).
    ToJson()
```

## Performance

- **Lazy Evaluation**: Operations are not executed until a terminal operation (ToSlice, Count, etc.) is called
- **Parallel Processing**: Use ParallelMap for CPU-intensive transformations
- **Memory Efficient**: Chained operations don't create intermediate collections

## License

MIT License
