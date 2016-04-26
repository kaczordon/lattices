package main

import (
	"fmt"
	"io/ioutil"
	"strings"
	"sort"
	"strconv"
)

type Key struct {
	currentName, toNode string
}

func check(e error) {
    if e != nil {
        panic(e)
    }
}

func NewQueue(size int) *Queue {
	return &Queue{
		nodes: make([]string, size),
		size:  size,
	}
}

// Queue is a basic queue based on a circular list that resizes as needed.
type Queue struct {
	nodes []string
	size  int
	head  int
	tail  int
	count int
}

// Push adds a node to the queue.
func (q *Queue) Push(n string) {
	if q.head == q.tail && q.count > 0 {
		nodes := make([]string, len(q.nodes)+q.size)
		copy(nodes, q.nodes[q.head:])
		copy(nodes[len(q.nodes)-q.head:], q.nodes[:q.head])
		q.head = 0
		q.tail = len(q.nodes)
		q.nodes = nodes
	}
	q.nodes[q.tail] = n
	q.tail = (q.tail + 1) % len(q.nodes)
	q.count++
}

// Pop removes and returns a node from the queue in first to last order.
func (q *Queue) Pop() string {
	if q.count == 0 {
		return ""
	}
	node := q.nodes[q.head]
	q.head = (q.head + 1) % len(q.nodes)
	q.count--
	return node
}

type pathProb struct {
	currentProb float32
	path []string
}

type node struct {
	name string
	probability float32
	timeS int
	timeE int
	word string
	path currentPaths
}
//used for sorting probability paths
type currentPaths []pathProb

func (nodes currentPaths) Len() int {
	return len(nodes)
}

func (slice currentPaths) Less(i, j int) bool {
	return slice[i].currentProb < slice[j].currentProb
}

func (slice currentPaths) Swap(i, j int) {
    slice[i], slice[j] = slice[j], slice[i]
}

//used for sorting nodes by time
type adjacencyList []node

func (nodes adjacencyList) Len() int {
	return len(nodes)
}

func (slice adjacencyList) Less(i, j int) bool {
	return slice[i].timeS < slice[j].timeS
}

func (slice adjacencyList) Swap(i, j int) {
    slice[i], slice[j] = slice[j], slice[i]
}

func main() {
	adjacency := make(map[string]adjacencyList)
	path := make(map[string]string)
	visited := make(map[string]bool)
	var nodeCount int
	var finalNode string
	dat, err := ioutil.ReadFile("C:/temp/work/src/github.com/kaczordon/hello/Data/test.lattices")
	check(err)
	
	lines := strings.Split(string(dat), "\n")//get source node and setup first path struct
	firstLine := strings.Split(lines[0], " ")
	source := firstLine[0]
	
	for _, line := range lines { //initializes all visited to false, we haven't visited any nodes yet
		values := strings.Split(line, " ")
		if _, exists := visited[values[0]]; !exists {
			visited[values[0]] = false
			nodeCount++
		}
	}
	
	q := NewQueue(nodeCount)
	for index, line := range lines { //adds all nodes to the adjacency list, which has node key strings in a map with slices of structs representing edges
		values := strings.Split(line, " ")
		//fmt.Println(values[0], values[1], values[2], values[3], values[4], values[5])
		if _, exist := adjacency[values[0]]; !exist {
			f, err := strconv.ParseFloat(values[2], 32)
			if err != nil {
				panic(err)
			}
			ts, err := strconv.Atoi(values[3])
			if err != nil {
				panic(err)
			}
			te, err := strconv.Atoi(values[4])
			if err != nil {
				panic(err)
			}
			path[values[0]] = ""
			newNode := node{values[0], 0, ts, te, values[5], make([]pathProb, 0)}
			if values[0] == source {//checks for first line
				firstPath := make([]string, 0)
				firstPath = append(firstPath, source)
				firstStruct := pathProb{0, firstPath}
				newNode.path = append(newNode.path, firstStruct)
			}
			adjacency[values[0]] = append(adjacency[values[0]], newNode)
			nextNode := node{values[1], float32(-f), ts, te, values[5], make([]pathProb, 0)}
			adjacency[values[0]] = append(adjacency[values[0]], nextNode)
			if index == len(lines) - 1 { //checks for last line
				lastNode := node{values[1], 0, ts, te, values[5], make([]pathProb, 0)}
				adjacency[values[1]] = append(adjacency[values[1]], lastNode)
				finalNode = values[1]
			}
		} else {
			f, err := strconv.ParseFloat(values[2], 32)
			if err != nil {
				panic(err)
			}
			ts, err := strconv.Atoi(values[3])
			if err != nil {
				panic(err)
			}
			te, err := strconv.Atoi(values[4])
			if err != nil {
				panic(err)
			}
			newNode := node{values[0], 0, ts, te, values[5], make([]pathProb, 0)}
			adjacency[values[0]] = append(adjacency[values[0]], newNode)
			nextNode := node{values[1], float32(-f), ts, te, values[5], make([]pathProb, 0)}
			adjacency[values[0]] = append(adjacency[values[0]], nextNode)
			//fmt.Println(values[0])
		}
	}
	
	visited[source] = true
	q.Push(source)
	for q.count > 0 {
		node := q.Pop()
		sort.Sort(adjacency[node])
		for _, value := range adjacency[node] {
			if node == value.name {
				if len(value.path) > 5 {// only take top 5
					sort.Sort(value.path)
					//length := len(value.path)
					var topPaths currentPaths
					for i := 0; i < 5; i++ {
						topPaths = append(topPaths, value.path[i])
					}
					value.path = topPaths
				}
				continue
			}
			if visited[value.name] == false {
				visited[value.name] = true
				q.Push(value.name)
			}
			for _, paths := range adjacency[node][0].path {
				var nextpath []string
				nextpath = append(nextpath, paths.path...)//so that we don't overwrite old slice with wrong path
				newPath := pathProb {value.probability + paths.currentProb, append(nextpath, value.name)}
				adjacency[value.name][0].path = append(adjacency[value.name][0].path, newPath)
			}
		}
	}
	
	if len(adjacency[finalNode][0].path) > 5 {// only take top 5
					sort.Sort(adjacency[finalNode][0].path)
					//length := len(value.path)
					var topPaths currentPaths
					for i := 0; i < 5; i++ {
						topPaths = append(topPaths, adjacency[finalNode][0].path[i])
					}
					adjacency[finalNode][0].path = topPaths
				}
	for _, value := range adjacency[finalNode][0].path {
		fmt.Println(value)
	}
} 


