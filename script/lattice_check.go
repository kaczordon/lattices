package main

import (
	"fmt"
	//"bufio"
	"io/ioutil"
	//"os"
	"strings"
	"math"
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

type word struct {
	
	prev string
	next string
	prob float32
	start int
	end int
	word string
}

//func makeWord(values []string) *word {
	//r := word{values[0], values[1], values[2], values[3], values[4], values[5]}
	//return r
//}

func minProbability(prob map[string]float32) string {
	min := float32(math.MaxFloat32)
	var current string
	for key, value := range prob {
		//fmt.Println()
		if value < min {
			fmt.Print("entered: ")
			min = value
			current = key
			fmt.Println(current)
		}
	} 
	return current
}

func main() {
	//s := word{"jeden", -5379.74, "", "", 0, 0}
	adjacency := make(map[Key]float32)
	currentProbability := make(map[string]float32)
	path := make(map[string]string)
	var final []string
	dat, err := ioutil.ReadFile("C:/temp/work/src/github.com/kaczordon/hello/Data/test.lattices")
	check(err)
	
	lines := strings.Split(string(dat), "\n")
	sources := strings.Split(lines[0], " ")
	source := sources[0]
	
	for _, line := range lines {
		values := strings.Split(line, " ")
		currentProbability[values[0]] = math.MaxFloat32
		
		f, err := strconv.ParseFloat(values[2], 32)
		if err != nil {
			panic(err)
		}
		
		path[values[0]] = ""
		adjacency[values[0], values[1]] = float32(f)
	}
	for key, value := range actualProbability {
		fmt.Println(key, " ", value)
	}
	currentProbability[source] = 0
	var empty bool
	for !empty {
		minNode := minProbability(currentProbability)
		neighbors := make([]string, len(adjacency[minNode]))
		for i, value := range adjacency[minNode] {
			neighbors[i] = value;
			fmt.Println("Neighbor", value, " Prob: ", actualProbability[value])
		}
		delete(adjacency, minNode)
		delete(currentProbability, minNode)
		
		for _, value := range neighbors {
			alt := currentProbability[minNode] - actualProbability[value]
			if alt < currentProbability[value] {
				currentProbability[value] = alt
				path[value] = minNode
				fmt.Println("Min value: ", alt, " Node: ", value)
				fmt.Println("Path: ", value, " ")
			}
			//fmt.Print(alt)
		}
		if len(adjacency) == 0 {
			fmt.Println("exiting")
			empty = true
		}
	}
	target := "L479"
	for {
		if value, exists := path[target]; exists {
		final = append(final, target)
		target = value
		} else {
			break;
		}
	}
	
	fmt.Println(final)
	
	//fmt.Println("end")
	/*for key, value := range adjacency {
		fmt.Print("Key:", key, " ")
		for _, line := range value {
			fmt.Print(line, " ")
		} 
		fmt.Print("\nexit\n")
	}
	//fmt.Println("new", adjacency["L0"][0])
	//fmt.Println(string(dat))
} 


