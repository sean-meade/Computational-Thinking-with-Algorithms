from random import randint
from tabulate import tabulate
import math
import time

"""
Insertion Sort Algo
Ref: 1.	interactivepython. 2019. The Insertion Sort. [ONLINE] Available at: http://interactivepython.org/courselib/static/pythonds/SortSearch/TheInsertionSort.html. [Accessed 1 May 2019].
"""
def insertionSort(aList):
    # For every index in list except first
    for index in range(1, len(aList)):

        # Create a variable for the current element in loop
        currentValue = aList[index]
        # Let index equla to position to allow manipulation of index 
        position = index
        
        # While position is above 0 AND the 
        # value at current element is less then the 
        # one before it.
        while position > 0 and aList[position - 1] > currentValue:
            # Then the current element is moved to the 
            # one before it.
            aList[position] = aList[position - 1]
            # Repeat for next value to its left.
            position -= 1

        # Then set this value at the correct index.
        aList[position] = currentValue

    # Return aList
    return aList

"""
Quick Sort
Ref: 2.	interactivepython. 2019. The Quick Sort. [ONLINE] Available at: http://interactivepython.org/courselib/static/pythonds/SortSearch/TheQuickSort.html. [Accessed 3 May 2019].
"""

# Create quick sort funtion that takes an unsorted array as 
# its only parameter
def quickSort(array):
    # Calls quickSortHelper function which takes the array,
    # first as first index (0) and and last as last index
    # (len(array) - 1) of the array 
    quickSortHelper(array, 0, len(array) - 1)

# Create quick sort helper function that takes three parameters
# the array, the first index and the last 
def quickSortHelper(array, first, last):
    # If the first index is less then the last
    if first < last:

        # Create a split point with same parameters (this is the
        # position the pivot point acually belongs in).
        splitpoint = partition(array, first, last)

        # Call quickSortHelper again to sort the elements before
        # (between first and splitpoint-1) and after (between 
        # splitpoint+1 and last) the splitpoint.
        # Before:
        quickSortHelper(array, first, splitpoint - 1)
        # After:
        quickSortHelper(array, splitpoint + 1, last)

# Create partition function with paramenters array, first and last
def partition(array, first, last):
    # Decide on pivot value and create. In this instance we are
    # using the first element
    pivotValue = array[first]

    # Create variable to contain the left marker
    leftMarker = first + 1
    # Create variable to contain the right marker
    rightMarker = last

    # Set a varible to false to indicate that the sort is not done
    done = False

    # While the sort is not done
    while not done:

        # and while the left marker is less then or equal to the right
        # AND the element at the left marker index is less then or equal
        # to the pivot value
        while leftMarker <= rightMarker and array[leftMarker] <= pivotValue:
            # Move the left marker along to the next index
            leftMarker = leftMarker + 1

        # Otherwise while the right marker is greater then or equal to the
        # left and the element at index of the right marker is greater then
        # or equal to the pivot value
        while rightMarker >= leftMarker and array[rightMarker] >= pivotValue:
            # Move the right marker index back to the previous index in array
            rightMarker = rightMarker - 1
        
        # If the right marker is less then the left marker (* check if works
        # when switched to left marker is greater then righ marker*)
        if rightMarker < leftMarker:
            # The sort is finished
            done = True
        # Otherwise
        else:
            # Create a temporary value equal to the element at the current left
            # marker index
            temp1 = array[leftMarker]
            # Then swap the element at left marker for the one at the right marker
            array[leftMarker] = array[rightMarker]
            # Then set the temp value to the element at the right marker index
            array[rightMarker] = temp1
        
    # Then set another temp value equal to the first value
    temp2 = array[first]
    # Then set the value at the first index equal to the value at the right marker
    array[first] = array[rightMarker]
    # Then set the value at right marker index to the temp2 value
    array[rightMarker] = temp2

    # Return the right marker (i.e. splitpoint)
    return rightMarker

"""
Bucket Sort
Ref: 3.	geeksforgeeks. 2019. Bucket Sort. [ONLINE] Available at: https://www.geeksforgeeks.org/bucket-sort-2/. [Accessed 6 May 2019].
"""

# Create Bucket sort function that takes in the array
def bucketSort(array):
    # Set bucket size
    bucketSize = 10

    # If the length of the array is zero
    if len(array) == 0:
        # Then return the array
        return array
    
    # Create variables with max and min values within array
    minVal = min(array)
    maxVal = max(array)

    # Set up buckets
    # Work out how many buckets to use
    bucketCount = math.floor((maxVal - minVal) / bucketSize) + 1
    # Create empty array to hold each bucket
    buckets = []
    # For every bucket in bucket count
    for bucket in range(0, bucketCount):
        # Add empty bucket into buckets array
        buckets.append([])

    # Place elements into their respective buckets by giving it the 
    # index (using floor) of its value minus the min value and deviding
    # it by the bucket size
    for i in range(0, len(array)):
        buckets[math.floor((array[i] - minVal) / bucketSize)].append(array[i])

    # Create an array to place sorted array in
    array = []
    # For every bucket in buckets array
    for bIndex in range(0, len(buckets)):
        # Sort the contents of that bucket
        insertionSort(buckets[bIndex])
        # For every element in each bucket
        for elem in range(0, len(buckets[bIndex])):
            # Add it to the new empty array
            array.append(buckets[bIndex][elem])

    # Return the sorted array
    return array

"""
Radix Sort
Ref: 4.	geeksforgeeks. 2019. Radix Sort. [ONLINE] Available at: https://www.geeksforgeeks.org/radix-sort/. [Accessed 8 May 2019].
"""

# Radix Sort Algorithm

# Create function for Radix Sort that takes an unsorted array
# as its only parameter
def radixSort(array):

    # Find the value in the array to get number of digits
    maxVal = max(array)

    # Set the integer range
    intRange = 10

    # Create variable for exponential (exp) function (starting at one)
    # Where exp is 10^i where i is the current digit number
    exp = 1
    # While the max value divided by exp is greater than zero
    while maxVal/exp > 0:
        # Do the counting on the array for the current digit
        countingSort(array, exp, intRange)
        # Go to the next digit
        exp *= 10
    
    # Return sorted array
    return array

# The counting sort for the Radix algorithm
def countingSort(array, exp, intRange):

    # Create a variable to hold number of elements in the array
    noElem = len(array)

    # Create an array of zero values matching the size of intRange
    counter = [0 for index in range(0, intRange)]
    # Create an array of zero values matching the number of elements in array
    sortedList = [0 for i in range(0, noElem)]

    # Sort each element into its respective counters
    for i in range(0, noElem):
            # This counts the integers going into each counter. Each element is divided 
            # by the power of ten depending on what digit its on. Then that is modularized
            # by the integer range (which is ten because the number system is base 10).
            counter[math.floor((array[i] / exp) % intRange)] += 1

    # Sort through each index between 1 and 10
    for index in range(1, intRange):
        # The previous counter gets added to the previous one. This fills out each counter 
        # to ten
        counter[index] += counter[index - 1]
    
    # Create a varible thats one less then the number of elements in the array
    i = noElem - 1

    # While i this number is greater than or equal to zero
    while i >= 0:
        # Let the element at index i of array be the value in the sorted array at index
        # in the counter.
        sortedList[counter[math.floor((array[i] / exp) % intRange)] - 1] = array[i]
        # Take one from that index
        counter[math.floor((array[i] / exp) % intRange)] -= 1
        # and take one from variable i
        i -= 1

    # For every element in sorted list let it equal to that index in the array
    for j in range(0, noElem):
        array[j] = sortedList[j]

"""
Timsort
Refs:5.	hackernoon. 2019. Timsort — the fastest sorting algorithm you’ve never heard of. [ONLINE] Available at: https://hackernoon.com/timsort-the-fastest-sorting-algorithm-youve-never-heard-of-36b28417f399. [Accessed 10 May 2019].
6.	medium. 2019. This is the fastest sorting algorithm ever. [ONLINE] Available at: https://medium.com/@george.seif94/this-is-the-fastest-sorting-algorithm-ever-b5cee86b559c. [Accessed 11 May 2019].
7.	medium. 2019. This is the fastest sorting algorithm ever. [ONLINE] Available at: https://medium.com/@george.seif94/this-is-the-fastest-sorting-algorithm-ever-b5cee86b559c. [Accessed 11 May 2019].
8.	dev. 2019. Timsort: The Fastest sorting algorithm for real-world problems.. [ONLINE] Available at: https://dev.to/s_awdesh/timsort-fastest-sorting-algorithm-for-real-world-problems--2jhd. [Accessed 11 May 2019].
"""
# Combination of Merge and Insertion Sort
# Size of merged arrays

# Binary search:
# This finds the position of a target value within a sorted array

def binarySearch(array, targetV, start, end):
    # If array is halfed all the way to one value
    if start == end:
        # If the first element is greater then the target value
        if array[start] > targetV:
            # Return start index
            return start
        # Otherwise
        else:
            # Return the start index + 1
            return start + 1
        
        # Get middle index
        mid = (start + end)/2
        # If the middle element is less then target value
        if array[mid] < targetV:
            # Call binary search on second half of array
            return binarySearch(array, targetV, mid + 1, end)
        # Else if the middle element is greater then target value
        elif array[mid] > targetV:
            # # Call binary search on first half of array
            return binarySearch(array, targetV, start, mid -1)
        # Otherwise
        else:
            # It is at this location
            return mid

# Merge sort 
# Takes two sorted lists and returns a single sorted list by
# comparing the elements on at a time.
def merge(left, right):
    # If left is empty return right array
    if not left:
        return right
    # If right is empty return left array
    if not right:
        return left
    # If the first element in the left is less then the first 
    # in the right array
    if left[0] < right[0]:
        # return lefts first element plus the merge of left 
        # and right where the left starts at the second element
        return [left[0]] + merge(left[1:], right)
    # Otherwise retrurn rights first element plus the merge
    # of left and right where the right starts at the second
    # element
    return [right[0]] + merge(left, right[1:])

# Timesort
def timsort(array):
    # Create two empty array to contain each run and the
    # output of sorted runs
    runs = []
    sortedRuns = []
    # Set a variable for the length of the array
    l = len(array)
    # Set an array containing the first element of the array
    newRun = [array[0]]
    # run through array from second element to the last
    for i in range(1, l):
        # if the index is for the second last element
        if i == l-1:
            # add that element to current run
            newRun.append(array[i])
            # and append that run to the runs array
            runs.append(newRun)
            # break the loop
            break
        # if the current element is less then the one before
        # it
        if array[i] < array[i - 1]:
            # and if the current run is empty
            if not newRun:
                # Add the element before it as its own array
                # to runs
                runs.append([array[i - 1]])
                # And add the current element to the current run
                newRun.append(array[i])
            # Otherwise
            else:
                # Add the current run to the list of runs
                runs.append(newRun)
                # And create an empty new run
                newRun = []
        # Otherwise
        else:
            # Add the current element to the current run
            newRun.append(array[i])
    # For every array in runs
    for each in runs:
        # Run it through the insertion algorithm and then
        # add it to the sortedRuns array
        sortedRuns.append(insertionSort(each))
    # Create an empty array to hold the sorted array
    sortedArray = []
    # For each sorted array in sortedRuns
    for run in sortedRuns:
        # Merge the sortedArray with the current sorted run
        # and each time let that equal the new sorted array
        sortedArray = merge(sortedArray, run)
    
    # Print finished sorted array and time elapsed.
    return sortedArray

# randomArray produces an array of random sizr n
# ref: 13.	P.Mannion (2019). Sorting Algorithms Lecture 11. Benchmarking Algorithms. Galway- Mayo Institute of Technology.
def randomArray(n):
    array = []
    for i in range(0, n, 1):
        array.append(randint(0,100))
    return array

# allSetsOfArrays takes in a sorting algorithm as its
# parameter, runs random arrays of various sizes through it
#  ten times each and finds the average of each.
def allSetsOfArrays(sortingAlgorithm):
    # Various input sizes n
    difSizeA = [100, 250, 500, 750, 1000, 1250, 3750,
                5000, 6250, 7500, 8750, 10000]
    # Array to hold avg times for each input size 
    timesForAlgorithm = []
    for i in range(0, len(difSizeA)):
        times = []
        for j in range(1, 10):
            startTime = time.time()
            sortingAlgorithm(randomArray(difSizeA[i]))
            endTime = time.time()
            totTime = ((endTime - startTime) * 1000)
            times.append(totTime)
        timer = round((sum(times) / len(times)), 3)

        timesForAlgorithm.append(timer)
    
    return timesForAlgorithm

# Because of recursion limitation I had to create timsort
# its own function to feed in input sizes and avgerage out 
# times
def ArraysForTimsort(sortingAlgorithm):
    difSizeA = [100, 250, 500, 750, 1000, 1250]
    timesForAlgorithm = []
    for i in range(0, len(difSizeA)):
        times = []
        for j in range(1, 10):
            startTime = time.time()
            sortingAlgorithm(randomArray(difSizeA[i]))
            endTime = time.time()
            totTime = ((endTime - startTime) * 1000)
            times.append(totTime)
        timed = round((sum(times) / len(times)), 3)

        timesForAlgorithm.append(timed)
    
    return timesForAlgorithm

def intToStr(sortedArray):
    sortedArray = [str(i) for i in sortedArray]
    return sortedArray

header = ["Size", "100", "250", "500", "750", "1000", "1250", "3750", "5000", "6250", "7500", "8750", "10000"]
insertionTimes = ["Insertion Sort:"] + allSetsOfArrays(insertionSort)
quickTimes = ["Quick Sort:   "] + allSetsOfArrays(quickSort)
bucketTimes = ["Bucket Sort:   "] + allSetsOfArrays(bucketSort)
radixTimes = ["Radix Sort:    "] + allSetsOfArrays(radixSort)
timsortTimes = ["Timsort:       "] + ArraysForTimsort(timsort)

finalTimesInStrings = intToStr(insertionTimes), intToStr(quickTimes), intToStr(bucketTimes), intToStr(radixTimes), intToStr(timsortTimes)

print('\n')
print(tabulate(finalTimesInStrings, header, tablefmt="orgtbl"))
print('\n')


