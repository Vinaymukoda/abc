'''a = 'vinaymanukoda'
a = a.replace('a','j' ) # Removes all occurrences of 'a'
print(a)  # Output: 'vinymnukod'
a = 'vinaymanukoda'
a = 'vinaymanukoda'
a = a[:3] + a[4:5] + a[6:]  # Removes index 3 ('y') and index 5 ('m')
print(a)  # Output: 'vinaanukoda
b='vinaymanukoda'
print(b.count(a))
b = 'vinaymanukoda'
a = 'a'
v='v'# Define 'a' before using it
print(b.count(a))
print(b.count(v))# ✅ Output: 3 (since 'a' appears 3 times in 'vinaymanukoda')
def minimumboxes( apple):
    total_apples=sum(apple)
    apple.sort(reverse=True)
    num_boxes=0
    for box_size in apple:
        total_apples -=box_size
        num_boxes +=1
        if total_apples <= 0:
            break
    return num_boxes'''


'''def minimum_boxes(apple):
    total_apples = sum(apple)
    required_apples = (total_apples + 1) // 2  # At least half
    apple.sort(reverse=True)  # Sort in descending order

    num_boxes = 0
    collected = 0

    for box_size in apple:
        collected += box_size
        num_boxes += 1
        if collected >= required_apples:
            break  # Stop when we have at least half of the apples

    return num_boxes


# Example Usage
apple_boxes = [8,5,1]
print(minimum_boxes(apple_boxes))  # Output: Number of boxes needed
f=open('aws.txt','r')
print(f.tell())
print(f.read(2))
class emplyee:
    'common base class for all employe'
    emcount=0
    def __int__(self,name,salary):
        self.name=name
        self.salary=salary
        emplyee.emcount +=1
    def display(self):
        print('total employee %d'%emplyee.emcount)
    def displayEmloyee(self):
        print('employee.__doc__',emplyee.__doc__)
        print('employee.__name__',emplyee.__name__)
        print('employee.__module__',emplyee.__module__)
        print('employee.__base__',emplyee.__bases__)
        print('employee.__dict__',emplyee.__dict__)

emp1 = emplyee("Alice", 50000)
emp2 = emplyee("Bob", 60000)

emp1.display_count()
emp2.display_employee()

class Employee:
    """Common base class for all employees"""
    emcount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.emcount += 1  # Use class name for static variable

    def display_count(self):
        print(f'Total employees: {Employee.emcount}')

    def display_employee(self):
        print('Employee.__doc__:', Employee.__doc__)
        print('Employee.__name__:', Employee.__name__)
        print('Employee.__module__:', Employee.__module__)
        print('Employee.__bases__:', Employee.__bases__)
        print('Employee.__dict__:', Employee.__dict__)


# Example Usage:
emp1 = Employee("Alice", 50000)
emp2 = Employee("Bob", 60000)

emp1.display_count()
emp2.display_employee()


def liner_search(arr,target):
    for i in range(arr,target):
        if arr[i]==target:
            return i
arr=[1,3,5,6,7]
target=9
result=liner_search(arr,target)
if result !=-1:
    print(f'elemat fournd at index{result}')
else:
    print(f'elemat not found')
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Return the index of the target element
    return -1  # Return -1 if the target is not found

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
result = linear_search(arr, target)

if result != -1:
    print(f"Element found at index {result}")
else:
    print("Element not found")
def binary_serch(arr,target):
    left,regiht=0,len(arr)-1
    while left<=regiht:
        mid=(left+regiht)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]<target:
            left=regiht+1
        else:
            regiht=mid-1
    return -1
arr=[1,3,5,7,9,11,15]
target=7
print(binary_serch(arr,target))
 def binary_search(arr, target):
    left, right = 0, len(arr) - 1  # Correct right boundary

    while left <= right:  # Use <= instead of <
        mid = (left + right) // 2  # Correct mid calculation

        if arr[mid] == target:
            return mid  # Found target, return index
        elif arr[mid] < target:
            left = mid + 1  # Search in right half
        else:
            right = mid - 1  # Search in left half

    return -1  # Target not found

# Example usage
arr = [1, 3, 5, 7, 9, 11, 15]
target = 7
print(binary_search(arr, target))  # Output: 3
import time
import random
import boto3
import mysql.connector
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Target not found

# Function to test binary search time complexity
def test_time_complexity():
    sizes = [10**i for i in range(1, 7)]  # Test sizes: 10, 100, 1000, ..., 1,000,000

    for size in sizes:
        arr = list(range(size))  # Sorted array
        target = random.randint(0, size - 1)  # Random target in the array

        start_time = time.time()  # Start timing
        binary_search(arr, target)
        end_time = time.time()  # End timing

        elapsed_time = end_time - start_time
        print(f"Size: {size}, Time taken: {elapsed_time:.6f} seconds")

# Run the test
test_time_complexity()

def find_mix_min(arr):
    mix_valu=arr[0]
    min_valu=arr[0]
    for num in arr:
        if num>mix_valu:
            mix_valu=num
        if num<min_valu:
            min_valu=num
    return mix_valu,min_valu
arr=[3,2,1,5,9]
mixnum,minmnum=find_mix_min(arr)
print(mixnum,minmnum)
def find_max_min(arr):
    max_val = arr[0]  # Corrected variable name
    min_val = arr[0]  # Corrected assignment

    for num in arr:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num

    return max_val, min_val  # Return correct values

arr = [3, 2, 1, 5, 9]
max_num, min_num = find_max_min(arr)  # Corrected variable names
print(max_num, min_num)  # Output the correct valu
def sorted_arr(arr):
    for i in range(len(arr)-1):
        if arr[i]>arr[i+1]:
            return False
    return True
print(sorted_arr([1,2,3,4]))
print(sorted_arr([1,2,4,3]))
class Node:
    def __init__ (self,data):
        self.data=data
        self.next=None
class linkdlsit:
    def __init__(self):
        self.hend=None
    def instert_at_end(self,data):
        new_node=Node(data)
        if not self.head:
            self.head=new_node
            return
        temp=self.head
        while temp.next:
            temp.next = new_node  # Correct variable name

    def instert_at_begining(self,data):
        new_node=Node(data)
        new_node.next=self.head
        self.head=new_node
    def delete_node(self,key):
        temp = self.head

        if temp and temp.data==key:
            self.head=temp.next
            temp=None
            return
        while temp and temp .data !=key:
            prev=temp
            temp=temp.next
        if temp is None :
            return
        prev.next=temp.next
        temp=None
    def display(self):
        temp=self=self.head
        while temp:
            print(temp.data,end='->')
            temp=temp.next
        print('None')
ll = linkdlsit()
ll.instert_at_end(1)
ll.instert_at_end(2)
ll.instert_at_end(3)
ll.instert_at_begining(0)
ll.display()
ll.delete_node(2)
ll.display(20)


class Node:
    """A node in a singly linked list."""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """A simple singly linked list."""

    def __init__(self):
        self.head = None

    def insert_at_end(self, data):
        """Insert a node at the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node

    def insert_at_beginning(self, data):
        """Insert a node at the beginning of the list."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete_node(self, key):
        """Delete a node by its value."""
        temp = self.head

        # If the head node itself holds the key
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return

        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next

        if temp is None:
            return  # Node not found

        prev.next = temp.next
        temp = None

    def display(self):
        """Print the linked list."""
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")


# Example Usage
ll = LinkedList()
ll.insert_at_end(1)
ll.insert_at_end(2)
ll.insert_at_end(3)
ll.insert_at_beginning(0)

ll.display()  # Output: 0 -> 1 -> 2 -> 3 -> None

ll.delete_node(2)

ll.display()  # Output: 0 -> 1 -> 3 -> Non

text = "vinay"
num=int(text)
print(num)
from collections import Counter
def find_duplications(n,arr):
    freq=Counter(arr)
    duplicates=sorted([num for num, count in freq.items() if count>1])
    return duplicates if duplicates else [-1]
input1=6
input2=[4,4,7,8,8,9]
print(find_duplications(input1,input2))

from collections import Counter

def find_duplicates(n, arr):
    freq = Counter(arr)
    duplicates = sorted([num for num, count in freq.items() if count > 1])
    return duplicates if duplicates else [-1]

# Example usage:
input1 = 6
input2 = [4, 4, 7, 8, 8, 9]
print(find_duplicates(input1, input2))  # Output: [4, 8]

from collections import Counter
def find_dulications(n,arr):
    freq=Counter(arr)
    duplications=sorted([num for num,count in freq.items()if count>1])
    return duplications if duplications else [-1]
input1=6
input2=[4,4,7,8,8,9]
print(find_dulications(input1,input2))
from collections import Counter
def find_duplicatons(n,arr):
    freq=Counter(arr)
    duplicatons=sorted([num for num,count in freq.items()if count>1])
    return duplicatons if duplicatons else [-1]
input1=6
input2=[4,4,7,8,8,9]
print(find_duplicatons(input1,input2))




from collections import  Counter
def find_duplication(n,arr):
    freq=Counter(arr)
    duplication=sorted([num for num,count in freq.items()if count>1])
    return duplication if duplication else [-1]
input1=6
input2=[4,4,7,8,8,9]
print(find_duplication(input1,input2))
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
class Linkedlist:
    def __int__(self):
        self.head=None
    def insert_at_end(self,data):
        new_node=Node(data)
        if not  self.head:
            self.head=new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):  # Utility method to display the list
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Example usage:
ll = Linkedlist()
ll.insert_at_end(10)
ll.insert_at_end(20)
ll.insert_at_end(30)

ll.print_list()  # Output: 10 -> 20 -> 30 -> None
'''
'''def bainar_saerch(arr,target):
    left,rghit=0,len(arr)-1
    while left < rghit:
        mid=(left+rghit)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]<target:
            left=mid+1
        else:
            rghit=mid-1
    return -1
arr=[8,4,6,7,9]
target=6
print(bainar_saerch(arr,target))'''

'''def binary_search(target, arr):
    arr.sort()  # Ensure the array is sorted
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [8, 4, 6, 7, 9]
target = 6
print(binary_search(target, arr))  # Output: Index of 6 in the sorted array
8

def convert_variable_name(name):
    if "_" in name:
        # Convert from C++ to Java
        parts = name.split("_")
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])
    else:
        # Convert from Java to C++
        result = ''
        for char in name:
            if char.isupper():
                result += '_' + char.lower()
            else:
                result += char
        return result

# Example usage
input1 = "modify_variableName"
print(convert_variable_name(input1))  # Output: modifyVariableName

'''
'''def convert_variable_name(name):
    if '_'in name:
        paets=name.split('_')
        return paets[0]+''.join(word.capitalize()for word in paets[1:])
    else:
        result =''
        for char in name:
            if char.isuppr():

                result +='_'+char.lower()
            else:
                result +=char
        return result
input1='modify_variableName'
print(convert_variable_name(input1))
def convert_variable_name(name):
    if '_'in name:
        paets=name.split('_')
        return paets[0]+''.join(word.capitalize()for word in paets[1:])
    else:
        result=''
        for char in name:
            if char.isuppr():
                result +='_'+char.lower()
            else:
                result +=char
        return result
input1='modify_variableName'
print(convert_variable_name(input1))
def convert(name):
    if "_"in name:
        paets=name.split('_')
        return paets[0]+''.join(word.capitalize()for word in paets[2:1])
    else:
        result=''
        for char in name:
            if char.isupper():
                result +='_'+char.lower()
            else:
                result +=char
            return result
input1='BheeshmaKumar'
print(convert(input1))
import storing


def is_pangrm(s):
    s = s.lower()
    return set(storing.ascii_lowercase.issubset(set(s)))


text1 = 'is up brown not text'
text2 = 'hello word'
print(is_pangrm(text1, text2))
def is_pangram(text):
    letters = [0] * 26  # 26 letters in the alphabet
    for char in text.lower():
        if 'a' <= char <= 'z':
            index = ord(char) - ord('a')
            letters[index] = 1
    return sum(letters) == 26
print(is_pangram("The quick brown fox jumps over the lazy dog"))  # True
print(is_pangram("Python is fun"))  # False
class _Constants:
    pi=3.14159
    GROVITY=9.8
    def __setattr__(self, name, value):
        raise ('type of errors only read only ')
constnt=_Constants

class solustion():
    def towsum(self,num:list[int],target:int)->list[int]:
        disnary={}
        for i in range (len(num)):
            if num[i] in disnary:
                return [disnary[num[i]],i]
            else:
                disnary[target - num[i]] = i

num=[2,7,11,15]
target=9
sol=solustion()
print(sol.towsum(num,target))


class solustion():
    def towsum(self, num: list[int], target: int) -> list[int]:
        disnary = {}
        for i in range(len(num)):
            if num[i] in disnary:
                return [disnary[num[i]], i]
            else:
                disnary[target - num[i]] = i

num = [2, 7, 11, 15]
target = 9
sol = solustion()
print(sol.towsum(num, target))
from typing import List
class solustion():
    def threesum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result=[]
        j,k=0,0
        for i in range (len(nums)):
            if i>0 and nums[i-1]==nums[i]:
                continue
            j,k=i+1,len(nums)-1
            while j<k:
                if nums[i]+nums[k]==0:
                    result.append([nums[i],nums[j],nums[k]])
                    j+=1
                    k-=1
                    while j<k and nums[j]==nums[j-1]:
                        j+=1
                    while j<k and nums[k]==nums[k+1]:
                        k-=1
                elif nums[i]+nums[j]+nums[k]<0:
                    k-=1
                elif nums[i]+nums[j]+nums[k]>0:
                    j+=1
        return result
sol=solustion()
nums= [-1, 0, 1, 2, -1, -4]
print(sol.threesum(nums)

from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                total = nums[i] + nums[j] + nums[k]
                if total == 0:
                    result.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif total < 0:
                    j += 1
                else:
                    k -= 1
        return result

# Test it
sol = Solution()
nums = [-1, 0, 1, 2, -1, -4]
print(sol.threeSum(nums))  # Expected: [[-1, -1, 2], [-1, 0, 1]]




from typing import List

class Solution:
    def threeSumTarget(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                total = nums[i] + nums[j] + nums[k]
                if total == target:
                    result.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif total < target:
                    j += 1
                else:
                    k -= 1
        return result

# Example usage
sol = Solution()
nums = [1, 2, -1, 0, -2, 3]
target = 4
print(sol.threeSumTarget(nums, target))  # Expected output: [[-1, 2, 3], [0, 1, 3]]'''



from typing import List
class solution:
    def treesumtarget(self,nums:[int],target:int)->List[list[int]]:
        nums.sort()
        result=[]
        for i in range(len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            j,k=i+1,len(nums)-1
            while j<k:
                total=nums[i]+nums[j]+nums[k]
                if total==target:
                    result.append([nums[i],nums[j],nums[k]])
                    j+=1
                    k-=1
                    while j<k and nums[j]==nums[j-1]:
                        j+=1
                    while j<k and nums[k]==nums[k+1]:
                        k-=1
                elif total < target:
                    j += 1
                else:
                    k -= 1
        return result
sol=solution()
nums=[1, 2, -1, 0, -2, 3]
target=4
print(sol.treesumtarget(nums,target))






















