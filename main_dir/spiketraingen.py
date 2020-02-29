import os, random

def not_gen(padding=0):
    for i in range(0,50):
        stri = bin(random.randint(0, 500))[2:]
        print(stri+',',end='',file=open('test cases/not_test_cases.txt','a'))
        print(padding*'0',end='',file=open('test cases/not_test_cases.txt','a'))

        for j in range(len(stri)):
            if stri[j] == '1':
                print('0',end='',file=open('test cases/not_test_cases.txt','a'))
            elif stri[j] == '0':
                print('1',end='',file=open('test cases/not_test_cases.txt','a'))
        print("",file=open('test cases/not_test_cases.txt','a'))

def and_gen(padding=0):
    for i in range(50):
        num1 = bin(random.randint(0, 500))[2:]
        num2 = bin(random.randint(0, 500))[2:]

        if len(num1) > len(num2):
            diff = len(num1) - len(num2)
            num2 = diff*'0' + num2
        elif len(num2) > len(num1):
            diff = len(num2) - len(num1)
            num1 = diff*'0' + num1
        
        print(num1 + ',' + num2 + ',', end='', file=open('test cases/and_test_cases.txt','a'))
        print(padding*'0', end='', file=open('test cases/and_test_cases.txt','a'))
        for j in range(len(num1)):
            print(str(int(num1[j]) & int(num2[j])),end='', file=open('test cases/and_test_cases.txt','a'))
        print("", file=open('test cases/and_test_cases.txt','a'))

def or_gen(padding=0):
    for i in range(50):
        num1 = bin(random.randint(0, 500))[2:]
        num2 = bin(random.randint(0, 500))[2:]

        if len(num1) > len(num2):
            diff = len(num1) - len(num2)
            num2 = diff*'0' + num2
        elif len(num2) > len(num1):
            diff = len(num2) - len(num1)
            num1 = diff*'0' + num1
        
        print(num1 + ',' + num2 + ',', end='', file=open('test cases/or_test_cases.txt','a'))
        print(padding*'0', end='', file=open('test cases/or_test_cases.txt','a'))
        for j in range(len(num1)):
            print(str(int(num1[j]) | int(num2[j])),end='', file=open('test cases/or_test_cases.txt','a'))
        print("", file=open('test cases/or_test_cases.txt','a'))

def add_gen(padding=0):
    for i in range(50):
        num1 = random.randint(0, 500)
        num2 = random.randint(0, 500)
        num3 = bin(num1 + num2)[2:]
        num1 = bin(num1)[2:]
        num2 = bin(num2)[2:]

        if len(num1) > len(num2):
            diff = len(num1) - len(num2)
            num2 = diff*'0' + num2
        elif len(num2) > len(num1):
            diff = len(num2) - len(num1)
            num1 = diff*'0' + num1
        print(num1[::-1] + ',' + num2[::-1] + ',' + padding*'0'+ num3[::-1], file=open('test cases/add_test_cases.txt','a'))

def sub_gen(padding=0):
    for i in range(50):
        num1 = random.randint(0, 500)
        num2 = random.randint(0, 500)
        while num1 < num2:  # no negative results please!
            num1 = random.randint(0, 500)
            num2 = random.randint(0, 500)
        num3 = bin(num1 - num2)[2:]
        num1 = bin(num1)[2:]
        num2 = bin(num2)[2:]

        if len(num1) > len(num2):
            diff = len(num1) - len(num2)
            num2 = diff*'0' + num2
        elif len(num2) > len(num1):
            diff = len(num2) - len(num1)
            num1 = diff*'0' + num1
        print(num1[::-1] + ',' + num2[::-1] + ',' + padding*'0' + num3[::-1], file=open('test cases/sub_test_cases.txt','a'))

def swap(num1, num2):
    temp = num2
    num2 = num1
    num1 = temp

def bitonic_gen(sort_nums, padding=0):
    for i in range(50):
        orig = []
        tobesorted = []
        for j in range(sort_nums):
            random_num = random.randint(0, 500)
            orig.append(random_num)
            tobesorted.append(random_num)
        print(orig)
        #generates a sorted list with the help of a bubble sort algorithm
        len_list = len(tobesorted)
        h = 1
        while h < len_list:
            k = 0
            while k < len_list - h:
                if  tobesorted[k] > tobesorted[k + 1]:
                    #swap(tobesorted[k], tobesorted[k+1])
                    temp = tobesorted[k+1]
                    tobesorted[k + 1] = tobesorted[k]
                    tobesorted[k] = temp
                k += 1
            h += 1
        print(tobesorted)
        k = 0
        while k < len(orig):
            input_string = ""
            temp = orig[k]
            while temp > 0:
                input_string += "1"
                temp -= 1
            print(input_string + ',', file=open('test_cases/bitonic_gen_test_cases.txt','a'))
            k += 1
        k = 0
        while k < len(tobesorted):
            output_string = ""
            temp = orig[k]
            while temp > 0:
                output_string += "1"
                temp -= 1
            print(padding*'0' + output_string + ',', file=open('test_cases/bitonic_gen_test_cases.txt','a'))
            k += 1

if not os.path.exists('test cases'):
    os.makedirs('test cases')

while True:
    print("What would you like to generate?")
    print("[0] Quit")
    print("[1] Not")
    print("[2] Add")
    print("[3] Sub")
    print("[4] And")
    print("[5] Or")
    print("[6] Bitonic")

    answer = input()
    if answer == "":
        answer = 0
    else:
        answer = int(answer)
    if answer == 0:
        break
    padding = -1
    while padding < 0:
        padding = input("How much padding (leading 0's) on the output would you like to add? (default:0) ")
        if padding == "":
            padding = 0
        else:
            padding = int(padding)
        if padding < 0:
            print("Enter a non negative number please")
    if answer == 1:
        not_gen(padding)
    elif answer == 2:
        add_gen(padding)
    elif answer == 3:
        sub_gen(padding)
    elif answer == 4:
        and_gen(padding)
    elif answer == 5:
        or_gen(padding)
    elif answer == 6:
        sort_nums = int(input("How many numbers would you like to sort: "))
        bitonic_gen(sort_nums, padding)

    print("Done!")

print("Exiting")
        

