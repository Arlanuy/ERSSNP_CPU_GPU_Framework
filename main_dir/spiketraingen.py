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

    print("Done!")

print("Exiting")
        

