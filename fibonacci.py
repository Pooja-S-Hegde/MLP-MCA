'''check if a number belongs to the fibonaccy sequence'''
#input a number
n=int(input("Enter the number"))
c=0
a=1
b=1
if n==0 or n==1:
    print("Yes,it is fibonacci number")
else:
    while c<n:
        c=a+b
        b=a
        a=c
     #after loop termination
    if c==n:
        print("yes,it is Fibonacci number")
    else:
        print("no")
