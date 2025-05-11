wheels=int(input("Enter number of wheels:"))

if wheels==2:
    print("It's likely a bike or scooter")
elif wheels==3:
    print("It's likely an Autorikshaw or tricycle")
elif wheels==4:
    print("It's likely a car or van")
elif wheels==5:
    print("It's likely a mini truck or delivery van")
elif wheels==8:
    print("It's likely a large truck")
elif wheels>8:
    print("It's likely a heavy duty vehicle like trailor")
else:
    print("Unknown vehicle type")