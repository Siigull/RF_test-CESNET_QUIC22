with open("wadsd.txt", "r") as f:
    counter = 0
    sum = 0
    for line in f:
        # print(float(line))
        sum += float(line)
        if counter == 100:
            counter = 0
            print(sum / 100)
            sum = 0


        counter += 1