import score 

score.init()


for x in range(50):
    data = "{\"uid\": \"" + str(x) + "\"}"
    print(score.run(data))



