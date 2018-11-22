scopeMulti = __import__('score-multiple')

scopeMulti.init()


for x in range(50):
    data = "{\"uid\": \"" + str(x) + "\"}"
    print(scopeMulti.run(data))



