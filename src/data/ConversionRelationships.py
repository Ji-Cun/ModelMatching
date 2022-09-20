import random
def conversion(id=0):
    '''
    transMatrix=[
        [0.2, 0.5, 0.1, 0.2, 0, 0],
        [0.2, 0.4, 0.2, 0.1, 0.1, 0],
        [0.3, 0.3, 0.4, 0, 0, 0],
        [0.1, 0, 0, 0.3, 0.3, 0.3],
        [0, 0.2, 0, 0.4, 0.2, 0.2],
        [0, 0, 0, 0, 0.5, 0.5],
    ]
    '''
    newID = 0
    a = random.randint(0, 9)
    if 0 == id:
        if a in [0, 1]:
            newID = 0
        elif a in [2, 3, 4, 5, 6]:
            newID = 1
        elif a == 7:
            newID = 2
        elif a in [8, 9]:
            newID = 3
    if 1 == id:
        if a in [0, 1]:
            newID = 0
        elif a in [2, 3, 4, 5]:
            newID = 1
        elif a in [6, 7]:
            newID = 2
        elif a == 8:
            newID = 3
        elif a == 9:
            newID = 4
    if 2 == id:
        if a in [0, 1, 2]:
            newID = 0
        elif a in [3, 4, 5]:
            newID = 1
        elif a in [6, 7, 8, 9]:
            newID = 2
    if 3 == id:
        if a == 0:
            newID = 0
        elif a in [1, 2, 3]:
            newID = 3
        elif a in [4, 5, 6]:
            newID = 4
        elif a in [7, 8, 9]:
            newID = 5
    if 4 == id:
        if a in [0, 1]:
            newID = 1
        elif a in [2, 3, 4, 5]:
            newID = 3
        elif a in [6, 7]:
            newID = 4
        elif a in [8, 9]:
            newID = 5
    if 5 == id:
        if a < 5:
            newID = 4
        else:
            newID = 5
    return newID
