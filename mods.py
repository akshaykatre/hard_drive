def howmanynulls(a, df=None):
    if type(a) != list: 
        print("% of null in ", a.name, " is ", a.isnull().sum()/a.shape[0] * 100) 
    elif type(a) == list:
        for a1 in a:
            howmanynulls(df[a1])


def mmm(a, df=None):
    if type(a) != list:
        print("For ", a.name, "mean: ", str(a.mean()), " median: ", str(a.median()))
    else:
        for a1 in a:
            mmm(df[a1])

