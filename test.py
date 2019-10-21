import pandas as pd

df = pd.read_csv("test2.txt")
df = df.sort_values(by="x")

df["dx"]=df.x.shift(-1)-df.x
df["dy"]=df.y.shift(-1)-df.y
df["d"]=(df.dx*df.dx+df.dy*df.dy)**0.5

candidate1 = df.d[df.d>7].min()
print(f"Candidate 1 {candidate1}")
df["di1"] = df.d/candidate1
df["dii1"] = (df.d/candidate1).round()

candidate2 = df.d[(df.d>7) & (df.d<1.2*candidate1)].mean()
print(f"Candidate 2 {candidate2}")
df["di2"] = df.d/candidate2
df["dii2"] = (df.d/candidate2).round()


d = list(df.d)[:-1]
def make_index(d, delta, index=0):
    if len(d)==0:
        return [index]
    else:
        di = int(d[0]/delta+0.5)
        if di==0:
            return [index]+make_index(d[1:], delta, index)
        else:
            return [index] + make_index(d[1:], d[0]/di, index+di)
df["i1"] = make_index(d, candidate1)
df["i2"] = make_index(d, candidate2)

print (df)

df.to_csv("test.csv")