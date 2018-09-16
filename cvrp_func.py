import numpy as np
from io import StringIO
import re
import binpacking as bp
import itertools
from IPython.display import clear_output
from matplotlib import pyplot as plt
import time
import math


#računa rastojanje
def dist(K1,K2):
    return np.round(np.sqrt((K1[0]-K2[0])**2 + (K1[1]-K2[1])**2))


#parsiranje instance
def load(file):
    with open("instance/"+file+".vrp","r") as inst:
        for line in inst:
            line=line.strip()
            if line.startswith("NAME"):
                results = re.search("NAME : (.+k(\d+))", line)
                name = results.group(1)
                vehicle_count = int(results.group(2))

            if line.startswith("CAPACITY"):
                capacity = int(re.search("CAPACITY : (\d+)", line).group(1))
                break

        data_str = inst.read()
        results = re.search("NODE_COORD_SECTION(.+)DEMAND_SECTION(.+)DEPOT_SECTION", data_str, re.S)
        coords = np.loadtxt(StringIO(results.group(1)), usecols=(1,2))
        demands = np.loadtxt(StringIO(results.group(2)), usecols=(1,))
        M=np.column_stack([coords,demands])
        r={'name': name, 'no_of_trucks':vehicle_count, 'capacity': capacity, 'coords_demands':M}
    return r


#parsiranje fajla sa vrednostima optimalnog rešenja
def get_opt_results(file,r):
    d=r['coords_demands'].shape[0]+r['no_of_trucks']-1
    v = -1*np.ones(d, dtype=np.int_)
    v[0]=0
    with open("instance/"+file+".opt","r") as inst:
        for line in inst:

            line=line.strip()

            if line.startswith("Route"):
                k = np.where(v == 0)[0][-1]
                rez = re.search("Route #\d: (.+)", line).group(1)
                rezV=np.fromstring(rez, dtype=np.int_, sep=' ')
                l=len(rezV)
                if l==1:
                    v[k+1]=int(rezV[0])
                    if (k+2) < d:
                        v[k+2] = 0

                else:
                    for i in range(0,l):
                        v[k+i+1]=int(rezV[i])
                    if (k + l+1) < d:
                        v[k +l+1] = 0

            if line.startswith("cost"):
                f = int(re.search("cost (\d+)", line).group(1))

    return f,v



#formira matricu rastojanja
def dist_matrix(M):
    d=np.shape(M)[0]
    C=np.zeros((d,d))
    for i in range(0,d-1):
        for j in range(i+1,d):
            C[i,j]=dist(M[i,[0,1]],M[j,[0,1]])
            C[j,i]=C[i,j]
    return C


#dopustivo rešenje prebacuje u oblik pogodan za računanje funkcije cilja
def fix_sol(s):
    n = s.shape[0]
    k = np.where(s == 0)[0][0]
    fs = s[(np.arange(k, n+k)) % n]
    return fs

#računanje funkcije cilja
def objective_function(s,C):
    row=s
    n=s.shape[0]
    clmn=s[np.arange(1, n+1) % n]
    return np.sum(C[row,clmn])


#provera ispunjenosti kapaciteta
def capacity_condition(sol, r):
    fs = fix_sol(sol)
    ind1 = np.where(fs == 0)[0]
    d = np.size(ind1)
    cpc = r['capacity']
    dmd = r['coords_demands'][:, 2]

    for i in range(d - 1):
        j = ind1[i]
        k = ind1[i + 1]
        s = np.sum(dmd[fs[j:k]])
        if s > cpc:
            return 0

    s = np.sum(dmd[fs[k:]])
    if s > cpc:
        return 0
    else:
        return 1

#generisanje inicijalng rešenja
def initial_solution_bp(r):
    dmd = r['coords_demands'][:,2]
    dmd1=dmd[1:]
    indexes = np.arange(1, dmd1.size+1)
    bpsol = bp.to_constant_volume({ key:value for key,value in zip(indexes , dmd1)}, r['capacity'])
    ll=[[0]+list(d.keys()) for d in bpsol]
    sol=np.array(list(itertools.chain(*ll)))
    return sol



#generisanje rešenja iz simpgen okoline rešenja s
def simpgen(s):
    d = np.size(s)
    s1 = s.copy()
    rv = np.sort(np.random.randint(0, d, size=4))
    s1[rv[0]:rv[3] + 1] = s1[rv[0]:rv[3] + 1][::-1]
    s1[rv[1]:rv[2] + 1] = s1[rv[1]:rv[2] + 1][::-1]
    return s1


#  Lokalna pretraga (eng. Local Search - LS)
#  argumenti:  r - podaci o instanci,
#              maxIter - maksimalan broj iteracija bez poboljšanja,
#              C - matrica rastojanja,
#              f_opt - opimalna vrednost,
#              (korišćenja za break kako bismo manje čekali pri pokretanju aplikacije)

def LS(r, maxIter, C,f_opt):
    t = time.time()
    s = initial_solution_bp(r)
    f = objective_function(s, C)
    Iter = 0
    br = 0

    while br < maxIter:
        ns = simpgen(s)
        while not (capacity_condition(ns, r)):
            ns = simpgen(s)
        nf = objective_function(ns, C)
        clear_output(wait=True)
        print('value:', nf, 'no_improvement:', br)
        Iter = Iter + 1
        if nf < f:
            br = 0
            s = ns
            f = nf
            if abs(f - f_opt) < 0.5:
                el = time.time() - t
                print('best value:',f)
                return s, int(f), el
        else:
            br = br + 1
    el = time.time() - t
    print('best value:', f)
    return s, int(f), el



#Simulirano kaljenje (eng. Simulated Annealing - SA)
#argumenti: T0 - početna temperatura,
#           Tf - krajnja temperatura,
#           alpha - parametra hlađenja,
#           K - parametar kojim se utiče
#               na verovatnoću izbora lošijeg rešenja,
#           C - matrica rastojanja,
#           r - podaci o instanci,
#           f_opt - opimalna vrednost

def SA(T0, Tf, alpha, K, C, r,f_opt):
    t = time.time()
    x = initial_solution_bp(r)
    x_best = x.copy()
    f = objective_function(x, C)
    f_best = f.copy()
    t = T0
    Iter = 10
    beta = 1.05
    while t > Tf:

        k = 1
        while k <= Iter:
            nx = simpgen(x)
            while not (capacity_condition(nx, r)):
                nx = simpgen(x)

            nf = objective_function(nx, C)
            clear_output(wait=True)
            print('temp:', t,'value:', f)
            delta = nf - f
            if delta <= 0:
                x = nx
                f = nf
            else:
                rnd = np.random.rand(1)
                if rnd < np.exp(-delta / (K * t)):
                    x = nx
                    f = nf

            if nf < f_best:
                x_best = nx
                f_best = nf
                if abs(f_best - f_opt) < 0.5:
                    el = time.time() - t
                    print('best value:', f_best)
                    return x_best, int(f_best), el
            k = k + 1

        Iter = min(250, Iter * beta)
        t = alpha * t
    el = time.time() - t
    print('best value:', f_best)
    return x_best, int(f_best), el


#prebacuje element sa pozicije i na poziciju j
def insertion(s):
    d=np.size(s)
    s1=s.copy()
    rv=np.sort(np.random.randint(0, d, size=2))
    s1=np.insert(s1,rv[0],s1[rv[1]])
    s1=np.delete(s1,rv[1]+1)
    return s1

#zamenjuje pozicije dva elementa rešenja s
def swap(s):
    d=np.size(s)
    s1=s.copy()
    rv=np.sort(np.random.randint(0, d, size=2))
    #print(rv)
    t=s1[rv[0]]
    s1[rv[0]]=s1[rv[1]]
    s1[rv[1]]=t
    return s1

#radi reverse nad jednim delom rešenja (nekoliko uzastopnih elemenata vektora s)
def two_opt(s):
    d=np.size(s)
    s1=s.copy()
    rv=np.sort(np.random.randint(0, d, size=2))
    s1[rv[0]:rv[1]+1]=s1[rv[0]:rv[1]+1][::-1]
    return s1

# okoline N_k, k=1,2,3 koje se koriste u fazi razmrdavanja BVNS metode
# N_1 -> insertion
# N_2 -> swap
# N_3 -> two_opt
def neighborhood(k,s,r):
    if k==1:
        s1=insertion(s)
        while not(capacity_condition(s1,r)):
            s1=insertion(s)
    elif k==2:
        s1=swap(s)
        while not(capacity_condition(s1,r)):
            s1=swap(s)
    else:
        s1=two_opt(s)
        while not(capacity_condition(s1,r)):
            s1=two_opt(s)
    return s1


# crtanje rute
def draw_route(s,r):
    M=r['coords_demands']
    plt.figure(figsize=(7,4), dpi=100)
    plt.plot(M[0,0],M[0,1],'ro',label='depot')
    plt.plot(M[1:,0],M[1:,1],'bo',label='clients')
    ind_0=np.where(s==0)[0]
    d=ind_0.size

    for i in range(0,d-1):
        v=s[ind_0[i]:ind_0[i+1]+1]
        plt.plot(M[v,0],M[v,1])
    v=np.append(s[ind_0[d-1]:],0)
    plt.plot(M[v,0],M[v,1])
    plt.legend()
    plt.show()


#Procedura koja se koristi u fazi lokalnog pretraživanja BVNS-a
#Ista kao LS jedina razlika je što LS_BVNS kao argument prima tekuće rešenje s
# dok LS generiše inicajlno rešenje

def LS_BVNS(s, r, maxIter, C,f_opt):
    f = objective_function(s, C)
    Iter = 0
    br = 0

    while br < maxIter:
        ns = simpgen(s)
        while not (capacity_condition(ns, r)):
            ns = simpgen(s)

        nf = objective_function(ns, C)
        Iter = Iter + 1
        if nf < f:
            br = 0
            s = ns
            f = nf
            if abs(f - f_opt) < 0.5:
                return s, int(f)
        else:
            br = br + 1

    return s, int(f)


#Osnovna metoda promenljivih okolina (eng. Basic Variable Neighborhood Search - BVNS)
#argumenti: r - podaci o instanci,
#           maxIter - maksimalan broj iteracija,
#           C - matrica rastojanja
#           f_opt - opimalna vrednost
#           (korišćenja za break kako bismo manje čekali pri pokretanju aplikacije)
def BVNS(r, maxIter, C,f_opt):
    t = time.time()
    s = initial_solution_bp(r)
    f = objective_function(s, C)
    k = 1
    Iter = 1

    while Iter <= maxIter:
        k = 1
        while k <= 3:
            ns = neighborhood(k, s, r)
            maxIterLS = int(100 + math.log2(Iter ** 100))
            [ns, nf] = LS_BVNS(ns, r, maxIterLS, C,f_opt)
            print('iter:',Iter,'value:',nf)
            Iter = Iter + 1
            if nf < f:
                s = ns
                f = nf
                k = 1
                if abs(f - f_opt) < 0.5:
                    el = time.time() - t
                    print('best value:', f)
                    return s, int(f), el
            else:
                k = k + 1
    el = time.time() - t
    print('best value:', f)
    return s, int(f), el


