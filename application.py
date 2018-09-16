import tkinter as tk
import numpy as np
import cvrp_func
from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = tk.Tk()
root.title('CVRP')
canvas1 = tk.Canvas(root, width=800, height=350)
canvas1.pack()
imagepath1 = './background.jpg'
image1 = ImageTk.PhotoImage(file=imagepath1)
canvas1.create_image(400, 200, image=image1)


button1 = tk.Button(root, text='Exit application',
                    command=root.destroy)
canvas1.create_window(720, 330,
                      window=button1)
label_t=canvas1.create_text((100,100), text="Do you need an optimal route?")
label_t1=canvas1.create_text((188,117), text="Choose an instance, select a method and click the button below.")
label_t2=canvas1.create_text((158,134), text="You'll get optimal or near optimal results. Good luck!")
label_t3=canvas1.create_text((400,330), text="")

label1=canvas1.create_text((130, 220), text="Instance:")
tkvar1 = tk.StringVar(root)
choices1= {'P-n16-k8', 'P-n19-k2', 'P-n20-k2','P-n22-k2','P-n22-k8','E-n22-k4','E-n23-k3','E-n30-k3'}
tkvar1.set('P-n16-k8')
mb1=tk.OptionMenu(canvas1,tkvar1,*choices1)
canvas1.create_window(211,220,
                      window=mb1)

label1=canvas1.create_text((95, 260), text="Metaheuristic method:")
tkvar2 = tk.StringVar(root)

choices2= {'LS', 'VNS', 'SA'}
tkvar2.set('VNS')
mb2=tk.OptionMenu(canvas1,tkvar2,*choices2)
canvas1.create_window(200,260,
                      window=mb2)

figure2 = Figure(figsize=(5, 4), dpi=80)
subplot2 = figure2.add_subplot(111)
figure1 = Figure(figsize=(5, 4), dpi=80 )
subplot1 = figure1.add_subplot(111)

def start_method():
    canvas1.itemconfigure(label_t3, text="Wait 'till I'm done...")
    canvas1.update()
    subplot1.cla()
    subplot2.cla()
    global  x1
    global x2
    x1 = tkvar1.get()
    x2 = tkvar2.get()
    r = cvrp_func.load(x1)
    C = cvrp_func.dist_matrix(r['coords_demands'])
    [f_opt, s_opt] = cvrp_func.get_opt_results(x1, r)

    if x2=='VNS':
        [s1, f, el] = cvrp_func.BVNS(r, 100, C,f_opt)
    elif x2=='SA':
        [s1,f, el]=cvrp_func.SA(30, 0.1, 0.98, 0.1, C, r,f_opt)
    else:
        [s1,f, el]=cvrp_func.LS(r, 5000, C,f_opt)


    s=cvrp_func.fix_sol(s1)
    f=int(f)
    M=r['coords_demands']

    subplot1.plot(M[0, 0], M[0, 1], 'ro',label='depot')
    subplot1.plot(M[1:, 0], M[1:, 1], 'bo', label='clients')
    ind_0 = np.where(s_opt == 0)[0]
    d = ind_0.size
    for i in range(0, d - 1):
        v = s_opt[ind_0[i]:ind_0[i + 1] + 1]
        subplot1.plot(M[v, 0], M[v, 1])
    v = np.append(s_opt[ind_0[d - 1]:], 0)
    subplot1.plot(M[v, 0], M[v, 1])
    subplot1.legend()
    subplot1.set_title("Optimal solution - cost "+str(f_opt))


    subplot2.plot(M[0, 0], M[0, 1], 'ro', label='depot')
    subplot2.plot(M[1:, 0], M[1:, 1], 'bo', label='clients')
    ind_0 = np.where(s == 0)[0]
    d = ind_0.size
    for i in range(0, d - 1):
        v = s[ind_0[i]:ind_0[i + 1] + 1]
        subplot2.plot(M[v, 0], M[v, 1])
    v = np.append(s[ind_0[d - 1]:], 0)
    subplot2.plot(M[v, 0], M[v, 1])
    subplot2.legend()
    subplot2.set_title("Heuristic solution - cost "+str(f))

    bar1.draw()
    bar2.draw()

    canvas1.itemconfigure(label_t3, text="I'm done doing!")

bar1 = FigureCanvasTkAgg(figure1, root)
bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
bar2 = FigureCanvasTkAgg(figure2, root)
bar2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=0)


button2 = tk.Button(root, text='Click to view results ',
                    command=start_method)
canvas1.create_window(95, 330, window=button2)

root.mainloop()
