from tkinter import *
from caption_generater import *
from tkinter import ttk
from PIL import ImageTk, Image

window = Tk()
window.title("SATELITE IMAGE CAPTIONING")
window.geometry("700x400")
label1 =Label(window, text ="Image Path")
label1.grid(row =0, column =0)
label1.config(font=("Courier", 15))

txtbx =Entry(window)
txtbx.grid(row =0, column =1)

answer_label =Label(window, text ="---")
answer_label.grid(row =1, column =2)

def clickExitButton():
        exit()

def fun():
    if (txtbx.get() != ""):
        pth = txtbx.get() 
        ans = generate_caption(pth)
        answer_label.configure(text =ans)
        answer_label.config(font=("Courier", 15))


        
generate =Button(window, text="Generate Caption", command= fun)
generate.grid(row =1, column =0, columnspan =2)
#generate.config(font=("Courier", 15))

ext =Button(window, text="Exit", command= clickExitButton)
ext.grid(row =2, column =0, columnspan =2)
#ext.config(font=("Courier", 15))

window.mainloop()




