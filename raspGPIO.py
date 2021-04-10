import RPi.GPIO as gp
gp.setup(19,gp.OUT)
gp.setup(26,gp.OUT)
gp.setup(16,gp.OUT)
gp.setup(20,gp.OUT)
def GPIOpinconfig(s):
    if(s == "Forward Straight"):
        gp.output(19,True)
        gp.output(26,False)
        gp.output(16,True)
        gp.output(20,False)
    elif(s == "Forward Right"):
        gp.output(19,False)
        gp.output(26,True)
        gp.output(16,True)
        gp.output(20,False)
    elif(s == "Forward Left"):
        gp.output(19,True)
        gp.output(26,False)
        gp.output(16,False)
        gp.output(20,True)
    elif(s == "Reverse"):
        gp.output(19,False)
        gp.output(26,True)
        gp.output(16,False)
        gp.output(20,True)
    else:
        gp.output(19,True)
        gp.output(26,True)
        gp.output(16,True)
        gp.output(20,True)
