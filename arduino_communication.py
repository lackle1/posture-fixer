import time
import serial.tools.list_ports

# get all communication ports on laptop and store those in a list
ports = serial.tools.list_ports.comports()
# we are using serial communications
serialInst = serial.Serial()
portsList = []

for x in ports:
    portsList.append(str(x))
    print(str(x))


com = input("Select com port for Arduino #: ")

# check if com is valid
for i in range(len(portsList)):
    if portsList[i].startswith("COM" + str(com)):
        use = "COM" + str(com)
        print(use)

#set up serial port
serialInst.baudrate = 9600
serialInst.port = use
serialInst.open()
time.sleep(2)

def doSomething(prediction):
    if prediction == 0:
        command = 'option_one'
        print(command)
        serialInst.write(command.encode('utf-8'))
        # print(command.encode('utf-8'))

    elif prediction == 1:
        command = 'option_two'
        print(command)
        serialInst.write(command.encode('utf-8'))

    elif prediction == 2:
        command = 'option_three'
        print(command)
        serialInst.write(command.encode('utf-8'))

    # elif prediction == 3:
    #     command = 'option_four'
    #     print(command)
    #     serialInst.write(command.encode('utf-8'))

