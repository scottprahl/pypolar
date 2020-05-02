# pylint: disable=invalid-name
"""
Simple interface to Gaerter 116A Ellipsometer.

Scott Prahl
Apr 2020
"""
import os
import sys
import time
import glob
import serial
import serial.tools.list_ports
import numpy as np

__all__ = ('get_reflectance',
           'avg_reflectance',
           'current_serial_ports',
           'connect_to_ellipsometer',
           'save_data_with_time_stamp',
           'save_data_with_name',
           'read_data_with_name',
           )


def get_reflectance(conn):
    """
    Obtain one analyzer revolution worth of reflectance values.

    Trigger the Arduino attached to the ellipsometer and obtain
    readings at each of 72 analyzer orientations.

    The idea is pretty simple. Just send a byte to the Arduino and
    it will send 144=72*2 bytes back.
    returns:
        an array of 72 integers
    """
#    conn.flush()                   # gets rid of stale bytes
    time.sleep(1)                  # wait for connection to stablize

    N = 72
    conn.write(b'1')               # trigger the Arduino
    a = bytearray(conn.read(2*N))  # read 144 bytes
#    conn.close()
    y = np.zeros(N)
    y = [(a[2*i+1] << 8) + a[2*i] for i in range(N)]  # assemble the bytes
    return np.array(y)


def avg_reflectance(conn, num_samples=10):
    """
    collect multiple samples and return the sum of all of them.

    returns:
        an array of 72 integers
    """
    total = get_reflectance(conn)

    for _i in range(1, num_samples):
        total += get_reflectance(conn)
    return total


def current_serial_ports():
    """
    Return a string listing all current serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    ports = list(serial.tools.list_ports.comports())
    s = "%35s  %s\n" % ('portname','info')
    for p in ports:
        s += "%35s  %s\n" %(p[0],p[1])
    return s


def list_serial_ports_slow():
    """
    Lists all the current serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    print(ports)
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def connect_to_ellipsometer(usb_serial_port_id):

    # 38400 is the fastest baud rate that works
    baud_rate = 38400

    x=os.system("ls " + usb_serial_port_id)

    if x!=0:
        print("ERROR: serial port '%s' is disconnected" % usb_serial_port_id)
        return None
    
    try:
        conn = serial.Serial(usb_serial_port_id, baud_rate, timeout=2)

    except serial.serialutil.SerialException:
        print('Error: No serial connection to Arduino established!')
        print('       Correct the serial port id above')
        print('       Use list_serial_ports() to find possible names')
        return None

    return conn

def save_data_with_time_stamp(signal):
    """
    Save signal in time-stamped file.

    Creates a filename using the current time.

    Args:
        signal: array of ellipsometer readings
    Returns:
        filename of the file created
    """
    t = time.localtime()
    time_stamp_name = time.strftime('lab3-%Y-%m-%d-%H-%M-%S.txt', t)
    np.savetxt(time_stamp_name, signal, fmt="%0.f", newline=', ')
    print("saved data to file named '%s'" % time_stamp_name)
    return time_stamp_name


def save_data_with_name(signal, basename, theta_i, P, QWP=False):
    """
    Save signal to file with ellipsometer setting in the filename.

    Creates a filename based on the settings and saves the array signal to it.

    Args:
        signal: array of ellipsometer readings
        basename: string that file begins with
        theta_i: angle of incidence
        P: polarizer angle
        QWP: boolean indicating if QWP is present
    Returns:
        filename of the file created
    """

    t = time.localtime()
    time_stamp_name = time.strftime('%Y-%m-%d-%H-%M-%S.txt', t)

    filename = basename + ", I=%.1f°"%theta_i + ", P=%.1f°"%P
    filename = filename + ", QWP=%r, "%QWP + time_stamp_name
    np.savetxt(filename, signal, fmt="%0.f", newline=', ')

    print("saved data to file named '%s'" % filename)
    return filename


def read_data_with_name(filename):
    """
    Read a text file into an array.

    This does the opposite of `save_data_with_time_stamp()` or `save_data_with_name()`
    The data is read and returned in a 72 element array.
    """
    x=os.system("ls " + filename)

    if x!=0:
        print("ERROR: file '%s' does not exist" % filename)
        return None

    signal = np.genfromtxt(filename, delimiter=', ')
    signal = np.delete(signal, -1, 0) # drop the last NaN value
    return signal
