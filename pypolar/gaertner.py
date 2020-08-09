# pylint: disable=invalid-name
"""
Simple interface to Gaerter 116A Ellipsometer.

This is intended to be used with the computer connected to an
Arduino controller that interfaces with the Gaerter ellipsometer.

Scott Prahl
Apr 2020
"""
import os
import time
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


def current_serial_ports():
    """Return a string listing all current serial port names."""
    ports = list(serial.tools.list_ports.comports())
    s = "%35s  %s\n" % ('portname', 'info')
    for p in ports:
        s += "%35s  %s\n" %(p[0], p[1])
    return s


def connect_to_ellipsometer(usb_serial_port_id):
    """
    Establish a serial connection with the ellipsometer.

    Args:
        usb_serial_port_id: name of serial port
    Returns
        Serial connection object
    """
    # 38400 is the fastest baud rate that works
    baud_rate = 38400

    x = os.system("ls " + usb_serial_port_id)

    if x != 0:
        print("ERROR: serial port '%s' is disconnected" % usb_serial_port_id)
        return None

    try:
        conn = serial.Serial(usb_serial_port_id, baud_rate, timeout=2)

    except serial.serialutil.SerialException:
        print('Error: No serial connection to Arduino established!')
        print('       Correct the serial port id above')
        print('       Use list_serial_ports() to find possible names')
        return None

    time.sleep(2)                  # wait for connection to stablize
    return conn


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
    N = 72
    conn.write(b'1')               # trigger the Arduino
    a = bytearray(conn.read(2*N))  # read 144 bytes

    if len(a) != 144:
        print("Bad number of bytes read (%d != 144)" % len(a))
        print("Make sure ellipsometer is rotating!")
        return np.zeros(72, dtype=np.int64)

    # assume Arduino and host have same endianness.  Reassemble as 16 ints
    y = [(a[2*i] << 8) + a[2*i+1] for i in range(N)]
    return np.array(y)


def avg_reflectance(conn, num_samples=10):
    """
    Collect multiple samples and return the sum of all of them.

    returns:
        an array of 72 integers
    """
    total = get_reflectance(conn)

    for _i in range(1, num_samples):
        total += get_reflectance(conn)

    return total // num_samples


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
        signal:   array of ellipsometer readings
        basename: string starting filename
        theta_i:  angle of incidence               [radians]
        P:        polarizer angle                  [radians]
        QWP:      True if QWP is present
    Returns:
        filename of the file created
    """
    I = np.degrees(theta_i)
    PP = np.degrees(P)

    t = time.localtime()
    time_stamp_name = time.strftime('%Y-%m-%d-%H-%M-%S.txt', t)

    filename = "%s, I=%.1f°, P=%.1f°, QWP=%r, " % (basename, I, PP, QWP)
    filename = filename + time_stamp_name
    np.savetxt(filename, signal, fmt="%0.f", newline=', ')

    print("saved data to file named '%s'" % filename)
    return filename


def read_data_with_name(filename):
    """
    Read a text file into an array.

    This does the opposite of `save_data_with_time_stamp()` or `save_data_with_name()`
    The data is read and returned in a 72 element array.
    """
    x = os.system("ls " + filename)

    if x != 0:
        print("ERROR: file '%s' does not exist" % filename)
        return None

    signal = np.genfromtxt(filename, delimiter=', ')
    signal = np.delete(signal, -1, 0) # drop the last NaN value
    return signal
