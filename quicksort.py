import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(-3* np.pi,3 * np.pi,0.3)



print x
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)


# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x,y_cos)
plt.plot(x,y_tan)
plt.plot(y_sin,x)
plt.xlabel('x axis lebel')
plt.ylabel('y axis lebel')
plt.title('sine and cosine tan grpah')
plt.legend(['Sin', 'Cos','Tan'])
plt.show()  # You must call plt.show() to make graphics appear.

