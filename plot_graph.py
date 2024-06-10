import matplotlib.pyplot as plt

# File path
file_path = 'out.txt'  # Replace with your actual file path

# Lists to store data points
x_values = []
y_values = []

# Read data from file
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        line = line.strip()
        if line:
            value = line
            x_values.append(index)
            y_values.append(float(value))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Data Points', markersize=1)
plt.title('Plot of Data Points')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
