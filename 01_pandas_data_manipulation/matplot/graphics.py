import numpy as np
import matplotlib.pyplot as plt


def task_one():
    # 1. Generate the x data points
    # np.linspace creates an array of 400 evenly spaced numbers between -10 and 10.
    # We use 400 points so the resulting curve looks perfectly smooth.
    x = np.linspace(-10, 10, 400)

    # 2. Calculate the y values
    # Notice there is no 'for' loop here!
    y = (x ** 2) * np.sin(x)

    # 3. Create and customize the plot
    plt.figure(figsize=(8, 5))  # Best practice: always define your canvas size explicitly
    plt.plot(x, y, color='blue', label = r'$f(x) = x^2 \cdot \sin(x)$')

    # Adding titles and labels makes the chart professional and readable
    plt.title("Task 1: Function Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.grid(True)
    plt.legend()  # Displays the label we defined in plt.plot()

    # Save and display
    plt.savefig("task1_plot.png")
    plt.show()


def task_two():
    mean = 5
    std_dev = 2
    num_samples = 1000

    # 2. Generate the array of random numbers
    # In Java, you would be writing a for-loop and calling Random.nextGaussian() 1000 times.
    # Here, numpy generates the entire array in a single line!
    data = np.random.normal(mean, std_dev, num_samples)

    # 3. Create the histogram
    plt.figure(figsize=(8, 5))

    # 'bins' controls how many bars the histogram is divided into.
    # 'edgecolor' makes the individual bars easier to see against each other.
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')

    # 4. Customize the plot
    plt.title("Task 2: Normal Distribution Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Adding a grid only on the y-axis makes reading bar heights much easier
    plt.grid(axis='y', alpha=0.75)

    # Save and display
    plt.savefig("task2_histogram.png")
    plt.show()


def task_three():
    hobbies = ['Python & AI Dev', 'Studying Japanese', 'Reading', 'After Effects',
               'University (Java/SQL)']

    # The 'shares' don't strictly have to add up to 100,
    # matplotlib will automatically calculate the percentages
    shares = [25, 30, 15, 10, 20]

    # 'explode' pulls a specific slice out of the pie to highlight it.
    # Let's pull out the 'Python & AI Dev' slice (the first item in the list)
    explode_settings = (0.1, 0, 0, 0, 0)

    # 2. Create the pie chart
    plt.figure(figsize=(8, 8))

    plt.pie(
        shares,
        labels=hobbies,
        explode=explode_settings,
        autopct='%1.1f%%',  # This formats the numbers on the chart to 1 decimal place (e.g., "25.0%")
        startangle=140,  # Rotates the starting point of the pie chart
        shadow=True,  # Adds a cool 3D shadow effect
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']  # Custom hex colors
    )

    # 3. Customize and display
    plt.title("Task 3: My Hobbies Breakdown")

    plt.savefig("task3_pie_chart.png")
    plt.show()


def task_four():
    # 1. Generate normal distribution data for 4 fruits (100 samples each)
    # np.random.normal(mean_weight, standard_deviation, num_samples)
    apples = np.random.normal(150, 10, 100)
    bananas = np.random.normal(120, 8, 100)
    oranges = np.random.normal(200, 15, 100)
    peaches = np.random.normal(140, 12, 100)

    # Put our data arrays and labels into lists so matplotlib can read them together
    fruit_data = [apples, bananas, oranges, peaches]
    fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    # 2. Create the box plot
    plt.figure(figsize=(8, 6))

    # patch_artist=True allows the boxes to be filled with color rather than just being outlines
    box_plot = plt.boxplot(fruit_data, labels=fruit_labels, patch_artist=True)

    # (Optional) Adding some basic colors to the boxes
    colors = ['#ff9999', '#ffe6b3', '#ffcc99', '#ffb3e6']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    # 3. Customize the chart
    plt.title("Task 4: Fruit Mass Distribution")
    plt.ylabel("Mass (grams)")

    # A dashed horizontal grid helps measure the boxes easily
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    plt.savefig("task4_boxplot.png")
    plt.show()


def task_five():
    # 1. Generate 100 random points for x and y
    # A "uniform" distribution means every number between 0 and 1 has an equal chance of being picked.
    x = np.random.uniform(0, 1, 100)
    y = np.random.uniform(0, 1, 100)

    # 2. Create the scatter plot
    plt.figure(figsize=(8, 6))

    # c='green' sets the color
    # alpha=0.6 sets the transparency (0.0 is invisible, 1.0 is solid)
    plt.scatter(x, y, c = 'green', alpha=0.6)

    # 3. Customize the chart with axis labels
    plt.title("Task 5: Uniform Distribution Scatter Plot")
    plt.xlabel("X-Axis (Random [0, 1])")
    plt.ylabel("Y-Axis (Random [0, 1])")

    # Adding a light grid makes scatter plots much easier to read
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save and show
    plt.savefig("task5_scatter.png")
    plt.show()


def task_six():
    # 1. Generate the x data points (using the same -10 to 10 range as Task 1)
    x = np.linspace(-10, 10, 400)

    # 2. Calculate the y values for all three functions
    # In Java, this would be three separate for-loops. Here, it's just pure math!
    f_x = np.sin(x)
    g_x = np.cos(x)
    h_x = np.sin(x) + np.cos(x)

    # 3. Create the plot and define the canvas size
    plt.figure(figsize=(10, 6))

    # 4. Plot each function.
    plt.plot(x, f_x, color='blue', label=r'$f(x) = \sin(x)$')
    plt.plot(x, g_x, color='red', label=r'$g(x) = \cos(x)$')

    plt.plot(x, h_x, color='green', linestyle='--', label=r'$h(x) = \sin(x) + \cos(x)$')

    plt.title("Task 6: Multiple Trigonometric Functions")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.grid(True)
    plt.legend()

    plt.savefig("task6_trig_functions.png")
    plt.show()


if __name__ == "__main__":
    task_one()
    task_two()
    task_three()
    task_four()
    task_five()
    task_six()