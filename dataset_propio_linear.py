import pandas as pd
import numpy as np

def main():
    x1_values, x2_values, x3_values, y_values, noise = [], [], [], [], []
    for i in range(500):
        x1_values.append(np.random.random()*100)
        x2_values.append(np.random.random()*100)
        x3_values.append(np.random.random()*100)
        noise.append(np.random.normal(0, 1))
        y_values.append(3*x1_values[i]+0.5*x2_values[i]+noise[i])
    df = pd.DataFrame({'date': range(1,501), 'x1':x1_values, 'x2':x2_values, 'x3':x3_values, 'y':y_values})
    df.to_csv("input_csv_files/own_dataset_linear.csv", index=False)
if __name__ == '__main__':
    main()